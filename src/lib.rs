use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    fmt,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard},
};

use futures::StreamExt;
use rig::{
    OneOrMany,
    agent::{Agent, MultiTurnStreamItem, PromptHook, StreamingError},
    completion::{CompletionModel, GetTokenUsage, PromptError},
    message::{AssistantContent, Message, ToolCall, ToolResult, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt},
    wasm_compat::WasmCompatSend,
};
use tokio::{
    sync::{
        broadcast,
        mpsc::{self, error::TryRecvError},
    },
    task::JoinHandle,
};

pub type PersistError = Box<dyn Error + Send + Sync + 'static>;

type PersistenceFuture = Pin<Box<dyn Future<Output = Result<(), PersistError>> + Send>>;
type PersistenceHook = Arc<dyn Fn(Vec<Message>) -> PersistenceFuture + Send + Sync>;
type SharedMessages = Arc<Mutex<Vec<Message>>>;
const EVENT_BUFFER_SIZE: usize = 1024;

fn lock_messages(messages: &SharedMessages) -> MutexGuard<'_, Vec<Message>> {
    match messages.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A small harness around `rig::agent::Agent` that owns the queue between prompts.
pub struct AgentLoop<M, P = ()>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    agent: Arc<Agent<M, P>>,
    max_turns: usize,
    persistence_hook: Option<PersistenceHook>,
}

impl<M, P> AgentLoop<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    pub fn new(agent: Agent<M, P>) -> Self {
        Self {
            agent: Arc::new(agent),
            max_turns: 100,
            persistence_hook: None,
        }
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Persist committed messages at the end of each turn before the loop advances.
    ///
    /// The hook receives only the messages that should be appended. For interrupted
    /// turns this is limited to completed tool-call roundtrips; if nothing was
    /// committed, the hook is not called.
    pub fn with_persistence_hook<F, Fut, E>(mut self, hook: F) -> Self
    where
        F: Fn(Vec<Message>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        self.persistence_hook = Some(Arc::new(move |messages| {
            let future = hook(messages);
            Box::pin(async move { future.await.map_err(|err| Box::new(err) as PersistError) })
        }));
        self
    }
}

impl<M, P> AgentLoop<M, P>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
{
    /// Start the loop with an initial user prompt and return a control handle.
    pub fn prompt(&self, prompt: impl Into<Message>) -> AgentLoopHandle<M::StreamingResponse> {
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (events_tx, _) = broadcast::channel(EVENT_BUFFER_SIZE);
        let state = Arc::new(Mutex::new(Vec::new()));
        let runner = Runner::new(
            self.agent.clone(),
            self.max_turns,
            self.persistence_hook.clone(),
            events_tx.clone(),
            state.clone(),
            prompt.into(),
        );
        let task = tokio::spawn(async move { runner.run(commands_rx).await });

        AgentLoopHandle {
            commands_tx,
            events_tx,
            state,
            task,
        }
    }
}

/// A running agent loop.
pub struct AgentLoopHandle<R>
where
    R: Clone,
{
    commands_tx: mpsc::UnboundedSender<Command>,
    events_tx: broadcast::Sender<AgentLoopEvent<R>>,
    state: SharedMessages,
    task: JoinHandle<Result<AgentLoopResult, AgentLoopError>>,
}

impl<R> AgentLoopHandle<R>
where
    R: Clone,
{
    /// Subscribe to harness lifecycle events and raw Rig stream items.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentLoopEvent<R>> {
        self.events_tx.subscribe()
    }

    /// Return a clone of the committed in-memory message history.
    pub fn state(&self) -> Vec<Message> {
        lock_messages(&self.state).clone()
    }

    /// Wait for the loop to become idle, abort, or error.
    pub async fn wait(self) -> Result<AgentLoopResult, AgentLoopError> {
        match self.task.await {
            Ok(result) => result,
            Err(err) => Err(AgentLoopError::TaskJoin(err)),
        }
    }

    /// Queue a message to run before follow-ups after the current prompt settles.
    pub fn steer(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        let message = message.into();
        self.send(Command::Steer(message.clone()))?;
        self.emit(AgentLoopEvent::Queued {
            kind: QueueKind::Steer,
            message: Some(message),
        });
        Ok(())
    }

    /// Queue a message to run after the current prompt would otherwise leave the loop idle.
    pub fn follow_up(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        let message = message.into();
        self.send(Command::FollowUp(message.clone()))?;
        self.emit(AgentLoopEvent::Queued {
            kind: QueueKind::FollowUp,
            message: Some(message),
        });
        Ok(())
    }

    /// Stop the current prompt, keep committed tool results, then run this message next.
    pub fn interrupt(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        let message = message.into();
        self.send(Command::Interrupt(message.clone()))?;
        self.emit(AgentLoopEvent::Queued {
            kind: QueueKind::Interrupt,
            message: Some(message),
        });
        Ok(())
    }

    /// Stop the loop. Completed tool results from the current prompt are kept.
    pub fn abort(&self) -> Result<(), AgentLoopError> {
        self.send(Command::Abort)?;
        self.emit(AgentLoopEvent::Queued {
            kind: QueueKind::Abort,
            message: None,
        });
        Ok(())
    }

    fn send(&self, command: Command) -> Result<(), AgentLoopError> {
        self.commands_tx
            .send(command)
            .map_err(|_| AgentLoopError::CommandChannelClosed)
    }

    fn emit(&self, event: AgentLoopEvent<R>) {
        let _ = self.events_tx.send(event);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EndReason {
    /// The prompt and all queued steering/follow-up messages completed.
    Idle,
    /// The caller aborted the loop.
    Aborted,
}

#[derive(Clone, Debug)]
pub struct AgentLoopResult {
    pub end_reason: EndReason,
    pub history: Vec<Message>,
    pub last_response: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueueKind {
    Steer,
    FollowUp,
    Interrupt,
    Abort,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AgentLoopEvent<R> {
    LoopStarted,
    Queued {
        kind: QueueKind,
        message: Option<Message>,
    },
    TurnStarted {
        prompt: Message,
    },
    Rig(MultiTurnStreamItem<R>),
    TurnCommitted {
        messages: Vec<Message>,
    },
    TurnInterrupted {
        messages: Vec<Message>,
    },
    TurnAborted {
        messages: Vec<Message>,
    },
    Persisted {
        messages: Vec<Message>,
    },
    LoopEnded {
        end_reason: EndReason,
    },
}

#[derive(Debug)]
pub enum AgentLoopError {
    CommandChannelClosed,
    Persist(PersistError),
    Rig(StreamingError),
    TaskJoin(tokio::task::JoinError),
}

impl fmt::Display for AgentLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandChannelClosed => f.write_str("agent loop command channel is closed"),
            Self::Persist(err) => write!(f, "persistence hook failed: {err}"),
            Self::Rig(err) => write!(f, "{err}"),
            Self::TaskJoin(err) => write!(f, "{err}"),
        }
    }
}

impl Error for AgentLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Persist(err) => Some(err.as_ref()),
            Self::Rig(err) => Some(err),
            Self::TaskJoin(err) => Some(err),
            Self::CommandChannelClosed => None,
        }
    }
}

impl From<StreamingError> for AgentLoopError {
    fn from(err: StreamingError) -> Self {
        Self::Rig(err)
    }
}

enum Command {
    Steer(Message),
    FollowUp(Message),
    Interrupt(Message),
    Abort,
}

struct Runner<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    agent: Arc<Agent<M, P>>,
    max_turns: usize,
    persistence_hook: Option<PersistenceHook>,
    events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
    state: SharedMessages,
    immediate: VecDeque<Message>,
    steering: VecDeque<Message>,
    follow_ups: VecDeque<Message>,
    last_response: Option<String>,
}

impl<M, P> Runner<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn new(
        agent: Arc<Agent<M, P>>,
        max_turns: usize,
        persistence_hook: Option<PersistenceHook>,
        events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
        state: SharedMessages,
        prompt: Message,
    ) -> Self {
        Self {
            agent,
            max_turns,
            persistence_hook,
            events_tx,
            state,
            immediate: VecDeque::from([prompt]),
            steering: VecDeque::new(),
            follow_ups: VecDeque::new(),
            last_response: None,
        }
    }

    fn next_prompt(&mut self) -> Option<Message> {
        self.immediate
            .pop_front()
            .or_else(|| self.steering.pop_front())
            .or_else(|| self.follow_ups.pop_front())
    }

    fn handle_idle_command(&mut self, command: Command) -> CommandAction {
        match command {
            Command::Steer(message) => self.steering.push_back(message),
            Command::FollowUp(message) => self.follow_ups.push_back(message),
            Command::Interrupt(message) => self.immediate.push_front(message),
            Command::Abort => return CommandAction::Abort,
        }

        CommandAction::Continue
    }

    fn drain_ready_commands(
        &mut self,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
    ) -> CommandAction {
        loop {
            match commands_rx.try_recv() {
                Ok(command) => {
                    if self.handle_idle_command(command) == CommandAction::Abort {
                        return CommandAction::Abort;
                    }
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                    return CommandAction::Continue;
                }
            }
        }
    }
}

impl<M, P> Runner<M, P>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
{
    async fn run(
        mut self,
        mut commands_rx: mpsc::UnboundedReceiver<Command>,
    ) -> Result<AgentLoopResult, AgentLoopError> {
        self.emit(AgentLoopEvent::LoopStarted);

        loop {
            if self.drain_ready_commands(&mut commands_rx) == CommandAction::Abort {
                return Ok(self.finish(EndReason::Aborted));
            }

            let Some(prompt) = self.next_prompt() else {
                return Ok(self.finish(EndReason::Idle));
            };

            match self.run_prompt(prompt, &mut commands_rx).await? {
                PromptAction::Continue => {}
                PromptAction::Abort => return Ok(self.finish(EndReason::Aborted)),
            }
        }
    }

    async fn run_prompt(
        &mut self,
        prompt: Message,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
    ) -> Result<PromptAction, AgentLoopError> {
        self.emit(AgentLoopEvent::TurnStarted {
            prompt: prompt.clone(),
        });

        let base_history = self.history_snapshot();
        let mut partial_turn = PartialTurn::default();
        let mut commands_closed = false;
        let mut stream = self
            .agent
            .stream_prompt(prompt.clone())
            .with_history(base_history.clone())
            .multi_turn(self.max_turns)
            .await;

        loop {
            tokio::select! {
                command = commands_rx.recv(), if !commands_closed => {
                    let Some(command) = command else {
                        commands_closed = true;
                        continue;
                    };

                    match command {
                        Command::Steer(message) => self.steering.push_back(message),
                        Command::FollowUp(message) => self.follow_ups.push_back(message),
                        Command::Interrupt(message) => {
                            let messages = partial_turn.append_messages(&prompt).unwrap_or_default();
                            self.commit_append(messages, CommitEvent::TurnInterrupted).await?;
                            self.immediate.push_front(message);
                            return Ok(PromptAction::Continue);
                        }
                        Command::Abort => {
                            let messages = partial_turn.append_messages(&prompt).unwrap_or_default();
                            self.commit_append(messages, CommitEvent::TurnAborted).await?;
                            return Ok(PromptAction::Abort);
                        }
                    }
                }
                item = stream.next() => {
                    let Some(item) = item else {
                        return Ok(PromptAction::Continue);
                    };

                    match item {
                        Ok(item) => {
                            self.emit(AgentLoopEvent::Rig(item.clone()));
                            self.handle_stream_item(item, &mut partial_turn).await?;
                        }
                        Err(err) => {
                            if let Some(history) = history_from_error(&err) {
                                self.commit_recovered_history(&base_history, history).await?;
                            }
                            return Err(err.into());
                        }
                    }
                }
            }
        }
    }

    async fn handle_stream_item(
        &mut self,
        item: MultiTurnStreamItem<M::StreamingResponse>,
        partial_turn: &mut PartialTurn,
    ) -> Result<(), AgentLoopError> {
        match item {
            MultiTurnStreamItem::StreamAssistantItem(item) => {
                partial_turn.note_assistant_item(item)
            }
            MultiTurnStreamItem::StreamUserItem(item) => partial_turn.note_user_item(item),
            MultiTurnStreamItem::FinalResponse(final_response) => {
                self.last_response = Some(final_response.response().to_string());
                if let Some(messages) = final_response.history() {
                    self.commit_append(messages.to_vec(), CommitEvent::TurnCommitted)
                        .await?;
                }
            }
            MultiTurnStreamItem::CompletionCall(_) => {}
            _ => {}
        }

        Ok(())
    }

    async fn commit_append(
        &mut self,
        messages: Vec<Message>,
        event: CommitEvent,
    ) -> Result<(), AgentLoopError> {
        if messages.is_empty() {
            self.emit(event.into_agent_event(messages));
            return Ok(());
        }

        if let Some(hook) = &self.persistence_hook {
            hook(messages.clone())
                .await
                .map_err(AgentLoopError::Persist)?;
            self.emit(AgentLoopEvent::Persisted {
                messages: messages.clone(),
            });
        }

        lock_messages(&self.state).extend(messages.clone());
        self.emit(event.into_agent_event(messages));
        Ok(())
    }

    async fn commit_recovered_history(
        &mut self,
        base_history: &[Message],
        recovered_history: Vec<Message>,
    ) -> Result<(), AgentLoopError> {
        if recovered_history.len() < base_history.len() {
            *lock_messages(&self.state) = recovered_history;
            return Ok(());
        }

        let append = recovered_history[base_history.len()..].to_vec();
        self.commit_append(append, CommitEvent::TurnCommitted).await
    }

    fn finish(self, end_reason: EndReason) -> AgentLoopResult {
        self.emit(AgentLoopEvent::LoopEnded {
            end_reason: end_reason.clone(),
        });

        AgentLoopResult {
            end_reason,
            history: self.history_snapshot(),
            last_response: self.last_response,
        }
    }

    fn emit(&self, event: AgentLoopEvent<M::StreamingResponse>) {
        let _ = self.events_tx.send(event);
    }

    fn history_snapshot(&self) -> Vec<Message> {
        lock_messages(&self.state).clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitEvent {
    TurnCommitted,
    TurnInterrupted,
    TurnAborted,
}

impl CommitEvent {
    fn into_agent_event<R>(self, messages: Vec<Message>) -> AgentLoopEvent<R> {
        match self {
            Self::TurnCommitted => AgentLoopEvent::TurnCommitted { messages },
            Self::TurnInterrupted => AgentLoopEvent::TurnInterrupted { messages },
            Self::TurnAborted => AgentLoopEvent::TurnAborted { messages },
        }
    }
}

#[derive(Default)]
struct PartialTurn {
    pending_tool_calls: HashMap<String, ToolCall>,
    completed_tool_calls: Vec<ToolCall>,
    completed_tool_results: Vec<ToolResult>,
}

impl PartialTurn {
    fn note_assistant_item<R>(&mut self, item: StreamedAssistantContent<R>)
    where
        R: Clone,
    {
        if let StreamedAssistantContent::ToolCall {
            internal_call_id,
            tool_call,
        } = item
        {
            self.pending_tool_calls.insert(internal_call_id, tool_call);
        }
    }

    fn note_user_item(&mut self, item: StreamedUserContent) {
        let StreamedUserContent::ToolResult {
            internal_call_id,
            tool_result,
        } = item;

        let Some(tool_call) = self.pending_tool_calls.remove(&internal_call_id) else {
            return;
        };

        self.completed_tool_calls.push(tool_call);
        self.completed_tool_results.push(tool_result);
    }

    fn append_messages(&self, prompt: &Message) -> Option<Vec<Message>> {
        if self.completed_tool_calls.is_empty() {
            return None;
        }

        let assistant_content = OneOrMany::many(
            self.completed_tool_calls
                .iter()
                .cloned()
                .map(AssistantContent::ToolCall)
                .collect::<Vec<_>>(),
        )
        .ok()?;
        let user_content = OneOrMany::many(
            self.completed_tool_results
                .iter()
                .cloned()
                .map(UserContent::ToolResult)
                .collect::<Vec<_>>(),
        )
        .ok()?;

        Some(vec![
            prompt.clone(),
            Message::Assistant {
                id: None,
                content: assistant_content,
            },
            Message::User {
                content: user_content,
            },
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandAction {
    Continue,
    Abort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptAction {
    Continue,
    Abort,
}

fn history_from_error(err: &StreamingError) -> Option<Vec<Message>> {
    let StreamingError::Prompt(err) = err else {
        return None;
    };

    match err.as_ref() {
        PromptError::MaxTurnsError { chat_history, .. } => Some(chat_history.as_ref().clone()),
        PromptError::PromptCancelled { chat_history, .. } => Some(chat_history.clone()),
        PromptError::UnknownToolCall { chat_history, .. } => Some(chat_history.as_ref().clone()),
        PromptError::CompletionError(_)
        | PromptError::ToolError(_)
        | PromptError::ToolServerError(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::{
        agent::AgentBuilder,
        message::UserContent,
        streaming::{StreamedAssistantContent, ToolCallDeltaContent},
        test_utils::{MockAddTool, MockCompletionModel, MockStreamEvent},
    };
    use std::sync::{Arc, Mutex};

    #[tokio::test(flavor = "current_thread")]
    async fn follow_up_keeps_loop_running_after_first_response() {
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("first"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("follow"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent);

        let handle = agent_loop.prompt(Message::user("start"));
        handle.follow_up(Message::user("follow-up")).unwrap();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(result.last_response.as_deref(), Some("follow"));
        assert_eq!(model.request_count(), 2);
        assert_eq!(user_prompts(&model), vec!["start", "follow-up"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn steering_runs_before_follow_ups() {
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("first"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("steered"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("followed"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent);

        let handle = agent_loop.prompt(Message::user("start"));
        handle.follow_up(Message::user("follow-up")).unwrap();
        handle.steer(Message::user("steer")).unwrap();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(model.request_count(), 3);
        assert_eq!(user_prompts(&model), vec!["start", "steer", "follow-up"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn abort_returns_aborted_end_reason() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("unused"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);

        let handle = agent_loop.prompt(Message::user("start"));
        handle.abort().unwrap();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Aborted);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn persistence_hook_receives_messages_to_append() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let persisted = Arc::new(Mutex::new(Vec::<Vec<Message>>::new()));
        let seen = persisted.clone();
        let agent_loop = AgentLoop::new(agent).with_persistence_hook(move |messages| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push(messages);
                Ok::<(), std::io::Error>(())
            }
        });

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        let persisted = persisted.lock().unwrap();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].len(), 2);
        assert_eq!(first_user_text(&persisted[0][0]).as_deref(), Some("start"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn persistence_hook_error_stops_the_loop() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent).with_persistence_hook(|_messages| async {
            Err(std::io::Error::new(std::io::ErrorKind::Other, "boom"))
        });

        let err = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .expect_err("persist failure should fail the run");

        assert!(matches!(err, AgentLoopError::Persist(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subscribe_forwards_rig_stream_items_with_tool_deltas() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("call_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("call_1", "internal_1", r#"{"x":1,"y":2}"#),
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::Rig(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        content: ToolCallDeltaContent::Name(name),
                        ..
                    }
                )) if name == "add"
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::Rig(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        content: ToolCallDeltaContent::Delta(arguments),
                        ..
                    }
                )) if arguments == r#"{"x":1,"y":2}"#
            )
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subscribe_emits_queue_and_lifecycle_events() {
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("first"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("follow"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        handle.follow_up(Message::user("follow-up")).unwrap();
        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::Queued {
                    kind: QueueKind::FollowUp,
                    ..
                }
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(event, AgentLoopEvent::TurnCommitted { messages } if !messages.is_empty())
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::LoopEnded {
                    end_reason: EndReason::Idle
                }
            )
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handle_state_returns_committed_messages_while_running() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        loop {
            match events.recv().await.unwrap() {
                AgentLoopEvent::TurnCommitted { messages } => {
                    assert_eq!(messages.len(), 2);
                    let state = handle.state();
                    assert_eq!(state.len(), 2);
                    assert_eq!(first_user_text(&state[0]).as_deref(), Some("start"));
                    break;
                }
                AgentLoopEvent::LoopEnded { .. } => panic!("loop ended before commit event"),
                _ => {}
            }
        }

        let result = handle.wait().await.unwrap();
        assert_eq!(result.history.len(), 2);
    }

    #[test]
    fn partial_turn_commits_completed_tool_results() {
        let mut partial = PartialTurn::default();
        let tool_call = ToolCall::new(
            "call_1".to_string(),
            rig::message::ToolFunction::new("echo".to_string(), serde_json::json!({"text": "hi"})),
        );
        let tool_result = ToolResult {
            id: "call_1".to_string(),
            call_id: None,
            content: rig::message::ToolResultContent::from_tool_output("echoed".to_string()),
        };

        partial.note_assistant_item(
            StreamedAssistantContent::<rig::test_utils::MockResponse>::ToolCall {
                internal_call_id: "internal_1".to_string(),
                tool_call,
            },
        );
        partial.note_user_item(StreamedUserContent::ToolResult {
            internal_call_id: "internal_1".to_string(),
            tool_result,
        });

        let messages = partial
            .append_messages(&Message::user("start"))
            .expect("completed tool results should produce history");

        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[1], Message::Assistant { .. }));
        assert!(
            matches!(&messages[2], Message::User { content } if matches!(content.first(), UserContent::ToolResult(_)))
        );
    }

    fn user_prompts(model: &MockCompletionModel) -> Vec<String> {
        model
            .requests()
            .into_iter()
            .map(|request| {
                request
                    .chat_history
                    .iter()
                    .filter_map(first_user_text)
                    .last()
                    .unwrap_or_default()
            })
            .collect()
    }

    fn first_user_text(message: &Message) -> Option<String> {
        let Message::User { content } = message else {
            return None;
        };

        content.iter().find_map(|item| match item {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
    }

    fn drain_events<R: Clone>(
        events: &mut broadcast::Receiver<AgentLoopEvent<R>>,
    ) -> Vec<AgentLoopEvent<R>> {
        let mut drained = Vec::new();
        while let Ok(event) = events.try_recv() {
            drained.push(event);
        }
        drained
    }
}
