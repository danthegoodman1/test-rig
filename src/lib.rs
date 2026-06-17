use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    fmt,
    sync::Arc,
};

use futures::StreamExt;
use rig::{
    OneOrMany,
    agent::{Agent, FinalResponse, MultiTurnStreamItem, PromptHook, StreamingError},
    completion::{CompletionModel, GetTokenUsage, PromptError},
    message::{AssistantContent, Message, ToolCall, ToolResult, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt},
    wasm_compat::WasmCompatSend,
};
use tokio::{
    sync::mpsc::{self, error::TryRecvError},
    task::JoinHandle,
};

/// A small harness around `rig::agent::Agent` that owns the queue between prompts.
pub struct AgentLoop<M, P = ()>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    agent: Arc<Agent<M, P>>,
    max_turns: usize,
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
        }
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
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
    pub fn prompt(&self, prompt: impl Into<Message>) -> AgentLoopHandle {
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let runner = Runner::new(self.agent.clone(), self.max_turns, prompt.into());
        let task = tokio::spawn(async move { runner.run(commands_rx).await });

        AgentLoopHandle { commands_tx, task }
    }
}

/// A running agent loop.
pub struct AgentLoopHandle {
    commands_tx: mpsc::UnboundedSender<Command>,
    task: JoinHandle<Result<AgentLoopResult, AgentLoopError>>,
}

impl AgentLoopHandle {
    /// Wait for the loop to become idle, abort, or error.
    pub async fn wait(self) -> Result<AgentLoopResult, AgentLoopError> {
        match self.task.await {
            Ok(result) => result,
            Err(err) => Err(AgentLoopError::TaskJoin(err)),
        }
    }

    /// Queue a message to run before follow-ups after the current prompt settles.
    pub fn steer(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        self.send(Command::Steer(message.into()))
    }

    /// Queue a message to run after the current prompt would otherwise leave the loop idle.
    pub fn follow_up(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        self.send(Command::FollowUp(message.into()))
    }

    /// Stop the current prompt, keep committed tool results, then run this message next.
    pub fn interrupt(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        self.send(Command::Interrupt(message.into()))
    }

    /// Stop the loop. Completed tool results from the current prompt are kept.
    pub fn abort(&self) -> Result<(), AgentLoopError> {
        self.send(Command::Abort)
    }

    fn send(&self, command: Command) -> Result<(), AgentLoopError> {
        self.commands_tx
            .send(command)
            .map_err(|_| AgentLoopError::CommandChannelClosed)
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

#[derive(Debug)]
pub enum AgentLoopError {
    CommandChannelClosed,
    Rig(StreamingError),
    TaskJoin(tokio::task::JoinError),
}

impl fmt::Display for AgentLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandChannelClosed => f.write_str("agent loop command channel is closed"),
            Self::Rig(err) => write!(f, "{err}"),
            Self::TaskJoin(err) => write!(f, "{err}"),
        }
    }
}

impl Error for AgentLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
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
    immediate: VecDeque<Message>,
    steering: VecDeque<Message>,
    follow_ups: VecDeque<Message>,
    history: Vec<Message>,
    last_response: Option<String>,
}

impl<M, P> Runner<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn new(agent: Arc<Agent<M, P>>, max_turns: usize, prompt: Message) -> Self {
        Self {
            agent,
            max_turns,
            immediate: VecDeque::from([prompt]),
            steering: VecDeque::new(),
            follow_ups: VecDeque::new(),
            history: Vec::new(),
            last_response: None,
        }
    }

    fn finish(self, end_reason: EndReason) -> AgentLoopResult {
        AgentLoopResult {
            end_reason,
            history: self.history,
            last_response: self.last_response,
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
        let base_history = self.history.clone();
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
                            if let Some(history) = partial_turn.committed_history(&base_history, &prompt) {
                                self.history = history;
                            }
                            self.immediate.push_front(message);
                            return Ok(PromptAction::Continue);
                        }
                        Command::Abort => {
                            if let Some(history) = partial_turn.committed_history(&base_history, &prompt) {
                                self.history = history;
                            }
                            return Ok(PromptAction::Abort);
                        }
                    }
                }
                item = stream.next() => {
                    let Some(item) = item else {
                        return Ok(PromptAction::Continue);
                    };

                    match item {
                        Ok(item) => self.handle_stream_item(item, &mut partial_turn),
                        Err(err) => {
                            if let Some(history) = history_from_error(&err) {
                                self.history = history;
                            }
                            return Err(err.into());
                        }
                    }
                }
            }
        }
    }

    fn handle_stream_item(
        &mut self,
        item: MultiTurnStreamItem<M::StreamingResponse>,
        partial_turn: &mut PartialTurn,
    ) {
        match item {
            MultiTurnStreamItem::StreamAssistantItem(item) => {
                partial_turn.note_assistant_item(item)
            }
            MultiTurnStreamItem::StreamUserItem(item) => partial_turn.note_user_item(item),
            MultiTurnStreamItem::FinalResponse(final_response) => {
                self.commit_final_response(&final_response);
            }
            MultiTurnStreamItem::CompletionCall(_) => {}
            _ => {}
        }
    }

    fn commit_final_response(&mut self, final_response: &FinalResponse) {
        self.last_response = Some(final_response.response().to_string());

        if let Some(messages) = final_response.history() {
            self.history.extend(messages.iter().cloned());
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

    fn committed_history(
        &self,
        base_history: &[Message],
        prompt: &Message,
    ) -> Option<Vec<Message>> {
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

        let mut history = base_history.to_vec();
        history.push(prompt.clone());
        history.push(Message::Assistant {
            id: None,
            content: assistant_content,
        });
        history.push(Message::User {
            content: user_content,
        });
        Some(history)
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
        test_utils::{MockCompletionModel, MockStreamEvent},
    };

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

        let history = partial
            .committed_history(&[], &Message::user("start"))
            .expect("completed tool results should produce history");

        assert_eq!(history.len(), 3);
        assert!(matches!(history[1], Message::Assistant { .. }));
        assert!(
            matches!(&history[2], Message::User { content } if matches!(content.first(), UserContent::ToolResult(_)))
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
}
