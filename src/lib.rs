use std::{
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    fmt,
    future::{Future, IntoFuture},
    pin::Pin,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures::StreamExt;
use rig::{
    OneOrMany,
    agent::{Agent, MultiTurnStreamItem, PromptHook, StreamingError},
    completion::{CompletionError, CompletionModel, GetTokenUsage, PromptError},
    message::{AssistantContent, Message, ToolCall, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt},
    wasm_compat::WasmCompatSend,
};
use tokio::{
    sync::{
        broadcast,
        mpsc::{self, error::TryRecvError},
    },
    task::JoinHandle,
    time::{Instant, sleep_until},
};

pub type TurnHookError = Box<dyn Error + Send + Sync + 'static>;

type TurnHookFuture = Pin<Box<dyn Future<Output = Result<TurnHookAction, TurnHookError>> + Send>>;
type TurnHook<S> = Arc<dyn Fn(Arc<S>, TurnHookContext) -> TurnHookFuture + Send + Sync>;
type SharedMessages = Arc<Mutex<Vec<Message>>>;
type SharedTurnRunning = Arc<AtomicBool>;
const EVENT_BUFFER_SIZE: usize = 1024;

fn lock_messages(messages: &SharedMessages) -> MutexGuard<'_, Vec<Message>> {
    match messages.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A small harness around `rig::agent::Agent` that owns the queue between prompts.
pub struct AgentLoop<M, P = (), S = ()>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    agent: Arc<Agent<M, P>>,
    app_state: Arc<S>,
    max_turns: usize,
    turn_timeout: Option<Duration>,
    loop_timeout: Option<Duration>,
    initial_history: Vec<Message>,
    turn_hook: Option<TurnHook<S>>,
}

impl<M, P> AgentLoop<M, P, ()>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    pub fn new(agent: Agent<M, P>) -> Self {
        Self {
            agent: Arc::new(agent),
            app_state: Arc::new(()),
            max_turns: 100,
            turn_timeout: None,
            loop_timeout: None,
            initial_history: Vec::new(),
            turn_hook: None,
        }
    }
}

impl<M, P> AgentLoop<M, P, ()>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Attach caller-owned state that stateful hooks can use without capture boilerplate.
    pub fn with_app_state<S>(self, app_state: S) -> AgentLoop<M, P, S>
    where
        S: Send + Sync + 'static,
    {
        let turn_hook = self.turn_hook.map(|hook| {
            Arc::new(move |_state: Arc<S>, turn| hook(Arc::new(()), turn)) as TurnHook<S>
        });

        AgentLoop {
            agent: self.agent,
            app_state: Arc::new(app_state),
            max_turns: self.max_turns,
            turn_timeout: self.turn_timeout,
            loop_timeout: self.loop_timeout,
            initial_history: self.initial_history,
            turn_hook,
        }
    }
}

impl<M, P, S> AgentLoop<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
    S: Send + Sync + 'static,
{
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set a wall-clock timeout for each active agent turn.
    pub fn turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = Some(timeout);
        self
    }

    /// Set a wall-clock timeout for the whole loop lifetime.
    pub fn loop_timeout(mut self, timeout: Duration) -> Self {
        self.loop_timeout = Some(timeout);
        self
    }

    /// Seed the loop's active in-memory history.
    pub fn with_history<H, T>(mut self, history: H) -> Self
    where
        H: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.initial_history = history.into_iter().map(Into::into).collect();
        self
    }

    /// Observe and optionally rewrite each turn commit.
    ///
    /// The hook receives `(app_state, turn)`. `turn.history` is the candidate
    /// full history after appending `turn.new_messages`. `turn.new_messages` is
    /// only the ordered append batch from this boundary. For interrupted turns,
    /// the append batch contains only completed tool-call/tool-result
    /// roundtrips. The hook is still called when the append batch is empty.
    pub fn with_turn_hook<F, Fut, E>(mut self, hook: F) -> Self
    where
        F: Fn(Arc<S>, TurnHookContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<TurnHookAction, E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        self.turn_hook = Some(Arc::new(move |state, turn| {
            let future = hook(state, turn);
            Box::pin(async move { future.await.map_err(|err| Box::new(err) as TurnHookError) })
        }));
        self
    }
}

impl<M, P, S> AgentLoop<M, P, S>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    /// Start the loop with an initial user prompt and return a control handle.
    pub fn prompt(&self, prompt: impl Into<Message>) -> AgentLoopHandle<M::StreamingResponse> {
        self.start(PendingTurn::single(prompt.into()))
    }

    /// Start the loop from the seeded history without appending a new prompt.
    pub fn resume(&self) -> Result<AgentLoopHandle<M::StreamingResponse>, AgentLoopError> {
        validate_resume_history(&self.initial_history).map_err(AgentLoopError::InvalidHistory)?;
        Ok(self.start(PendingTurn::Resume))
    }

    fn start(&self, initial_turn: PendingTurn) -> AgentLoopHandle<M::StreamingResponse> {
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (events_tx, _) = broadcast::channel(EVENT_BUFFER_SIZE);
        let state = Arc::new(Mutex::new(self.initial_history.clone()));
        let turn_running = Arc::new(AtomicBool::new(false));
        let runner = Runner::new(
            self.agent.clone(),
            self.app_state.clone(),
            self.max_turns,
            self.turn_timeout,
            self.loop_timeout,
            self.turn_hook.clone(),
            events_tx.clone(),
            state.clone(),
            turn_running.clone(),
            initial_turn,
        );
        let task = tokio::spawn(async move { runner.run(commands_rx).await });

        AgentLoopHandle {
            commands_tx,
            events_tx,
            state,
            turn_running,
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
    turn_running: SharedTurnRunning,
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

    /// Wait for the loop to become idle, abort, or reach a known end reason.
    pub async fn wait(self) -> Result<AgentLoopResult, AgentLoopError> {
        let AgentLoopHandle {
            commands_tx, task, ..
        } = self;
        drop(commands_tx);

        match task.await {
            Ok(result) => result,
            Err(err) => Err(AgentLoopError::TaskJoin(err)),
        }
    }

    /// Queue a message to run before follow-ups after the current prompt settles.
    ///
    /// All steering messages ready for the next turn are applied together, in
    /// queue order.
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
    ///
    /// Follow-ups are applied one turn at a time.
    pub fn follow_up(&self, message: impl Into<Message>) -> Result<(), AgentLoopError> {
        let message = message.into();
        self.send(Command::FollowUp(message.clone()))?;
        self.emit(AgentLoopEvent::Queued {
            kind: QueueKind::FollowUp,
            message: Some(message),
        });
        Ok(())
    }

    /// Queue another agent turn using the committed history as-is.
    ///
    /// This does not append a new user message. The current committed history
    /// must already be valid when this is called.
    pub fn resume(&self) -> Result<(), AgentLoopError> {
        if self.turn_running.load(Ordering::SeqCst) {
            return Ok(());
        }

        validate_resume_history(&self.state()).map_err(AgentLoopError::InvalidHistory)?;
        self.send(Command::Resume)?;
        self.emit(AgentLoopEvent::Queued {
            kind: QueueKind::Resume,
            message: None,
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
    /// A turn hook stopped the loop after a commit boundary.
    AbortedByHook { reason: String },
    /// The provider refused or filtered the request or response.
    ContentFilter { error: ApiErrorInfo },
    /// The provider rejected the request because the context was too large.
    ContextFull { error: ApiErrorInfo },
    /// The provider stopped because an output/token limit was reached.
    Length { error: ApiErrorInfo },
    /// Rig stopped after exceeding the configured multi-turn tool-call limit.
    MaxTurns { max_turns: usize },
    /// The active turn exceeded the configured wall-clock timeout.
    TurnTimedOut { timeout: Duration },
    /// The loop exceeded the configured wall-clock timeout.
    LoopTimedOut { timeout: Duration },
    /// The turn failed while using tools.
    ToolError { message: String },
    /// The provider or client returned an API/request/response error.
    ApiError { error: ApiErrorInfo },
    /// A non-API error ended the loop.
    Other { message: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiErrorInfo {
    pub kind: ApiErrorKind,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiErrorKind {
    Http,
    Json,
    Url,
    Request,
    Response,
    Provider,
}

#[derive(Clone, Debug)]
pub struct TurnHookContext {
    /// The candidate full history after appending `new_messages`.
    pub history: Vec<Message>,
    /// The ordered messages generated at this turn boundary. This can be empty.
    pub new_messages: Vec<Message>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TurnHookAction {
    /// Keep `TurnHookContext::history` as the loop's active history.
    Continue,
    /// Replace the loop's active history with this complete history.
    ReplaceHistory(Vec<Message>),
    /// Keep `TurnHookContext::history`, then stop the loop.
    Abort { reason: String },
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
    Resume,
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
    TurnTimedOut {
        messages: Vec<Message>,
    },
    HistoryReplaced {
        messages: Vec<Message>,
    },
    LoopEnded {
        end_reason: EndReason,
    },
}

#[derive(Debug)]
pub enum AgentLoopError {
    CommandChannelClosed,
    InvalidHistory(InvalidHistoryError),
    TurnHook(TurnHookError),
    Rig(StreamingError),
    TaskJoin(tokio::task::JoinError),
}

impl fmt::Display for AgentLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandChannelClosed => f.write_str("agent loop command channel is closed"),
            Self::InvalidHistory(err) => write!(f, "{err}"),
            Self::TurnHook(err) => write!(f, "turn hook failed: {err}"),
            Self::Rig(err) => write!(f, "{err}"),
            Self::TaskJoin(err) => write!(f, "{err}"),
        }
    }
}

impl Error for AgentLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidHistory(err) => Some(err),
            Self::TurnHook(err) => Some(err.as_ref()),
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidHistoryError {
    pub message: String,
}

impl InvalidHistoryError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for InvalidHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid agent history: {}", self.message)
    }
}

impl Error for InvalidHistoryError {}

enum Command {
    Steer(Message),
    FollowUp(Message),
    Resume,
    Interrupt(Message),
    Abort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimeoutKind {
    Turn,
    Loop,
}

#[derive(Clone, Copy, Debug)]
struct TimeoutDeadline {
    kind: TimeoutKind,
    at: Instant,
    timeout: Duration,
}

impl TimeoutDeadline {
    fn new(kind: TimeoutKind, timeout: Duration) -> Self {
        Self {
            kind,
            at: Instant::now() + timeout,
            timeout,
        }
    }

    fn end_reason(self) -> EndReason {
        match self.kind {
            TimeoutKind::Turn => EndReason::TurnTimedOut {
                timeout: self.timeout,
            },
            TimeoutKind::Loop => EndReason::LoopTimedOut {
                timeout: self.timeout,
            },
        }
    }
}

fn next_timeout(
    turn: Option<TimeoutDeadline>,
    loop_deadline: Option<TimeoutDeadline>,
) -> Option<TimeoutDeadline> {
    match (turn, loop_deadline) {
        (Some(turn), Some(loop_deadline)) if loop_deadline.at <= turn.at => Some(loop_deadline),
        (Some(turn), Some(_)) => Some(turn),
        (Some(turn), None) => Some(turn),
        (None, Some(loop_deadline)) => Some(loop_deadline),
        (None, None) => None,
    }
}

fn elapsed_timeout(deadline: Option<TimeoutDeadline>) -> Option<TimeoutDeadline> {
    deadline.filter(|deadline| Instant::now() >= deadline.at)
}

async fn wait_for_timeout(deadline: Option<TimeoutDeadline>) {
    if let Some(deadline) = deadline {
        sleep_until(deadline.at).await;
    } else {
        std::future::pending::<()>().await;
    }
}

struct Runner<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    agent: Arc<Agent<M, P>>,
    app_state: Arc<S>,
    max_turns: usize,
    turn_timeout: Option<Duration>,
    loop_timeout: Option<Duration>,
    turn_hook: Option<TurnHook<S>>,
    events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
    state: SharedMessages,
    turn_running: SharedTurnRunning,
    immediate: VecDeque<Message>,
    steering: VecDeque<Message>,
    resumes: usize,
    follow_ups: VecDeque<Message>,
    last_response: Option<String>,
}

impl<M, P, S> Runner<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn new(
        agent: Arc<Agent<M, P>>,
        app_state: Arc<S>,
        max_turns: usize,
        turn_timeout: Option<Duration>,
        loop_timeout: Option<Duration>,
        turn_hook: Option<TurnHook<S>>,
        events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
        state: SharedMessages,
        turn_running: SharedTurnRunning,
        initial_turn: PendingTurn,
    ) -> Self {
        let (immediate, resumes) = match initial_turn {
            PendingTurn::Prompt { prelude, prompt } => {
                debug_assert!(prelude.is_empty());
                (VecDeque::from([prompt]), 0)
            }
            PendingTurn::Resume => (VecDeque::new(), 1),
        };

        Self {
            agent,
            app_state,
            max_turns,
            turn_timeout,
            loop_timeout,
            turn_hook,
            events_tx,
            state,
            turn_running,
            immediate,
            steering: VecDeque::new(),
            resumes,
            follow_ups: VecDeque::new(),
            last_response: None,
        }
    }

    fn next_turn(&mut self) -> Option<PendingTurn> {
        if let Some(message) = self.immediate.pop_front() {
            return Some(PendingTurn::single(message));
        }

        if !self.steering.is_empty() {
            return PendingTurn::new(self.steering.drain(..).collect());
        }

        if self.resumes > 0 {
            self.resumes -= 1;
            return Some(PendingTurn::Resume);
        }

        self.follow_ups.pop_front().map(PendingTurn::single)
    }

    fn has_pending_turn(&self) -> bool {
        !self.immediate.is_empty()
            || !self.steering.is_empty()
            || self.resumes > 0
            || !self.follow_ups.is_empty()
    }

    fn handle_idle_command(&mut self, command: Command) -> CommandAction {
        match command {
            Command::Steer(message) => self.steering.push_back(message),
            Command::FollowUp(message) => self.follow_ups.push_back(message),
            Command::Resume => self.resumes += 1,
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

enum PendingTurn {
    Prompt {
        prelude: Vec<Message>,
        prompt: Message,
    },
    Resume,
}

impl PendingTurn {
    fn new(mut messages: Vec<Message>) -> Option<Self> {
        let prompt = messages.pop()?;
        Some(Self::Prompt {
            prelude: messages,
            prompt,
        })
    }

    fn single(prompt: Message) -> Self {
        Self::Prompt {
            prelude: Vec::new(),
            prompt,
        }
    }
}

struct PreparedTurn {
    committed_base_history: Vec<Message>,
    request_history: Vec<Message>,
    prelude: Vec<Message>,
    prompt: Message,
    commit_prompt: bool,
}

impl PreparedTurn {
    fn from_pending(
        turn: PendingTurn,
        committed_base_history: Vec<Message>,
    ) -> Result<Self, AgentLoopError> {
        match turn {
            PendingTurn::Prompt { prelude, prompt } => {
                let mut request_history = committed_base_history.clone();
                request_history.extend(prelude.clone());
                let mut full_request_history = request_history.clone();
                full_request_history.push(prompt.clone());
                validate_message_history(&full_request_history)
                    .map_err(AgentLoopError::InvalidHistory)?;

                Ok(Self {
                    committed_base_history,
                    request_history,
                    prelude,
                    prompt,
                    commit_prompt: true,
                })
            }
            PendingTurn::Resume => {
                validate_resume_history(&committed_base_history)
                    .map_err(AgentLoopError::InvalidHistory)?;
                let Some((prompt, history)) = committed_base_history.split_last() else {
                    unreachable!("validated resume history is non-empty");
                };

                Ok(Self {
                    committed_base_history: committed_base_history.clone(),
                    request_history: history.to_vec(),
                    prelude: Vec::new(),
                    prompt: prompt.clone(),
                    commit_prompt: false,
                })
            }
        }
    }

    fn append_messages(&self, mut messages: Vec<Message>) -> Vec<Message> {
        if !self.commit_prompt && !messages.is_empty() {
            messages.remove(0);
        }

        if self.prelude.is_empty() {
            return messages;
        }

        let mut append = self.prelude.clone();
        append.append(&mut messages);
        append
    }

    fn partial_messages(&self, partial_turn: &PartialTurn) -> Vec<Message> {
        partial_turn
            .append_messages(&self.prompt)
            .map(|messages| self.append_messages(messages))
            .unwrap_or_default()
    }
}

impl<M, P, S> Runner<M, P, S>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    async fn run(
        mut self,
        mut commands_rx: mpsc::UnboundedReceiver<Command>,
    ) -> Result<AgentLoopResult, AgentLoopError> {
        self.emit(AgentLoopEvent::LoopStarted);
        let loop_deadline = self
            .loop_timeout
            .map(|timeout| TimeoutDeadline::new(TimeoutKind::Loop, timeout));

        loop {
            if let Some(deadline) = elapsed_timeout(loop_deadline) {
                return Ok(self.finish(deadline.end_reason()));
            }

            if self.drain_ready_commands(&mut commands_rx) == CommandAction::Abort {
                return Ok(self.finish(EndReason::Aborted));
            }

            if !self.has_pending_turn() {
                let command = tokio::select! {
                    command = commands_rx.recv() => command,
                    _ = wait_for_timeout(loop_deadline), if loop_deadline.is_some() => {
                        let deadline = loop_deadline.expect("loop timeout branch requires a deadline");
                        return Ok(self.finish(deadline.end_reason()));
                    }
                };

                let Some(command) = command else {
                    return Ok(self.finish(EndReason::Idle));
                };

                if self.handle_idle_command(command) == CommandAction::Abort {
                    return Ok(self.finish(EndReason::Aborted));
                }
                continue;
            }

            if self.drain_ready_commands(&mut commands_rx) == CommandAction::Abort {
                return Ok(self.finish(EndReason::Aborted));
            }

            let Some(turn) = self.next_turn() else {
                continue;
            };

            match self.run_turn(turn, &mut commands_rx, loop_deadline).await? {
                PromptAction::Continue => {}
                PromptAction::Abort => return Ok(self.finish(EndReason::Aborted)),
                PromptAction::Finish(end_reason) => return Ok(self.finish(end_reason)),
            }
        }
    }

    async fn run_turn(
        &mut self,
        turn: PendingTurn,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
        loop_deadline: Option<TimeoutDeadline>,
    ) -> Result<PromptAction, AgentLoopError> {
        let committed_base_history = self.history_snapshot();
        let turn = PreparedTurn::from_pending(turn, committed_base_history)?;
        self.set_turn_running(true);

        self.emit(AgentLoopEvent::TurnStarted {
            prompt: turn.prompt.clone(),
        });

        let mut partial_turn = PartialTurn::default();
        let mut commands_closed = false;
        let turn_deadline = self
            .turn_timeout
            .map(|timeout| TimeoutDeadline::new(TimeoutKind::Turn, timeout));
        let stream_future = self
            .agent
            .stream_prompt(turn.prompt.clone())
            .with_history(turn.request_history.clone())
            .multi_turn(self.max_turns)
            .into_future();
        tokio::pin!(stream_future);

        let mut stream = loop {
            let deadline = next_timeout(turn_deadline, loop_deadline);
            if let Some(deadline) = elapsed_timeout(deadline) {
                return self.timeout_turn(&turn, &partial_turn, deadline).await;
            }

            tokio::select! {
                command = commands_rx.recv(), if !commands_closed => {
                    let Some(command) = command else {
                        commands_closed = true;
                        continue;
                    };

                    if let Some(action) = self
                        .handle_turn_command(command, &turn, &partial_turn)
                        .await?
                    {
                        return Ok(action);
                    }
                }
                _ = wait_for_timeout(deadline), if deadline.is_some() => {
                    let deadline = deadline.expect("timeout branch requires a deadline");
                    return self.timeout_turn(&turn, &partial_turn, deadline).await;
                }
                stream = &mut stream_future => break stream,
            }
        };

        loop {
            let deadline = next_timeout(turn_deadline, loop_deadline);
            if let Some(deadline) = elapsed_timeout(deadline) {
                return self.timeout_turn(&turn, &partial_turn, deadline).await;
            }

            tokio::select! {
                command = commands_rx.recv(), if !commands_closed => {
                    let Some(command) = command else {
                        commands_closed = true;
                        continue;
                    };

                    if let Some(action) = self
                        .handle_turn_command(command, &turn, &partial_turn)
                        .await?
                    {
                        return Ok(action);
                    }
                }
                _ = wait_for_timeout(deadline), if deadline.is_some() => {
                    let deadline = deadline.expect("timeout branch requires a deadline");
                    return self.timeout_turn(&turn, &partial_turn, deadline).await;
                }
                item = stream.next() => {
                    let Some(item) = item else {
                        self.set_turn_running(false);
                        return Ok(PromptAction::Continue);
                    };

                    match item {
                        Ok(item) => {
                            self.emit(AgentLoopEvent::Rig(item.clone()));
                            if let Some(end_reason) = self
                                .handle_stream_item(item, &turn, &mut partial_turn)
                                .await?
                            {
                                return Ok(PromptAction::Finish(end_reason));
                            }
                        }
                        Err(err) => {
                            if let Some(history) = history_from_error(&err) {
                                self.commit_recovered_history(&turn.committed_base_history, history).await?;
                            } else {
                                self.set_turn_running(false);
                            }
                            return Ok(PromptAction::Finish(end_reason_from_streaming_error(&err)));
                        }
                    }
                }
            }
        }
    }

    async fn handle_turn_command(
        &mut self,
        command: Command,
        turn: &PreparedTurn,
        partial_turn: &PartialTurn,
    ) -> Result<Option<PromptAction>, AgentLoopError> {
        match command {
            Command::Steer(message) => self.steering.push_back(message),
            Command::FollowUp(message) => self.follow_ups.push_back(message),
            Command::Resume => self.resumes += 1,
            Command::Interrupt(message) => {
                let messages = turn.partial_messages(partial_turn);
                if let CommitOutcome::Abort(end_reason) = self
                    .commit_append(messages, CommitEvent::TurnInterrupted)
                    .await?
                {
                    return Ok(Some(PromptAction::Finish(end_reason)));
                }
                self.immediate.push_front(message);
                return Ok(Some(PromptAction::Continue));
            }
            Command::Abort => {
                let messages = turn.partial_messages(partial_turn);
                self.commit_append(messages, CommitEvent::TurnAborted)
                    .await?;
                return Ok(Some(PromptAction::Abort));
            }
        }

        Ok(None)
    }

    async fn timeout_turn(
        &mut self,
        turn: &PreparedTurn,
        partial_turn: &PartialTurn,
        deadline: TimeoutDeadline,
    ) -> Result<PromptAction, AgentLoopError> {
        let messages = turn.partial_messages(partial_turn);
        self.commit_append(messages, CommitEvent::TurnTimedOut)
            .await?;
        Ok(PromptAction::Finish(deadline.end_reason()))
    }

    async fn handle_stream_item(
        &mut self,
        item: MultiTurnStreamItem<M::StreamingResponse>,
        turn: &PreparedTurn,
        partial_turn: &mut PartialTurn,
    ) -> Result<Option<EndReason>, AgentLoopError> {
        match item {
            MultiTurnStreamItem::StreamAssistantItem(item) => {
                partial_turn.note_assistant_item(item)
            }
            MultiTurnStreamItem::StreamUserItem(item) => partial_turn.note_user_item(item),
            MultiTurnStreamItem::FinalResponse(final_response) => {
                self.last_response = Some(final_response.response().to_string());
                if let Some(messages) = final_response.history() {
                    if let CommitOutcome::Abort(end_reason) = self
                        .commit_append(
                            turn.append_messages(messages.to_vec()),
                            CommitEvent::TurnCommitted,
                        )
                        .await?
                    {
                        return Ok(Some(end_reason));
                    }
                }
            }
            MultiTurnStreamItem::CompletionCall(_) => {}
            _ => {}
        }

        Ok(None)
    }

    async fn commit_append(
        &mut self,
        messages: Vec<Message>,
        event: CommitEvent,
    ) -> Result<CommitOutcome, AgentLoopError> {
        let before = self.history_snapshot();
        let mut next_history = before;
        next_history.extend(messages.clone());
        validate_message_history(&next_history).map_err(AgentLoopError::InvalidHistory)?;

        let mut replaced_history = None;
        let mut abort_reason = None;

        if let Some(hook) = &self.turn_hook {
            let turn = TurnHookContext {
                history: next_history.clone(),
                new_messages: messages.clone(),
            };

            match hook(self.app_state.clone(), turn)
                .await
                .map_err(AgentLoopError::TurnHook)?
            {
                TurnHookAction::Continue => {}
                TurnHookAction::ReplaceHistory(history) => {
                    validate_message_history(&history).map_err(AgentLoopError::InvalidHistory)?;
                    next_history = history.clone();
                    replaced_history = Some(history);
                }
                TurnHookAction::Abort { reason } => {
                    abort_reason = Some(reason);
                }
            }
        }

        *lock_messages(&self.state) = next_history;
        self.set_turn_running(false);
        self.emit(event.into_agent_event(messages));

        if let Some(messages) = replaced_history {
            self.emit(AgentLoopEvent::HistoryReplaced { messages });
        }

        if let Some(reason) = abort_reason {
            return Ok(CommitOutcome::Abort(EndReason::AbortedByHook { reason }));
        }

        Ok(CommitOutcome::Continue)
    }

    async fn commit_recovered_history(
        &mut self,
        base_history: &[Message],
        recovered_history: Vec<Message>,
    ) -> Result<(), AgentLoopError> {
        if recovered_history.len() < base_history.len() {
            *lock_messages(&self.state) = recovered_history;
            self.set_turn_running(false);
            return Ok(());
        }

        let append = recovered_history[base_history.len()..].to_vec();
        self.commit_append(append, CommitEvent::TurnCommitted)
            .await
            .map(|_| ())
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

    fn set_turn_running(&self, running: bool) {
        self.turn_running.store(running, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitEvent {
    TurnCommitted,
    TurnInterrupted,
    TurnAborted,
    TurnTimedOut,
}

impl CommitEvent {
    fn into_agent_event<R>(self, messages: Vec<Message>) -> AgentLoopEvent<R> {
        match self {
            Self::TurnCommitted => AgentLoopEvent::TurnCommitted { messages },
            Self::TurnInterrupted => AgentLoopEvent::TurnInterrupted { messages },
            Self::TurnAborted => AgentLoopEvent::TurnAborted { messages },
            Self::TurnTimedOut => AgentLoopEvent::TurnTimedOut { messages },
        }
    }
}

#[derive(Default)]
struct PartialTurn {
    pending_tool_calls: HashMap<String, ToolCall>,
    assistant_content: Vec<AssistantContent>,
    completed_messages: Vec<Message>,
}

impl PartialTurn {
    fn note_assistant_item<R>(&mut self, item: StreamedAssistantContent<R>)
    where
        R: Clone,
    {
        match item {
            StreamedAssistantContent::Text(text) => {
                self.assistant_content.push(AssistantContent::Text(text));
            }
            StreamedAssistantContent::Reasoning(reasoning) => {
                self.assistant_content
                    .push(AssistantContent::Reasoning(reasoning));
            }
            StreamedAssistantContent::ToolCall {
                internal_call_id,
                tool_call,
            } => {
                self.pending_tool_calls.insert(internal_call_id, tool_call);
            }
            StreamedAssistantContent::ToolCallDelta { .. }
            | StreamedAssistantContent::ReasoningDelta { .. }
            | StreamedAssistantContent::Final(_) => {}
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

        self.assistant_content
            .push(AssistantContent::ToolCall(tool_call));

        let Ok(assistant_content) = OneOrMany::many(std::mem::take(&mut self.assistant_content))
        else {
            return;
        };

        self.completed_messages.push(Message::Assistant {
            id: None,
            content: assistant_content,
        });
        self.completed_messages.push(Message::User {
            content: OneOrMany::one(UserContent::ToolResult(tool_result)),
        });
    }

    fn append_messages(&self, prompt: &Message) -> Option<Vec<Message>> {
        if self.completed_messages.is_empty() {
            return None;
        }

        let mut messages = Vec::with_capacity(self.completed_messages.len() + 1);
        messages.push(prompt.clone());
        messages.extend(self.completed_messages.clone());
        Some(messages)
    }
}

fn validate_message_history(messages: &[Message]) -> Result<(), InvalidHistoryError> {
    let mut pending_tool_calls: Option<(usize, Vec<String>)> = None;

    for (index, message) in messages.iter().enumerate() {
        if let Some((assistant_index, expected_ids)) = pending_tool_calls.take() {
            let Message::User { content } = message else {
                return Err(InvalidHistoryError::new(format!(
                    "assistant message {assistant_index} contains tool calls, but message {index} is not the required user tool-result message"
                )));
            };

            let result_ids = tool_result_ids(content);
            if result_ids.is_empty() {
                return Err(InvalidHistoryError::new(format!(
                    "assistant message {assistant_index} contains tool calls, but user message {index} contains no tool results"
                )));
            }

            ensure_unique_ids(&result_ids, index, "tool result")?;

            for id in &expected_ids {
                if !result_ids.contains(id) {
                    return Err(InvalidHistoryError::new(format!(
                        "assistant tool call `{id}` at message {assistant_index} is missing a matching tool result in message {index}"
                    )));
                }
            }

            for id in &result_ids {
                if !expected_ids.contains(id) {
                    return Err(InvalidHistoryError::new(format!(
                        "tool result `{id}` at message {index} does not match any tool call from message {assistant_index}"
                    )));
                }
            }

            continue;
        }

        match message {
            Message::Assistant { content, .. } => {
                let tool_call_ids = assistant_tool_call_ids(content);
                ensure_unique_ids(&tool_call_ids, index, "tool call")?;
                if !tool_call_ids.is_empty() {
                    pending_tool_calls = Some((index, tool_call_ids));
                }
            }
            Message::User { content } => {
                let result_ids = tool_result_ids(content);
                if let Some(id) = result_ids.first() {
                    return Err(InvalidHistoryError::new(format!(
                        "tool result `{id}` at message {index} has no immediately preceding assistant tool call"
                    )));
                }
            }
            Message::System { .. } => {}
        }
    }

    if let Some((assistant_index, expected_ids)) = pending_tool_calls {
        return Err(InvalidHistoryError::new(format!(
            "assistant message {assistant_index} contains unanswered tool calls: {}",
            expected_ids.join(", ")
        )));
    }

    Ok(())
}

fn validate_resume_history(messages: &[Message]) -> Result<(), InvalidHistoryError> {
    validate_message_history(messages)?;

    match messages.last() {
        Some(Message::User { .. } | Message::Assistant { .. }) => Ok(()),
        Some(Message::System { .. }) => Err(InvalidHistoryError::new(
            "resume requires the last committed message to be user or assistant content",
        )),
        None => Err(InvalidHistoryError::new(
            "resume requires at least one committed message",
        )),
    }
}

fn assistant_tool_call_ids(content: &OneOrMany<AssistantContent>) -> Vec<String> {
    content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids(content: &OneOrMany<UserContent>) -> Vec<String> {
    content
        .iter()
        .filter_map(|item| match item {
            UserContent::ToolResult(tool_result) => Some(tool_result.id.clone()),
            _ => None,
        })
        .collect()
}

fn ensure_unique_ids(
    ids: &[String],
    message_index: usize,
    kind: &str,
) -> Result<(), InvalidHistoryError> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(InvalidHistoryError::new(format!(
                "duplicate {kind} id `{id}` at message {message_index}"
            )));
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandAction {
    Continue,
    Abort,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CommitOutcome {
    Continue,
    Abort(EndReason),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PromptAction {
    Continue,
    Abort,
    Finish(EndReason),
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

fn end_reason_from_streaming_error(err: &StreamingError) -> EndReason {
    match err {
        StreamingError::Completion(err) => end_reason_from_completion_error(err),
        StreamingError::Prompt(err) => end_reason_from_prompt_error(err),
        StreamingError::Tool(err) => EndReason::ToolError {
            message: err.to_string(),
        },
    }
}

fn end_reason_from_prompt_error(err: &PromptError) -> EndReason {
    match err {
        PromptError::CompletionError(err) => end_reason_from_completion_error(err),
        PromptError::ToolError(err) => EndReason::ToolError {
            message: err.to_string(),
        },
        PromptError::ToolServerError(err) => EndReason::ToolError {
            message: err.to_string(),
        },
        PromptError::MaxTurnsError { max_turns, .. } => EndReason::MaxTurns {
            max_turns: *max_turns,
        },
        PromptError::PromptCancelled { reason, .. } => EndReason::Other {
            message: reason.clone(),
        },
        PromptError::UnknownToolCall { tool_name, .. } => EndReason::ToolError {
            message: format!("unknown tool call: {tool_name}"),
        },
    }
}

fn end_reason_from_completion_error(err: &CompletionError) -> EndReason {
    let error = ApiErrorInfo::from_completion_error(err);

    if is_content_filter_error(&error.message) {
        EndReason::ContentFilter { error }
    } else if is_context_full_error(&error.message) {
        EndReason::ContextFull { error }
    } else if is_length_error(&error.message) {
        EndReason::Length { error }
    } else {
        EndReason::ApiError { error }
    }
}

impl ApiErrorInfo {
    fn from_completion_error(err: &CompletionError) -> Self {
        let (kind, message) = match err {
            CompletionError::HttpError(err) => (ApiErrorKind::Http, err.to_string()),
            CompletionError::JsonError(err) => (ApiErrorKind::Json, err.to_string()),
            CompletionError::UrlError(err) => (ApiErrorKind::Url, err.to_string()),
            CompletionError::RequestError(err) => (ApiErrorKind::Request, err.to_string()),
            CompletionError::ResponseError(message) => (ApiErrorKind::Response, message.clone()),
            CompletionError::ProviderError(message) => (ApiErrorKind::Provider, message.clone()),
        };

        Self { kind, message }
    }
}

fn is_content_filter_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    contains_any(
        &message,
        &[
            "content_filter",
            "content filter",
            "safety filter",
            "policy violation",
            "responsible_ai_policy",
            "unsafe prompt",
            "blocked by policy",
        ],
    )
}

fn is_context_full_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    contains_any(
        &message,
        &[
            "context_length_exceeded",
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "too many input tokens",
        ],
    )
}

fn is_length_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    contains_any(
        &message,
        &[
            "max_output_tokens",
            "max output tokens",
            "output token limit",
            "finish_reason: length",
            "finish reason length",
        ],
    )
}

fn contains_any(message: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| message.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::{
        agent::AgentBuilder,
        completion::{
            CompletionError, CompletionModel, CompletionRequest, CompletionResponse, ToolDefinition,
        },
        message::{ToolResult, UserContent},
        streaming::{
            RawStreamingChoice, StreamedAssistantContent, StreamingCompletionResponse,
            ToolCallDeltaContent,
        },
        test_utils::{MockAddTool, MockCompletionModel, MockResponse, MockStreamEvent},
        tool::Tool,
    };
    use std::sync::{Arc, Mutex};
    use tokio::{
        sync::Notify,
        time::{Duration, sleep, timeout},
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
    async fn with_history_seeds_active_state_and_first_request() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent).with_history([Message::user("previous")]);

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        let requests = model.requests();
        assert_eq!(
            user_texts(requests[0].chat_history.iter()),
            vec!["previous", "start"]
        );
        assert_eq!(user_texts(result.history.iter()), vec!["previous", "start"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn agent_loop_resume_starts_from_history_without_appending_a_user_message() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("resumed"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent)
            .with_history([Message::user("start"), Message::assistant("first")]);

        let result = agent_loop.resume().unwrap().wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(model.request_count(), 1);
        assert_eq!(user_texts(result.history.iter()), vec!["start"]);
        assert_eq!(
            assistant_texts(result.history.iter()),
            vec!["first", "resumed"]
        );

        let requests = model.requests();
        assert_eq!(user_texts(requests[0].chat_history.iter()), vec!["start"]);
        assert_eq!(
            assistant_texts(requests[0].chat_history.iter()),
            vec!["first"]
        );
    }

    #[test]
    fn agent_loop_resume_errors_when_seeded_history_is_invalid() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("unused"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent).with_history([Message::system("summary")]);

        let err = match agent_loop.resume() {
            Ok(_) => panic!("invalid history should not start a resumed loop"),
            Err(err) => err,
        };

        assert!(matches!(err, AgentLoopError::InvalidHistory(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_history_with_unanswered_tool_call_fails_before_request() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("unused"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop =
            AgentLoop::new(agent).with_history([assistant_tool_call_message("call_1")]);

        let err = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .expect_err("invalid history should fail before a request is sent");

        assert!(matches!(err, AgentLoopError::InvalidHistory(_)));
        assert_eq!(model.request_count(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_hook_replaces_active_history_between_turns() {
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
        let compacted = Arc::new(Mutex::new(false));
        let should_compact = compacted.clone();
        let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
            let should_compact = should_compact.clone();
            async move {
                let replace = {
                    let mut compacted = should_compact.lock().unwrap();
                    let replace = !*compacted && turn.history.len() >= 2;
                    if replace {
                        *compacted = true;
                    }
                    replace
                };

                if replace {
                    Ok::<_, std::io::Error>(TurnHookAction::ReplaceHistory(vec![Message::user(
                        "summary",
                    )]))
                } else {
                    Ok(TurnHookAction::Continue)
                }
            }
        });

        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();
        handle.follow_up(Message::user("follow-up")).unwrap();

        let result = handle.wait().await.unwrap();

        let requests = model.requests();
        assert_eq!(
            user_texts(requests[1].chat_history.iter()),
            vec!["summary", "follow-up"]
        );
        assert_eq!(
            user_texts(result.history.iter()),
            vec!["summary", "follow-up"]
        );

        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::HistoryReplaced { messages }
                    if user_texts(messages.iter()) == vec!["summary"]
            )
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_hook_error_stops_the_loop() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent).with_turn_hook(|_, _turn| async {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "hook failed",
            ))
        });

        let err = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .expect_err("turn hook failure should fail the run");

        assert!(matches!(err, AgentLoopError::TurnHook(_)));
        assert_eq!(model.request_count(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_hook_invalid_replacement_fails_the_commit() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent).with_turn_hook(|_, _turn| async {
            Ok::<_, std::io::Error>(TurnHookAction::ReplaceHistory(vec![
                assistant_tool_call_message("call_1"),
            ]))
        });

        let err = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .expect_err("invalid replacement history should fail the commit");

        assert!(matches!(err, AgentLoopError::InvalidHistory(_)));
        assert_eq!(model.request_count(), 1);
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
    async fn steering_messages_are_applied_together_by_default() {
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
        handle.steer(Message::user("steer-one")).unwrap();
        handle.steer(Message::user("steer-two")).unwrap();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(model.request_count(), 3);
        assert_eq!(
            user_prompts(&model),
            vec!["start", "steer-two", "follow-up"]
        );

        let requests = model.requests();
        assert_eq!(
            user_texts(requests[1].chat_history.iter()),
            vec!["start", "steer-one", "steer-two"]
        );
        assert_eq!(
            user_texts(result.history.iter()),
            vec!["start", "steer-one", "steer-two", "follow-up"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn follow_ups_are_applied_one_at_a_time() {
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("first"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("follow-one"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("follow-two"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent);

        let handle = agent_loop.prompt(Message::user("start"));
        handle.follow_up(Message::user("follow-up-one")).unwrap();
        handle.follow_up(Message::user("follow-up-two")).unwrap();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(model.request_count(), 3);
        assert_eq!(
            user_prompts(&model),
            vec!["start", "follow-up-one", "follow-up-two"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn resume_runs_again_without_appending_a_user_message() {
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("first"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("resumed"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        loop {
            match events.recv().await.unwrap() {
                AgentLoopEvent::TurnCommitted { messages }
                    if first_user_text(&messages[0]).is_some() =>
                {
                    handle.resume().unwrap();
                    break;
                }
                AgentLoopEvent::LoopEnded { .. } => {
                    panic!("loop ended before resume could be queued")
                }
                _ => {}
            }
        }

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(result.last_response.as_deref(), Some("resumed"));
        assert_eq!(model.request_count(), 2);
        assert_eq!(user_texts(result.history.iter()), vec!["start"]);
        assert_eq!(
            assistant_texts(result.history.iter()),
            vec!["first", "resumed"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn resume_is_noop_while_turn_is_running() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model.clone())
            .tool(BlockingTool {
                entered: entered.clone(),
                release: release.clone(),
            })
            .build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));

        entered.notified().await;
        handle.resume().unwrap();
        release.notify_one();

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(result.last_response.as_deref(), Some("done"));
        assert_eq!(model.request_count(), 2);
    }

    #[test]
    fn resume_history_requires_valid_committed_context() {
        assert!(validate_resume_history(&[]).is_err());
        assert!(validate_resume_history(&[Message::system("summary")]).is_err());
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
    async fn turn_hook_receives_candidate_history_and_new_messages() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize, Option<String>)>::new()));
        let hook_seen = seen.clone();
        let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
            let hook_seen = hook_seen.clone();
            let first_user = turn.new_messages.first().and_then(first_user_text);
            async move {
                hook_seen.lock().unwrap().push((
                    turn.history.len(),
                    turn.new_messages.len(),
                    first_user,
                ));
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        });

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        let seen = seen.lock().unwrap();
        assert_eq!(&seen[..], &[(2, 2, Some("start".to_string()))]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_hook_abort_stops_after_committing_the_turn() {
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("first"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            [
                MockStreamEvent::text("unused"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent).with_turn_hook(|_, _turn| async {
            Ok::<_, std::io::Error>(TurnHookAction::Abort {
                reason: "waiting for human".to_string(),
            })
        });

        let handle = agent_loop.prompt(Message::user("start"));
        handle.follow_up(Message::user("follow-up")).unwrap();

        let result = handle.wait().await.unwrap();

        assert!(matches!(
            result.end_reason,
            EndReason::AbortedByHook { reason } if reason == "waiting for human"
        ));
        assert_eq!(user_texts(result.history.iter()), vec!["start"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_hook_runs_for_abort_with_empty_append_batch() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model)
            .tool(BlockingTool {
                entered: entered.clone(),
                release: release.clone(),
            })
            .build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
        let hook_seen = seen.clone();
        let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
            let hook_seen = hook_seen.clone();
            async move {
                hook_seen
                    .lock()
                    .unwrap()
                    .push((turn.history.len(), turn.new_messages.len()));
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        });

        let handle = agent_loop.prompt(Message::user("start"));

        entered.notified().await;
        handle.abort().unwrap();
        let result = timeout(Duration::from_secs(1), handle.wait())
            .await
            .expect("abort should finish without waiting for an unanswered tool")
            .unwrap();
        release.notify_one();

        assert_eq!(result.end_reason, EndReason::Aborted);
        assert_eq!(&seen.lock().unwrap()[..], &[(0, 0)]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_timeout_covers_stream_setup_and_runs_hook() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let turn_timeout = Duration::from_millis(50);
        let agent = AgentBuilder::new(BlockingStreamModel {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
        let hook_seen = seen.clone();
        let agent_loop = AgentLoop::new(agent)
            .turn_timeout(turn_timeout)
            .with_turn_hook(move |_, turn| {
                let hook_seen = hook_seen.clone();
                async move {
                    hook_seen
                        .lock()
                        .unwrap()
                        .push((turn.history.len(), turn.new_messages.len()));
                    Ok::<_, std::io::Error>(TurnHookAction::Continue)
                }
            });

        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        entered.notified().await;
        let result = timeout(Duration::from_secs(1), handle.wait())
            .await
            .expect("turn timeout should finish while stream setup is blocked")
            .unwrap();
        release.notify_one();

        assert_eq!(
            result.end_reason,
            EndReason::TurnTimedOut {
                timeout: turn_timeout
            }
        );
        assert!(result.history.is_empty());
        assert_eq!(&seen.lock().unwrap()[..], &[(0, 0)]);
        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(event, AgentLoopEvent::TurnTimedOut { messages } if messages.is_empty())
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn turn_timeout_runs_hook_with_safe_partial_append() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let turn_timeout = Duration::from_millis(50);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model)
            .tool(BlockingTool {
                entered: entered.clone(),
                release: release.clone(),
            })
            .build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
        let hook_seen = seen.clone();
        let agent_loop = AgentLoop::new(agent)
            .turn_timeout(turn_timeout)
            .with_turn_hook(move |_, turn| {
                let hook_seen = hook_seen.clone();
                async move {
                    hook_seen
                        .lock()
                        .unwrap()
                        .push((turn.history.len(), turn.new_messages.len()));
                    Ok::<_, std::io::Error>(TurnHookAction::Continue)
                }
            });

        let handle = agent_loop.prompt(Message::user("start"));

        entered.notified().await;
        let result = timeout(Duration::from_secs(1), handle.wait())
            .await
            .expect("turn timeout should finish without waiting for tool completion")
            .unwrap();
        release.notify_one();

        assert_eq!(
            result.end_reason,
            EndReason::TurnTimedOut {
                timeout: turn_timeout
            }
        );
        assert!(result.history.is_empty());
        assert_eq!(&seen.lock().unwrap()[..], &[(0, 0)]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn loop_timeout_wins_over_turn_timeout_during_active_turn() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let loop_timeout = Duration::from_millis(50);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model)
            .tool(BlockingTool {
                entered: entered.clone(),
                release: release.clone(),
            })
            .build();
        let agent_loop = AgentLoop::new(agent)
            .turn_timeout(Duration::from_secs(5))
            .loop_timeout(loop_timeout);

        let handle = agent_loop.prompt(Message::user("start"));

        entered.notified().await;
        let result = timeout(Duration::from_secs(1), handle.wait())
            .await
            .expect("loop timeout should finish while turn is blocked")
            .unwrap();
        release.notify_one();

        assert_eq!(
            result.end_reason,
            EndReason::LoopTimedOut {
                timeout: loop_timeout
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn loop_timeout_while_idle_does_not_run_turn_hook_again() {
        let loop_timeout = Duration::from_millis(50);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model).build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
        let hook_seen = seen.clone();
        let agent_loop = AgentLoop::new(agent)
            .loop_timeout(loop_timeout)
            .with_turn_hook(move |_, turn| {
                let hook_seen = hook_seen.clone();
                async move {
                    hook_seen
                        .lock()
                        .unwrap()
                        .push((turn.history.len(), turn.new_messages.len()));
                    Ok::<_, std::io::Error>(TurnHookAction::Continue)
                }
            });
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        loop {
            match events.recv().await.unwrap() {
                AgentLoopEvent::TurnCommitted { .. } => break,
                AgentLoopEvent::LoopEnded { end_reason } => {
                    panic!("loop ended before commit: {end_reason:?}")
                }
                _ => {}
            }
        }

        sleep(loop_timeout + Duration::from_millis(50)).await;
        let result = handle.wait().await.unwrap();

        assert_eq!(
            result.end_reason,
            EndReason::LoopTimedOut {
                timeout: loop_timeout
            }
        );
        assert_eq!(&seen.lock().unwrap()[..], &[(2, 2)]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn provider_content_filter_error_returns_end_reason() {
        let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
            "content_filter: request blocked by policy",
        )]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert!(matches!(
            result.end_reason,
            EndReason::ContentFilter { error }
                if error.kind == ApiErrorKind::Provider
                    && error.message == "content_filter: request blocked by policy"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn provider_context_full_error_returns_end_reason() {
        let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
            "context_length_exceeded: maximum context length exceeded",
        )]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert!(matches!(
            result.end_reason,
            EndReason::ContextFull { error }
                if error.kind == ApiErrorKind::Provider
                    && error.message == "context_length_exceeded: maximum context length exceeded"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn provider_length_error_returns_end_reason() {
        let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
            "OpenAI response stream was incomplete: max_output_tokens",
        )]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert!(matches!(
            result.end_reason,
            EndReason::Length { error }
                if error.kind == ApiErrorKind::Provider
                    && error.message == "OpenAI response stream was incomplete: max_output_tokens"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn provider_api_error_returns_end_reason() {
        let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
            "server_error: response stream failed",
        )]]);
        let agent = AgentBuilder::new(model).build();
        let agent_loop = AgentLoop::new(agent);

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert!(matches!(
            result.end_reason,
            EndReason::ApiError { error }
                if error.kind == ApiErrorKind::Provider
                    && error.message == "server_error: response stream failed"
        ));
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
        let tool_call = test_tool_call("call_1");
        let tool_result = test_tool_result("call_1");

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

    #[test]
    fn partial_turn_preserves_text_before_completed_tool_result() {
        let mut partial = PartialTurn::default();

        partial.note_assistant_item(
            StreamedAssistantContent::<rig::test_utils::MockResponse>::Text(
                rig::message::Text::new("I will call the tool."),
            ),
        );
        partial.note_assistant_item(
            StreamedAssistantContent::<rig::test_utils::MockResponse>::ToolCall {
                internal_call_id: "internal_1".to_string(),
                tool_call: test_tool_call("call_1"),
            },
        );
        partial.note_user_item(StreamedUserContent::ToolResult {
            internal_call_id: "internal_1".to_string(),
            tool_result: test_tool_result("call_1"),
        });

        let messages = partial
            .append_messages(&Message::user("start"))
            .expect("completed tool results should produce history");
        let Message::Assistant { content, .. } = &messages[1] else {
            panic!("expected assistant message");
        };
        let items = content.iter().collect::<Vec<_>>();

        assert!(
            matches!(items[0], AssistantContent::Text(text) if text.text == "I will call the tool.")
        );
        assert!(
            matches!(items[1], AssistantContent::ToolCall(tool_call) if tool_call.id == "call_1")
        );
    }

    #[test]
    fn partial_turn_ignores_unanswered_tool_call() {
        let mut partial = PartialTurn::default();

        partial.note_assistant_item(
            StreamedAssistantContent::<rig::test_utils::MockResponse>::ToolCall {
                internal_call_id: "internal_1".to_string(),
                tool_call: test_tool_call("call_1"),
            },
        );

        assert!(partial.append_messages(&Message::user("start")).is_none());
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

    fn user_texts<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<String> {
        messages.into_iter().filter_map(first_user_text).collect()
    }

    fn assistant_texts<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<String> {
        messages
            .into_iter()
            .filter_map(|message| {
                let Message::Assistant { content, .. } = message else {
                    return None;
                };

                content.iter().find_map(|item| match item {
                    AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
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

    fn assistant_tool_call_message(id: &str) -> Message {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(test_tool_call(id))),
        }
    }

    fn test_tool_call(id: &str) -> ToolCall {
        ToolCall::new(
            id.to_string(),
            rig::message::ToolFunction::new("echo".to_string(), serde_json::json!({"text": "hi"})),
        )
    }

    fn test_tool_result(id: &str) -> ToolResult {
        ToolResult {
            id: id.to_string(),
            call_id: None,
            content: rig::message::ToolResultContent::from_tool_output("echoed".to_string()),
        }
    }

    #[derive(Clone)]
    struct BlockingTool {
        entered: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[derive(Debug)]
    struct BlockingToolError;

    impl fmt::Display for BlockingToolError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("blocking tool failed")
        }
    }

    impl Error for BlockingToolError {}

    impl Tool for BlockingTool {
        const NAME: &'static str = "blocking_tool";

        type Error = BlockingToolError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Blocks until the test releases it".to_string(),
                parameters: serde_json::json!({ "type": "object" }),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.entered.notify_one();
            self.release.notified().await;
            Ok("released".to_string())
        }
    }

    #[derive(Clone)]
    struct BlockingStreamModel {
        entered: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl CompletionModel for BlockingStreamModel {
        type Response = MockResponse;
        type StreamingResponse = MockResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self {
                entered: Arc::new(Notify::new()),
                release: Arc::new(Notify::new()),
            }
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            Err(CompletionError::ProviderError(
                "blocking stream model does not support non-streaming completion".to_string(),
            ))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            self.entered.notify_one();
            self.release.notified().await;
            let stream: rig::streaming::StreamingResult<Self::StreamingResponse> =
                Box::pin(futures::stream::empty::<
                    Result<RawStreamingChoice<MockResponse>, CompletionError>,
                >());
            Ok(StreamingCompletionResponse::stream(stream))
        }
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
