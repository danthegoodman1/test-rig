use std::{
    collections::{HashSet, VecDeque},
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
    message::{AssistantContent, Message, ToolCall, ToolResult, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt},
    wasm_compat::WasmCompatSend,
};
use tokio::{
    sync::{
        broadcast,
        mpsc::{self, error::TryRecvError},
    },
    task::{AbortHandle, JoinHandle},
    time::{Instant, sleep_until},
};

pub type TurnHookError = Box<dyn Error + Send + Sync + 'static>;
pub type AssistantMessageHookError = Box<dyn Error + Send + Sync + 'static>;

type TurnHookFuture = Pin<Box<dyn Future<Output = Result<TurnHookAction, TurnHookError>> + Send>>;
type TurnHook<S> = Arc<dyn Fn(Arc<S>, TurnHookContext) -> TurnHookFuture + Send + Sync>;
type AssistantMessageHookFuture =
    Pin<Box<dyn Future<Output = Result<(), AssistantMessageHookError>> + Send>>;
type AssistantMessageHook<S> =
    Arc<dyn Fn(Arc<S>, AssistantMessageContext) -> AssistantMessageHookFuture + Send + Sync>;
type SharedMessages = Arc<Mutex<Vec<Message>>>;
type SharedTurnRunning = Arc<AtomicBool>;
const EVENT_BUFFER_SIZE: usize = 1024;
const DEFAULT_TOOL_CRASH_MESSAGE: &str = "tool crashed before a result returned";

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
    assistant_message_hook: Option<AssistantMessageHook<S>>,
    unanswered_tool_call_repair: Option<String>,
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
            assistant_message_hook: None,
            unanswered_tool_call_repair: Some(DEFAULT_TOOL_CRASH_MESSAGE.to_string()),
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
        let assistant_message_hook = self.assistant_message_hook.map(|hook| {
            Arc::new(move |_state: Arc<S>, context| hook(Arc::new(()), context))
                as AssistantMessageHook<S>
        });

        AgentLoop {
            agent: self.agent,
            app_state: Arc::new(app_state),
            max_turns: self.max_turns,
            turn_timeout: self.turn_timeout,
            loop_timeout: self.loop_timeout,
            initial_history: self.initial_history,
            turn_hook,
            assistant_message_hook,
            unanswered_tool_call_repair: self.unanswered_tool_call_repair,
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

    /// Override the synthetic tool-result text used when repairing persisted
    /// assistant tool calls that do not have matching tool results.
    pub fn with_unanswered_tool_call_repair_message(mut self, message: impl Into<String>) -> Self {
        self.unanswered_tool_call_repair = Some(message.into());
        self
    }

    /// Observe each completed assistant message before it is part of a valid turn commit.
    ///
    /// When the assistant message contains tool calls, `context.new_messages`
    /// is not provider-valid until matching user tool results are appended or
    /// recovered by the default unanswered-tool-call history repair.
    pub fn on_assistant_message_finished<F, Fut, E>(mut self, hook: F) -> Self
    where
        F: Fn(Arc<S>, AssistantMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        self.assistant_message_hook = Some(Arc::new(move |state, context| {
            let future = hook(state, context);
            Box::pin(async move {
                future
                    .await
                    .map_err(|err| Box::new(err) as AssistantMessageHookError)
            })
        }));
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
        self.start(
            PendingTurn::single(prompt.into()),
            self.repaired_initial_history(),
        )
    }

    /// Start the loop from the seeded history without appending a new prompt.
    pub fn resume(&self) -> Result<AgentLoopHandle<M::StreamingResponse>, AgentLoopError> {
        let initial_history = self.repaired_initial_history();
        validate_resume_history(&initial_history).map_err(AgentLoopError::InvalidMessageHistory)?;
        Ok(self.start(PendingTurn::Resume, initial_history))
    }

    fn repaired_initial_history(&self) -> Vec<Message> {
        let history = self.initial_history.clone();
        match &self.unanswered_tool_call_repair {
            Some(message) => repair_unanswered_tool_calls(history, message),
            None => history,
        }
    }

    fn start(
        &self,
        initial_turn: PendingTurn,
        initial_history: Vec<Message>,
    ) -> AgentLoopHandle<M::StreamingResponse> {
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (events_tx, _) = broadcast::channel(EVENT_BUFFER_SIZE);
        let state = Arc::new(Mutex::new(initial_history));
        let turn_running = Arc::new(AtomicBool::new(false));
        let runner = Runner::new(RunnerInit {
            agent: self.agent.clone(),
            app_state: self.app_state.clone(),
            max_turns: self.max_turns,
            turn_timeout: self.turn_timeout,
            loop_timeout: self.loop_timeout,
            turn_hook: self.turn_hook.clone(),
            assistant_message_hook: self.assistant_message_hook.clone(),
            events_tx: events_tx.clone(),
            state: state.clone(),
            turn_running: turn_running.clone(),
            initial_turn,
        });
        let task = tokio::spawn(async move { runner.run(commands_rx).await });

        AgentLoopHandle {
            commands_tx: Some(commands_tx),
            events_tx,
            state,
            turn_running,
            task: Some(task),
        }
    }
}

/// A running agent loop.
pub struct AgentLoopHandle<R>
where
    R: Clone,
{
    commands_tx: Option<mpsc::UnboundedSender<Command>>,
    events_tx: broadcast::Sender<AgentLoopEvent<R>>,
    state: SharedMessages,
    turn_running: SharedTurnRunning,
    task: Option<JoinHandle<Result<AgentLoopResult, AgentLoopError>>>,
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
    pub async fn wait(mut self) -> Result<AgentLoopResult, AgentLoopError> {
        drop(self.commands_tx.take());
        let task = self.task.take().expect("agent loop task is missing");
        let mut abort_on_drop = AbortOnDrop::new(task.abort_handle());

        let result = match task.await {
            Ok(result) => result,
            Err(err) => Err(AgentLoopError::TaskJoin(err)),
        };
        abort_on_drop.disarm();
        result
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

        validate_resume_history(&self.state()).map_err(AgentLoopError::InvalidMessageHistory)?;
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
            .as_ref()
            .ok_or(AgentLoopError::CommandChannelClosed)?
            .send(command)
            .map_err(|_| AgentLoopError::CommandChannelClosed)
    }

    fn emit(&self, event: AgentLoopEvent<R>) {
        let _ = self.events_tx.send(event);
    }
}

impl<R> Drop for AgentLoopHandle<R>
where
    R: Clone,
{
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

struct AbortOnDrop {
    handle: Option<AbortHandle>,
}

impl AbortOnDrop {
    fn new(handle: AbortHandle) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    fn disarm(&mut self) {
        self.handle = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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

#[derive(Clone, Debug)]
pub struct AssistantMessageContext {
    /// The assistant message that just finished streaming.
    pub message: Message,
    /// The candidate full history after appending `new_messages`.
    ///
    /// This history may be temporarily invalid for provider replay when the
    /// assistant message contains tool calls without matching tool results.
    pub history: Vec<Message>,
    /// The ordered partial append batch ending with `message`.
    pub new_messages: Vec<Message>,
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    AssistantMessageFinished {
        message: Message,
        messages: Vec<Message>,
    },
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
    LoopFailed {
        error: AgentLoopErrorInfo,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AgentLoopErrorInfo {
    pub message: String,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum AgentLoopError {
    CommandChannelClosed,
    InvalidMessageHistory(InvalidMessageHistoryError),
    TurnHook(TurnHookError),
    AssistantMessageHook(AssistantMessageHookError),
    Rig(StreamingError),
    TaskJoin(tokio::task::JoinError),
}

impl fmt::Display for AgentLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandChannelClosed => f.write_str("agent loop command channel is closed"),
            Self::InvalidMessageHistory(err) => write!(f, "{err}"),
            Self::TurnHook(err) => write!(f, "turn hook failed: {err}"),
            Self::AssistantMessageHook(err) => write!(f, "assistant message hook failed: {err}"),
            Self::Rig(err) => write!(f, "{err}"),
            Self::TaskJoin(err) => write!(f, "{err}"),
        }
    }
}

impl Error for AgentLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidMessageHistory(err) => Some(err),
            Self::TurnHook(err) => Some(err.as_ref()),
            Self::AssistantMessageHook(err) => Some(err.as_ref()),
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

impl AgentLoopErrorInfo {
    fn from_error(err: &AgentLoopError) -> Self {
        Self {
            message: err.to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidMessageHistoryError {
    pub message: String,
}

impl InvalidMessageHistoryError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for InvalidMessageHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid Rig message history: {}", self.message)
    }
}

impl Error for InvalidMessageHistoryError {}

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

async fn poll_assistant_message_hook(
    hook: &mut Option<AssistantMessageHookTask>,
) -> Result<(), AgentLoopError> {
    match hook {
        Some(hook) => match (&mut hook.handle).await {
            Ok(result) => result,
            Err(err) => Err(AgentLoopError::TaskJoin(err)),
        },
        None => std::future::pending().await,
    }
}

async fn wait_for_assistant_message_hook(
    hook: &mut Option<AssistantMessageHookTask>,
) -> Result<(), AgentLoopError> {
    if let Some(mut hook) = hook.take() {
        match (&mut hook.handle).await {
            Ok(result) => result?,
            Err(err) => return Err(AgentLoopError::TaskJoin(err)),
        }
    }

    Ok(())
}

struct AssistantMessageHookTask {
    handle: JoinHandle<Result<(), AgentLoopError>>,
}

impl AssistantMessageHookTask {
    fn spawn(hook: AssistantMessageHookFuture) -> Self {
        Self {
            handle: tokio::spawn(async move {
                hook.await.map_err(AgentLoopError::AssistantMessageHook)
            }),
        }
    }
}

impl Drop for AssistantMessageHookTask {
    fn drop(&mut self) {
        self.handle.abort();
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
    assistant_message_hook: Option<AssistantMessageHook<S>>,
    events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
    state: SharedMessages,
    turn_running: SharedTurnRunning,
    immediate: VecDeque<Message>,
    steering: VecDeque<Message>,
    resumes: usize,
    follow_ups: VecDeque<Message>,
    last_response: Option<String>,
}

struct RunnerInit<M, P, S>
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
    assistant_message_hook: Option<AssistantMessageHook<S>>,
    events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
    state: SharedMessages,
    turn_running: SharedTurnRunning,
    initial_turn: PendingTurn,
}

impl<M, P, S> Runner<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn new(init: RunnerInit<M, P, S>) -> Self {
        let (immediate, resumes) = match init.initial_turn {
            PendingTurn::Prompt { prelude, prompt } => {
                debug_assert!(prelude.is_empty());
                (VecDeque::from([*prompt]), 0)
            }
            PendingTurn::Resume => (VecDeque::new(), 1),
        };

        Self {
            agent: init.agent,
            app_state: init.app_state,
            max_turns: init.max_turns,
            turn_timeout: init.turn_timeout,
            loop_timeout: init.loop_timeout,
            turn_hook: init.turn_hook,
            assistant_message_hook: init.assistant_message_hook,
            events_tx: init.events_tx,
            state: init.state,
            turn_running: init.turn_running,
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
        prompt: Box<Message>,
    },
    Resume,
}

impl PendingTurn {
    fn new(mut messages: Vec<Message>) -> Option<Self> {
        let prompt = messages.pop()?;
        Some(Self::Prompt {
            prelude: messages,
            prompt: Box::new(prompt),
        })
    }

    fn single(prompt: Message) -> Self {
        Self::Prompt {
            prelude: Vec::new(),
            prompt: Box::new(prompt),
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
                let prompt = *prompt;
                let mut request_history = committed_base_history.clone();
                request_history.extend(prelude.clone());
                let mut full_request_history = request_history.clone();
                full_request_history.push(prompt.clone());
                validate_message_history(&full_request_history)
                    .map_err(AgentLoopError::InvalidMessageHistory)?;

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
                    .map_err(AgentLoopError::InvalidMessageHistory)?;
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

    fn assistant_messages(&self, partial_turn: &PartialTurn, message: Message) -> Vec<Message> {
        let mut messages = Vec::new();
        if !self.prelude.is_empty() {
            messages.extend(self.prelude.clone());
        }
        if self.commit_prompt {
            messages.push(self.prompt.clone());
        }
        messages.extend(partial_turn.completed_messages.clone());
        messages.push(message);
        messages
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
        let result = self.run_inner(&mut commands_rx).await;
        if let Err(err) = &result {
            self.set_turn_running(false);
            self.emit(AgentLoopEvent::LoopFailed {
                error: AgentLoopErrorInfo::from_error(err),
            });
        }

        result
    }

    async fn run_inner(
        &mut self,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
    ) -> Result<AgentLoopResult, AgentLoopError> {
        self.emit(AgentLoopEvent::LoopStarted);
        let loop_deadline = self
            .loop_timeout
            .map(|timeout| TimeoutDeadline::new(TimeoutKind::Loop, timeout));

        loop {
            if let Some(deadline) = elapsed_timeout(loop_deadline) {
                return Ok(self.finish(deadline.end_reason()));
            }

            if self.drain_ready_commands(commands_rx) == CommandAction::Abort {
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

            if self.drain_ready_commands(commands_rx) == CommandAction::Abort {
                return Ok(self.finish(EndReason::Aborted));
            }

            let Some(turn) = self.next_turn() else {
                continue;
            };

            match self.run_turn(turn, commands_rx, loop_deadline).await? {
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
        let mut assistant_message_hook = None;
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
                wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
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
                    wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                    return self.timeout_turn(&turn, &partial_turn, deadline).await;
                }
                stream = &mut stream_future => break stream,
            }
        };

        loop {
            let deadline = next_timeout(turn_deadline, loop_deadline);
            if let Some(deadline) = elapsed_timeout(deadline) {
                wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                return self.timeout_turn(&turn, &partial_turn, deadline).await;
            }

            tokio::select! {
                command = commands_rx.recv(), if !commands_closed => {
                    let Some(command) = command else {
                        commands_closed = true;
                        continue;
                    };

                    if matches!(command, Command::Interrupt(_) | Command::Abort) {
                        wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                    }
                    if let Some(action) = self
                        .handle_turn_command(command, &turn, &partial_turn)
                        .await?
                    {
                        return Ok(action);
                    }
                }
                _ = wait_for_timeout(deadline), if deadline.is_some() => {
                    let deadline = deadline.expect("timeout branch requires a deadline");
                    wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                    return self.timeout_turn(&turn, &partial_turn, deadline).await;
                }
                hook = poll_assistant_message_hook(&mut assistant_message_hook), if assistant_message_hook.is_some() => {
                    assistant_message_hook = None;
                    hook?;
                }
                item = stream.next() => {
                    let Some(item) = item else {
                        wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                        self.set_turn_running(false);
                        return Ok(PromptAction::Continue);
                    };

                    match item {
                        Ok(item) => {
                            self.emit(AgentLoopEvent::Rig(item.clone()));
                            if let Some(end_reason) = self
                                .handle_stream_item(
                                    item,
                                    &turn,
                                    &mut partial_turn,
                                    &mut assistant_message_hook,
                                )
                                .await?
                            {
                                return Ok(PromptAction::Finish(end_reason));
                            }
                        }
                        Err(err) => {
                            wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                            if let Some(history) = history_from_error(&err) {
                                if let CommitOutcome::Abort(end_reason) = self
                                    .commit_recovered_history(&turn.committed_base_history, history)
                                    .await?
                                {
                                    return Ok(PromptAction::Finish(end_reason));
                                }
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
                    .commit_append(messages, CommitEvent::Interrupted)
                    .await?
                {
                    return Ok(Some(PromptAction::Finish(end_reason)));
                }
                self.immediate.push_front(message);
                return Ok(Some(PromptAction::Continue));
            }
            Command::Abort => {
                let messages = turn.partial_messages(partial_turn);
                self.commit_append(messages, CommitEvent::Aborted).await?;
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
        self.commit_append(messages, CommitEvent::TimedOut).await?;
        Ok(PromptAction::Finish(deadline.end_reason()))
    }

    async fn handle_stream_item(
        &mut self,
        item: MultiTurnStreamItem<M::StreamingResponse>,
        turn: &PreparedTurn,
        partial_turn: &mut PartialTurn,
        assistant_message_hook: &mut Option<AssistantMessageHookTask>,
    ) -> Result<Option<EndReason>, AgentLoopError> {
        match item {
            MultiTurnStreamItem::StreamAssistantItem(item) => {
                let is_message_boundary = matches!(item, StreamedAssistantContent::ToolCall { .. });
                partial_turn.note_assistant_item(item);
                if is_message_boundary {
                    self.start_assistant_message_hook(turn, partial_turn, assistant_message_hook)
                        .await?;
                }
            }
            MultiTurnStreamItem::StreamUserItem(item) => partial_turn.note_user_item(item),
            MultiTurnStreamItem::FinalResponse(final_response) => {
                self.start_assistant_message_hook(turn, partial_turn, assistant_message_hook)
                    .await?;
                wait_for_assistant_message_hook(assistant_message_hook).await?;
                self.last_response = Some(final_response.response().to_string());
                if let Some(messages) = final_response.history()
                    && let CommitOutcome::Abort(end_reason) = self
                        .commit_append(
                            turn.append_messages(messages.to_vec()),
                            CommitEvent::Committed,
                        )
                        .await?
                {
                    return Ok(Some(end_reason));
                }
            }
            MultiTurnStreamItem::CompletionCall(_) => {}
            _ => {}
        }

        Ok(None)
    }

    async fn start_assistant_message_hook(
        &self,
        turn: &PreparedTurn,
        partial_turn: &mut PartialTurn,
        assistant_message_hook: &mut Option<AssistantMessageHookTask>,
    ) -> Result<(), AgentLoopError> {
        let Some(message) = partial_turn.assistant_message() else {
            return Ok(());
        };

        wait_for_assistant_message_hook(assistant_message_hook).await?;

        let messages = turn.assistant_messages(partial_turn, message.clone());
        let mut history = self.history_snapshot();
        history.extend(messages.clone());
        partial_turn.mark_assistant_message_finished(message.clone());
        self.emit(AgentLoopEvent::AssistantMessageFinished {
            message: message.clone(),
            messages: messages.clone(),
        });

        let Some(hook) = &self.assistant_message_hook else {
            return Ok(());
        };

        let context = AssistantMessageContext {
            message,
            history,
            new_messages: messages,
        };
        *assistant_message_hook = Some(AssistantMessageHookTask::spawn(hook(
            self.app_state.clone(),
            context,
        )));
        Ok(())
    }

    async fn commit_append(
        &mut self,
        messages: Vec<Message>,
        event: CommitEvent,
    ) -> Result<CommitOutcome, AgentLoopError> {
        let before = self.history_snapshot();
        let mut next_history = before;
        next_history.extend(messages.clone());
        validate_message_history(&next_history).map_err(AgentLoopError::InvalidMessageHistory)?;

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
                    validate_message_history(&history)
                        .map_err(AgentLoopError::InvalidMessageHistory)?;
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
    ) -> Result<CommitOutcome, AgentLoopError> {
        validate_message_history(&recovered_history)
            .map_err(AgentLoopError::InvalidMessageHistory)?;

        if !recovered_history.starts_with(base_history) {
            return Err(AgentLoopError::InvalidMessageHistory(
                InvalidMessageHistoryError::new(
                    "recovered history does not extend the committed base history",
                ),
            ));
        }

        let append = recovered_history[base_history.len()..].to_vec();
        self.commit_append(append, CommitEvent::Committed).await
    }

    fn finish(&mut self, end_reason: EndReason) -> AgentLoopResult {
        self.emit(AgentLoopEvent::LoopEnded {
            end_reason: end_reason.clone(),
        });

        AgentLoopResult {
            end_reason,
            history: self.history_snapshot(),
            last_response: self.last_response.take(),
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
    Committed,
    Interrupted,
    Aborted,
    TimedOut,
}

impl CommitEvent {
    fn into_agent_event<R>(self, messages: Vec<Message>) -> AgentLoopEvent<R> {
        match self {
            Self::Committed => AgentLoopEvent::TurnCommitted { messages },
            Self::Interrupted => AgentLoopEvent::TurnInterrupted { messages },
            Self::Aborted => AgentLoopEvent::TurnAborted { messages },
            Self::TimedOut => AgentLoopEvent::TurnTimedOut { messages },
        }
    }
}

#[derive(Default)]
struct PartialTurn {
    pending_tool_calls: Vec<PendingToolCall>,
    assistant_content: Vec<AssistantContent>,
    finished_assistant_message: Option<Message>,
    active_tool_results: Vec<UserContent>,
    completed_messages: Vec<Message>,
}

struct PendingToolCall {
    internal_call_id: String,
    tool_call: ToolCall,
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
                self.pending_tool_calls.push(PendingToolCall {
                    internal_call_id,
                    tool_call,
                });
            }
            StreamedAssistantContent::ToolCallDelta { .. }
            | StreamedAssistantContent::ReasoningDelta { .. }
            | StreamedAssistantContent::Final(_) => {}
        }
    }

    fn assistant_message(&self) -> Option<Message> {
        if self.finished_assistant_message.is_some() || !self.active_tool_results.is_empty() {
            return None;
        }

        let mut content = self.assistant_content.clone();
        content.extend(
            self.pending_tool_calls
                .iter()
                .map(|call| AssistantContent::ToolCall(call.tool_call.clone())),
        );

        if content.is_empty() {
            return None;
        }

        OneOrMany::many(content)
            .ok()
            .map(|content| Message::Assistant { id: None, content })
    }

    fn mark_assistant_message_finished(&mut self, message: Message) {
        self.finished_assistant_message = Some(message);
    }

    fn note_user_item(&mut self, item: StreamedUserContent) {
        let StreamedUserContent::ToolResult {
            internal_call_id,
            tool_result,
        } = item;

        let Some(position) = self
            .pending_tool_calls
            .iter()
            .position(|call| call.internal_call_id == internal_call_id)
        else {
            return;
        };

        if self.active_tool_results.is_empty() {
            let assistant_message = self
                .finished_assistant_message
                .take()
                .or_else(|| self.assistant_message());
            if let Some(message) = assistant_message {
                self.assistant_content.clear();
                self.completed_messages.push(message);
            }
        }

        self.pending_tool_calls.remove(position);
        self.active_tool_results
            .push(UserContent::ToolResult(tool_result));

        if self.pending_tool_calls.is_empty()
            && let Ok(content) = OneOrMany::many(std::mem::take(&mut self.active_tool_results))
        {
            self.completed_messages.push(Message::User { content });
        }
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

fn validate_message_history(messages: &[Message]) -> Result<(), InvalidMessageHistoryError> {
    let mut pending_tool_calls: Option<(usize, Vec<String>)> = None;

    for (index, message) in messages.iter().enumerate() {
        if let Some((assistant_index, expected_ids)) = pending_tool_calls.take() {
            let Message::User { content } = message else {
                return Err(InvalidMessageHistoryError::new(format!(
                    "assistant message {assistant_index} contains tool calls, but message {index} is not the required user tool-result message"
                )));
            };

            let result_ids = tool_result_ids(content);
            if result_ids.is_empty() {
                return Err(InvalidMessageHistoryError::new(format!(
                    "assistant message {assistant_index} contains tool calls, but user message {index} contains no tool results"
                )));
            }

            ensure_unique_ids(&result_ids, index, "tool result")?;

            for id in &expected_ids {
                if !result_ids.contains(id) {
                    return Err(InvalidMessageHistoryError::new(format!(
                        "assistant tool call `{id}` at message {assistant_index} is missing a matching tool result in message {index}"
                    )));
                }
            }

            for id in &result_ids {
                if !expected_ids.contains(id) {
                    return Err(InvalidMessageHistoryError::new(format!(
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
                    return Err(InvalidMessageHistoryError::new(format!(
                        "tool result `{id}` at message {index} has no immediately preceding assistant tool call"
                    )));
                }
            }
            Message::System { .. } => {}
        }
    }

    if let Some((assistant_index, expected_ids)) = pending_tool_calls {
        return Err(InvalidMessageHistoryError::new(format!(
            "assistant message {assistant_index} contains unanswered tool calls: {}",
            expected_ids.join(", ")
        )));
    }

    Ok(())
}

fn validate_resume_history(messages: &[Message]) -> Result<(), InvalidMessageHistoryError> {
    validate_message_history(messages)?;

    match messages.last() {
        Some(Message::User { .. } | Message::Assistant { .. }) => Ok(()),
        Some(Message::System { .. }) => Err(InvalidMessageHistoryError::new(
            "resume requires the last committed message to be user or assistant content",
        )),
        None => Err(InvalidMessageHistoryError::new(
            "resume requires at least one committed message",
        )),
    }
}

fn repair_unanswered_tool_calls(mut messages: Vec<Message>, repair_message: &str) -> Vec<Message> {
    let mut index = 0;
    while index < messages.len() {
        let expected_ids = match &messages[index] {
            Message::Assistant { content, .. } => assistant_tool_call_ids(content),
            Message::User { .. } | Message::System { .. } => {
                index += 1;
                continue;
            }
        };

        if expected_ids.is_empty() {
            index += 1;
            continue;
        }

        let next_index = index + 1;
        let Some(next_message) = messages.get_mut(next_index) else {
            messages.push(repair_tool_result_message(&expected_ids, repair_message));
            index += 2;
            continue;
        };

        let Message::User { content } = next_message else {
            messages.insert(
                next_index,
                repair_tool_result_message(&expected_ids, repair_message),
            );
            index += 2;
            continue;
        };

        let result_ids = tool_result_ids(content);
        if result_ids.is_empty() {
            messages.insert(
                next_index,
                repair_tool_result_message(&expected_ids, repair_message),
            );
            index += 2;
            continue;
        }

        let missing_ids = expected_ids
            .iter()
            .filter(|id| !result_ids.contains(id))
            .cloned()
            .collect::<Vec<_>>();

        if missing_ids.is_empty() {
            index += 2;
            continue;
        }

        let mut repaired_content = content.iter().cloned().collect::<Vec<_>>();
        repaired_content.extend(repair_tool_result_contents(&missing_ids, repair_message));
        if let Ok(next_content) = OneOrMany::many(repaired_content) {
            *content = next_content;
        }
        index += 2;
    }

    messages
}

fn repair_tool_result_message(ids: &[String], repair_message: &str) -> Message {
    Message::User {
        content: OneOrMany::many(repair_tool_result_contents(ids, repair_message))
            .expect("repair tool result content is non-empty"),
    }
}

fn repair_tool_result_contents(ids: &[String], repair_message: &str) -> Vec<UserContent> {
    ids.iter()
        .map(|id| {
            UserContent::ToolResult(ToolResult {
                id: id.clone(),
                call_id: None,
                content: ToolResultContent::from_tool_output(repair_message.to_string()),
            })
        })
        .collect()
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
) -> Result<(), InvalidMessageHistoryError> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(InvalidMessageHistoryError::new(format!(
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
    use std::sync::{Arc, Mutex, atomic::AtomicBool};
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
            Ok(_) => panic!("invalid message history should not start a resumed loop"),
            Err(err) => err,
        };

        assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unanswered_tool_call_history_is_repaired_before_prompt() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("recovered"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop =
            AgentLoop::new(agent).with_history([assistant_tool_call_message("call_1")]);

        let result = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        assert_eq!(model.request_count(), 1);
        let requests = model.requests();
        assert_eq!(
            tool_result_texts(requests[0].chat_history.iter()),
            vec![DEFAULT_TOOL_CRASH_MESSAGE]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unanswered_tool_call_repair_message_can_be_overridden() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("recovered"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent)
            .with_history([assistant_tool_call_message("call_1")])
            .with_unanswered_tool_call_repair_message("custom tool crash message");

        agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .unwrap();

        let requests = model.requests();
        assert_eq!(
            tool_result_texts(requests[0].chat_history.iter()),
            vec!["custom tool crash message"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_message_history_with_orphan_tool_result_still_fails_before_request() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("unused"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent).with_history([Message::User {
            content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_1"))),
        }]);

        let err = agent_loop
            .prompt(Message::user("start"))
            .wait()
            .await
            .expect_err("unrepairable message history should fail before a request is sent");

        assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
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
        let agent_loop = AgentLoop::new(agent)
            .with_turn_hook(|_, _turn| async { Err(std::io::Error::other("hook failed")) });
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        let err = handle
            .wait()
            .await
            .expect_err("turn hook failure should fail the run");

        assert!(matches!(err, AgentLoopError::TurnHook(_)));
        assert_eq!(model.request_count(), 1);

        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::LoopFailed { error }
                    if error.message.contains("turn hook failed: hook failed")
            )
        }));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::LoopEnded { .. }))
        );
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

        assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
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

    #[test]
    fn repair_unanswered_tool_calls_adds_missing_results_to_next_user_message() {
        let history = vec![
            Message::Assistant {
                id: None,
                content: OneOrMany::many(vec![
                    AssistantContent::ToolCall(test_tool_call("call_1")),
                    AssistantContent::ToolCall(test_tool_call("call_2")),
                ])
                .unwrap(),
            },
            Message::User {
                content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_1"))),
            },
        ];

        let repaired = repair_unanswered_tool_calls(history, "tool repaired");

        validate_message_history(&repaired).unwrap();
        assert_eq!(
            tool_result_texts(repaired.iter()),
            vec!["echoed", "tool repaired"]
        );
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
    async fn dropping_handle_aborts_running_loop() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let agent = AgentBuilder::new(BlockingStreamModel {
            entered: entered.clone(),
            release,
            dropped: dropped.clone(),
        })
        .build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        entered.notified().await;
        drop(handle);

        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .expect("dropping the handle should abort the in-flight stream");
        let events = drain_events(&mut events);
        assert!(!events.iter().any(|event| matches!(
            event,
            AgentLoopEvent::LoopEnded { .. } | AgentLoopEvent::LoopFailed { .. }
        )));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_wait_aborts_running_loop() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let agent = AgentBuilder::new(BlockingStreamModel {
            entered: entered.clone(),
            release,
            dropped: dropped.clone(),
        })
        .build();
        let agent_loop = AgentLoop::new(agent);
        let handle = agent_loop.prompt(Message::user("start"));

        entered.notified().await;
        let mut wait = Box::pin(handle.wait());
        assert!(futures::poll!(wait.as_mut()).is_pending());
        drop(wait);

        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .expect("cancelling wait should abort the in-flight stream");
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
    async fn assistant_message_hook_runs_while_tool_execution_is_in_flight() {
        let hook_entered = Arc::new(Notify::new());
        let hook_release = Arc::new(Notify::new());
        let tool_entered = Arc::new(Notify::new());
        let tool_release = Arc::new(Notify::new());
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("I will call the tool."),
                MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
                MockStreamEvent::final_response_with_default_usage(),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_default_usage(),
            ],
        ]);
        let agent = AgentBuilder::new(model)
            .tool(BlockingTool {
                entered: tool_entered.clone(),
                release: tool_release.clone(),
            })
            .build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize, usize)>::new()));
        let hook_seen = seen.clone();
        let hook_entered_for_hook = hook_entered.clone();
        let hook_release_for_hook = hook_release.clone();
        let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(move |_, context| {
            let hook_seen = hook_seen.clone();
            let hook_entered = hook_entered_for_hook.clone();
            let hook_release = hook_release_for_hook.clone();
            async move {
                let tool_call_count = match &context.message {
                    Message::Assistant { content, .. } => assistant_tool_call_ids(content).len(),
                    _ => 0,
                };
                hook_seen.lock().unwrap().push((
                    context.history.len(),
                    context.new_messages.len(),
                    tool_call_count,
                ));
                if tool_call_count > 0 {
                    hook_entered.notify_one();
                    hook_release.notified().await;
                }
                Ok::<_, std::io::Error>(())
            }
        });

        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        timeout(Duration::from_secs(1), hook_entered.notified())
            .await
            .expect("assistant message hook should start");
        timeout(Duration::from_secs(1), tool_entered.notified())
            .await
            .expect("tool execution should start while hook is still pending");

        hook_release.notify_one();
        tool_release.notify_one();
        let result = timeout(Duration::from_secs(1), handle.wait())
            .await
            .expect("loop should finish after hook and tool are released")
            .unwrap();

        assert_eq!(result.end_reason, EndReason::Idle);
        let seen = seen.lock().unwrap();
        assert!(seen.contains(&(2, 2, 1)));
        assert!(seen.contains(&(4, 4, 0)));
        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::AssistantMessageFinished { message, messages }
                    if matches!(message, Message::Assistant { .. }) && messages.len() == 2
            )
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn assistant_message_snapshot_can_be_repaired_after_restart() {
        let persisted_messages = Arc::new(Mutex::new(None::<Vec<Message>>));
        let persisted = Arc::new(Notify::new());
        let tool_entered = Arc::new(Notify::new());
        let tool_release = Arc::new(Notify::new());
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("I will call the tool."),
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model)
            .tool(BlockingTool {
                entered: tool_entered.clone(),
                release: tool_release.clone(),
            })
            .build();
        let messages_for_hook = persisted_messages.clone();
        let persisted_for_hook = persisted.clone();
        let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(move |_, context| {
            let messages_for_hook = messages_for_hook.clone();
            let persisted = persisted_for_hook.clone();
            async move {
                let has_tool_call = matches!(
                    &context.message,
                    Message::Assistant { content, .. } if !assistant_tool_call_ids(content).is_empty()
                );
                if has_tool_call {
                    *messages_for_hook.lock().unwrap() = Some(context.new_messages);
                    persisted.notify_one();
                }
                Ok::<_, std::io::Error>(())
            }
        });

        let handle = agent_loop.prompt(Message::user("start"));

        timeout(Duration::from_secs(1), persisted.notified())
            .await
            .expect("assistant snapshot should be persisted before tool result");
        timeout(Duration::from_secs(1), tool_entered.notified())
            .await
            .expect("tool should be running when the process crashes");

        drop(handle);
        tool_release.notify_waiters();

        let partial_history = persisted_messages
            .lock()
            .unwrap()
            .clone()
            .expect("assistant snapshot should be captured");
        assert!(validate_message_history(&partial_history).is_err());

        let recovery_model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("recovered"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let recovery_agent = AgentBuilder::new(recovery_model.clone()).build();
        let result = AgentLoop::new(recovery_agent)
            .with_history(partial_history)
            .resume()
            .unwrap()
            .wait()
            .await
            .unwrap();

        validate_message_history(&result.history).unwrap();
        assert_eq!(
            tool_result_texts(result.history.iter()),
            vec![DEFAULT_TOOL_CRASH_MESSAGE]
        );
        assert_eq!(
            assistant_texts(result.history.iter()),
            vec!["I will call the tool.", "recovered"]
        );
        assert_eq!(recovery_model.request_count(), 1);

        let requests = recovery_model.requests();
        assert_eq!(
            tool_result_texts(requests[0].chat_history.iter()),
            vec![DEFAULT_TOOL_CRASH_MESSAGE]
        );
        let tool_call_ids = requests[0]
            .chat_history
            .iter()
            .flat_map(|message| match message {
                Message::Assistant { content, .. } => assistant_tool_call_ids(content),
                Message::User { .. } | Message::System { .. } => Vec::new(),
            })
            .collect::<Vec<_>>();
        assert_eq!(tool_call_ids, vec!["call_1"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn assistant_message_hook_error_stops_the_loop() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = AgentBuilder::new(model.clone()).build();
        let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(|_, _context| async {
            Err(std::io::Error::other("assistant hook failed"))
        });
        let handle = agent_loop.prompt(Message::user("start"));
        let mut events = handle.subscribe();

        let err = handle
            .wait()
            .await
            .expect_err("assistant message hook failure should fail the run");

        assert!(matches!(err, AgentLoopError::AssistantMessageHook(_)));
        assert_eq!(model.request_count(), 1);

        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentLoopEvent::LoopFailed { error }
                    if error.message.contains("assistant message hook failed: assistant hook failed")
            )
        }));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::LoopEnded { .. }))
        );
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
            dropped: Arc::new(Notify::new()),
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
    async fn recovered_history_from_max_turns_commits_through_hook() {
        let model = MockCompletionModel::from_stream_turns([
            [MockStreamEvent::tool_call(
                "call_1",
                "add",
                serde_json::json!({"x": 1, "y": 2}),
            )],
            [MockStreamEvent::tool_call(
                "call_2",
                "add",
                serde_json::json!({"x": 3, "y": 4}),
            )],
            [MockStreamEvent::tool_call(
                "call_3",
                "add",
                serde_json::json!({"x": 5, "y": 6}),
            )],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
        let hook_seen = seen.clone();
        let agent_loop = AgentLoop::new(agent)
            .max_turns(1)
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

        let result = handle.wait().await.unwrap();

        assert_eq!(result.end_reason, EndReason::MaxTurns { max_turns: 1 });
        let committed_len = result.history.len();
        assert!(committed_len >= 3);
        assert_eq!(&seen.lock().unwrap()[..], &[(committed_len, committed_len)]);
        let events = drain_events(&mut events);
        assert!(events.iter().any(|event| {
            matches!(event, AgentLoopEvent::TurnCommitted { messages } if messages.len() == committed_len)
        }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recovered_history_must_extend_committed_base() {
        let base = vec![Message::user("start"), Message::assistant("base")];
        let state = Arc::new(Mutex::new(base.clone()));
        let (events_tx, mut events) = broadcast::channel(EVENT_BUFFER_SIZE);
        let mut runner = test_runner(state.clone(), events_tx);

        for recovered_history in [
            vec![Message::user("different")],
            vec![Message::user("start"), Message::assistant("different")],
        ] {
            let err = runner
                .commit_recovered_history(&base, recovered_history)
                .await
                .expect_err("non-appendable recovered history should fail");
            assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
            assert_eq!(*lock_messages(&state), base);
        }

        let events = drain_events(&mut events);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::TurnCommitted { .. }))
        );
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

    fn tool_result_texts<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<String> {
        messages
            .into_iter()
            .flat_map(|message| {
                let Message::User { content } = message else {
                    return Vec::new();
                };

                content
                    .iter()
                    .filter_map(|item| {
                        let UserContent::ToolResult(tool_result) = item else {
                            return None;
                        };

                        tool_result
                            .content
                            .iter()
                            .find_map(|content| match content {
                                ToolResultContent::Text(text) => Some(text.text.clone()),
                                ToolResultContent::Image(_) => None,
                            })
                    })
                    .collect::<Vec<_>>()
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
        dropped: Arc<Notify>,
    }

    impl CompletionModel for BlockingStreamModel {
        type Response = MockResponse;
        type StreamingResponse = MockResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self {
                entered: Arc::new(Notify::new()),
                release: Arc::new(Notify::new()),
                dropped: Arc::new(Notify::new()),
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
            let _drop = NotifyOnDrop(self.dropped.clone());
            self.entered.notify_one();
            self.release.notified().await;
            let stream: rig::streaming::StreamingResult<Self::StreamingResponse> =
                Box::pin(futures::stream::empty::<
                    Result<RawStreamingChoice<MockResponse>, CompletionError>,
                >());
            Ok(StreamingCompletionResponse::stream(stream))
        }
    }

    struct NotifyOnDrop(Arc<Notify>);

    impl Drop for NotifyOnDrop {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }

    fn test_runner(
        state: SharedMessages,
        events_tx: broadcast::Sender<AgentLoopEvent<MockResponse>>,
    ) -> Runner<MockCompletionModel, (), ()> {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let agent = Arc::new(AgentBuilder::new(model).build());
        Runner::new(RunnerInit {
            agent,
            app_state: Arc::new(()),
            max_turns: 100,
            turn_timeout: None,
            loop_timeout: None,
            turn_hook: None,
            assistant_message_hook: None,
            events_tx,
            state,
            turn_running: Arc::new(AtomicBool::new(false)),
            initial_turn: PendingTurn::Resume,
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
