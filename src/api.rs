use std::{
    error::Error,
    fmt,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use rig::{
    agent::{Agent, MultiTurnStreamItem, PromptHook, StreamingError},
    completion::{CompletionModel, GetTokenUsage, Usage},
    message::{Message, ToolCall, ToolResult},
    wasm_compat::WasmCompatSend,
};
use tokio::{
    sync::{broadcast, mpsc},
    task::{AbortHandle, JoinHandle},
};

use crate::{
    engine::{PendingTurn, Runner, RunnerInit},
    history::{repair_unanswered_tool_calls, validate_resume_history},
};

pub type TurnHookError = Box<dyn Error + Send + Sync + 'static>;
pub type AssistantMessageHookError = Box<dyn Error + Send + Sync + 'static>;
pub type ToolResultPersistenceError = Box<dyn Error + Send + Sync + 'static>;

pub(crate) type TurnHookFuture =
    Pin<Box<dyn Future<Output = Result<TurnHookAction, TurnHookError>> + Send>>;
pub(crate) type TurnHook<S> = Arc<dyn Fn(Arc<S>, TurnHookContext) -> TurnHookFuture + Send + Sync>;
pub(crate) type AssistantMessageHookFuture =
    Pin<Box<dyn Future<Output = Result<(), AssistantMessageHookError>> + Send>>;
pub(crate) type AssistantMessageHook<S> =
    Arc<dyn Fn(Arc<S>, AssistantMessageContext) -> AssistantMessageHookFuture + Send + Sync>;
pub type ToolResultPersistenceFuture<T> =
    Pin<Box<dyn Future<Output = Result<T, ToolResultPersistenceError>> + Send>>;
pub(crate) type SharedMessages = Arc<Mutex<Vec<Message>>>;
pub(crate) type SharedTurnRunning = Arc<AtomicBool>;
pub(crate) const EVENT_BUFFER_SIZE: usize = 1024;

/// Default multi-turn tool-call budget for an agent run.
///
/// This is intentionally high enough to stay out of the way for normal agent
/// loops, while remaining finite because Rig performs arithmetic on the limit.
pub const DEFAULT_MAX_TURNS: usize = 1_000_000;

/// Default synthetic tool-result text used to repair unanswered assistant tool calls.
pub const DEFAULT_UNANSWERED_TOOL_CALL_REPAIR_MESSAGE: &str = "Recovery message: no result was recorded for this tool call. It may or may not have completed.";

pub(crate) const DEFAULT_TOOL_REPAIR_MESSAGE: &str = DEFAULT_UNANSWERED_TOOL_CALL_REPAIR_MESSAGE;

pub(crate) fn lock_messages(messages: &SharedMessages) -> MutexGuard<'_, Vec<Message>> {
    match messages.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Stable lookup key for an individual tool result.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ToolResultKey {
    /// Rig/provider tool-call id.
    pub id: String,
    /// Provider-specific call id, when distinct from `id`.
    pub call_id: Option<String>,
}

impl ToolResultKey {
    pub fn new(id: impl Into<String>, call_id: Option<String>) -> Self {
        Self {
            id: id.into(),
            call_id,
        }
    }

    pub fn from_tool_call(tool_call: &ToolCall) -> Self {
        Self::new(tool_call.id.clone(), tool_call.call_id.clone())
    }

    pub fn from_tool_result(tool_result: &ToolResult) -> Self {
        Self::new(tool_result.id.clone(), tool_result.call_id.clone())
    }
}

impl From<&ToolCall> for ToolResultKey {
    fn from(tool_call: &ToolCall) -> Self {
        Self::from_tool_call(tool_call)
    }
}

impl From<&ToolResult> for ToolResultKey {
    fn from(tool_result: &ToolResult) -> Self {
        Self::from_tool_result(tool_result)
    }
}

/// Optional persistence hook for tool results as they complete.
///
/// When configured, each returned tool result is persisted before the result is
/// committed into the loop history. If startup history contains assistant tool
/// calls without corresponding user tool results, repair first asks this store
/// for each missing result before falling back to the configured recovery text.
pub trait IncrementalToolResultPersistence: Send + Sync + 'static {
    fn persist_tool_result(
        &self,
        key: ToolResultKey,
        result: ToolResult,
    ) -> ToolResultPersistenceFuture<()>;

    fn load_tool_result(
        &self,
        key: ToolResultKey,
    ) -> ToolResultPersistenceFuture<Option<ToolResult>>;
}

impl<T> IncrementalToolResultPersistence for Arc<T>
where
    T: IncrementalToolResultPersistence + ?Sized,
{
    fn persist_tool_result(
        &self,
        key: ToolResultKey,
        result: ToolResult,
    ) -> ToolResultPersistenceFuture<()> {
        (**self).persist_tool_result(key, result)
    }

    fn load_tool_result(
        &self,
        key: ToolResultKey,
    ) -> ToolResultPersistenceFuture<Option<ToolResult>> {
        (**self).load_tool_result(key)
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
    pub(crate) initial_history: Vec<Message>,
    turn_hook: Option<TurnHook<S>>,
    assistant_message_hook: Option<AssistantMessageHook<S>>,
    pub(crate) unanswered_tool_call_repair: Option<String>,
    tool_result_persistence: Option<Arc<dyn IncrementalToolResultPersistence>>,
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
            max_turns: DEFAULT_MAX_TURNS,
            turn_timeout: None,
            loop_timeout: None,
            initial_history: Vec::new(),
            turn_hook: None,
            assistant_message_hook: None,
            unanswered_tool_call_repair: Some(DEFAULT_TOOL_REPAIR_MESSAGE.to_string()),
            tool_result_persistence: None,
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
            tool_result_persistence: self.tool_result_persistence,
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

    /// Persist each completed tool result and use those results during startup repair.
    pub fn with_incremental_tool_result_persistence(
        mut self,
        persistence: impl IncrementalToolResultPersistence,
    ) -> Self {
        self.tool_result_persistence = Some(Arc::new(persistence));
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
        self.start(PendingTurn::single(prompt.into()), self.initial_history())
    }

    /// Start the loop from the seeded history without appending a new prompt.
    pub fn resume(&self) -> Result<AgentLoopHandle<M::StreamingResponse>, AgentLoopError> {
        let validation_history = self.repaired_initial_history();
        validate_resume_history(&validation_history)
            .map_err(AgentLoopError::InvalidMessageHistory)?;
        Ok(self.start(PendingTurn::Resume, self.initial_history()))
    }

    fn initial_history(&self) -> Vec<Message> {
        if self.tool_result_persistence.is_some() {
            self.initial_history.clone()
        } else {
            self.repaired_initial_history()
        }
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
            unanswered_tool_call_repair: self.unanswered_tool_call_repair.clone(),
            tool_result_persistence: self.tool_result_persistence.clone(),
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
    pub(crate) commands_tx: Option<mpsc::UnboundedSender<Command>>,
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

    pub(crate) fn finish_when_idle(&self) -> Result<(), AgentLoopError> {
        self.send(Command::FinishWhenIdle)
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
    /// A managed wait reached idle without starting or resuming an agent turn.
    NoRun,
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
    /// The expected terminal reason for this loop, if this boundary is expected
    /// to end the current run. `None` means more work is already queued or the
    /// loop cannot yet prove it is ending.
    pub end_reason: Option<EndReason>,
    /// Aggregated provider usage for the completed turn when available.
    ///
    /// Providers may omit usage; in that case this is `None`.
    pub usage: Option<Usage>,
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
    /// The expected terminal reason for this loop, if this assistant message is
    /// expected to be part of the final committed boundary for the current run.
    /// `None` means the turn is still active, more work is already queued, or
    /// the loop cannot yet prove it is ending.
    pub end_reason: Option<EndReason>,
    /// Aggregated provider usage for the completed turn when available.
    ///
    /// This is only set for assistant snapshots emitted from a final response.
    /// Earlier snapshots, such as assistant tool-call messages, do not have usage
    /// yet. Providers may also omit usage.
    pub usage: Option<Usage>,
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
    /// Replace the loop's active history, then stop the loop.
    ReplaceHistoryAndAbort {
        history: Vec<Message>,
        reason: String,
    },
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
    ToolResultPersistence(ToolResultPersistenceError),
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
            Self::ToolResultPersistence(err) => write!(f, "tool result persistence failed: {err}"),
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
            Self::ToolResultPersistence(err) => Some(err.as_ref()),
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
    pub(crate) fn from_error(err: &AgentLoopError) -> Self {
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
    pub(crate) fn new(message: impl Into<String>) -> Self {
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

pub(crate) enum Command {
    Steer(Message),
    FollowUp(Message),
    Resume,
    Interrupt(Message),
    Abort,
    FinishWhenIdle,
}
