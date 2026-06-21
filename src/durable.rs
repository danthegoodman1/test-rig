use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use rig::{
    OneOrMany,
    agent::{Agent, PromptHook},
    completion::{CompletionModel, GetTokenUsage, Usage},
    message::{AssistantContent, Message, Text, ToolResult, UserContent},
    wasm_compat::WasmCompatSend,
};
use serde_json::{Map, Value};
use tokio::{
    sync::{Mutex as AsyncMutex, broadcast, mpsc, oneshot},
    task::JoinHandle,
    time::Instant,
};

use super::{
    AgentLoop, AgentLoopError, AgentLoopEvent, AgentLoopHandle, AssistantMessageContext, Command,
    EndReason, IncrementalToolResultPersistence, ToolResultKey, ToolResultPersistenceFuture,
    TurnHookAction, TurnHookContext, TurnOutcomeKind,
    repair_unanswered_tool_calls_with_persistence, validate_message_history,
};

pub const INBOX_ENTRY_ID_PARAM: &str = "rigloop_inbox_entry_id";

pub type DurableAgentStoreError = Box<dyn Error + Send + Sync + 'static>;
pub type DurableAgentCallbackError = Box<dyn Error + Send + Sync + 'static>;
pub type DurableAgentFuture<T> =
    Pin<Box<dyn Future<Output = Result<T, DurableAgentStoreError>> + Send>>;

type ObserverFuture = Pin<Box<dyn Future<Output = Result<(), DurableAgentCallbackError>> + Send>>;
type EventObserver<R> = Arc<dyn Fn(AgentLoopEvent<R>) -> ObserverFuture + Send + Sync>;
type CheckpointFuture = Pin<
    Box<dyn Future<Output = Result<DurableCheckpointAction, DurableAgentCallbackError>> + Send>,
>;
type CheckpointHandler = Arc<dyn Fn(DurableCheckpoint) -> CheckpointFuture + Send + Sync>;
type InboxIdGenerator = Arc<dyn Fn(usize) -> String + Send + Sync>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DurableInboxKind {
    Steer,
    FollowUp,
    Interrupt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DurableInboxStatus {
    Submitted,
    Acked,
}

#[derive(Clone, Debug)]
pub struct DurableInboxEntry {
    pub id: String,
    pub kind: DurableInboxKind,
    pub message: Message,
    pub status: DurableInboxStatus,
    pub submitted_at: Instant,
}

#[derive(Clone, Debug)]
pub struct PersistMessagesArgs {
    /// Transcript messages that became durable in this checkpoint.
    pub messages: Vec<Message>,
    /// Inbox entries whose tagged user messages are included in `messages`.
    pub ack_inbox_entry_ids: Vec<String>,
    /// Turn metadata to persist atomically when this checkpoint is a turn boundary.
    pub turn_outcome: Option<DurableTurnOutcome>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DurableTurnOutcome {
    /// What happened to this turn at the commit boundary.
    pub kind: TurnOutcomeKind,
    /// Terminal loop reason when this turn boundary also ends the current run.
    ///
    /// `None` means the turn ended, but more work is queued or the loop cannot
    /// yet prove it is ending.
    pub end_reason: Option<EndReason>,
    /// Aggregated provider usage for the completed turn when available.
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum DurableCheckpoint {
    /// A streamed assistant message is available and has been persisted.
    ///
    /// If the assistant message contains tool calls, this snapshot is not yet a
    /// provider-valid replay history until matching tool results are appended or
    /// repaired. Use this boundary for crash recovery and early observation.
    AssistantMessageFinished(AssistantMessageContext),
    /// A turn append boundary has been persisted.
    ///
    /// This history has passed provider-order validation. Prefer this boundary
    /// for history maintenance work such as compaction.
    TurnBoundary(TurnHookContext),
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DurableCheckpointAction {
    Continue,
    Abort { reason: String },
}

pub trait DurableAgentStore: IncrementalToolResultPersistence {
    fn submit_inbox_entry(&self, entry: DurableInboxEntry) -> DurableAgentFuture<()>;

    fn persist_messages_and_ack(&self, args: PersistMessagesArgs) -> DurableAgentFuture<()>;
}

/// In-process durable store for examples, tests, and local prototypes.
///
/// This store implements the same append-and-ACK contract as a real
/// [`DurableAgentStore`], but it is not durable across process restarts.
#[derive(Clone, Debug, Default)]
pub struct InMemoryDurableAgentStore {
    inner: Arc<Mutex<InMemoryDurableAgentStoreState>>,
}

#[derive(Clone, Debug, Default)]
struct InMemoryDurableAgentStoreState {
    submitted_inbox_entries: Vec<DurableInboxEntry>,
    acked_inbox_entries: Vec<DurableInboxEntry>,
    acked_inbox_ids: Vec<String>,
    persisted_messages: Vec<Message>,
    transactions: Vec<PersistMessagesArgs>,
    turn_outcomes: Vec<DurableTurnOutcome>,
    tool_results: BTreeMap<ToolResultKey, ToolResult>,
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryDurableAgentStoreSnapshot {
    pub submitted_inbox_entries: Vec<DurableInboxEntry>,
    pub pending_inbox_entries: Vec<DurableInboxEntry>,
    pub acked_inbox_entries: Vec<DurableInboxEntry>,
    pub acked_inbox_ids: Vec<String>,
    pub persisted_messages: Vec<Message>,
    pub transactions: Vec<PersistMessagesArgs>,
    pub turn_outcomes: Vec<DurableTurnOutcome>,
    pub tool_results: BTreeMap<ToolResultKey, ToolResult>,
}

impl InMemoryDurableAgentStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> InMemoryDurableAgentStoreSnapshot {
        self.inner
            .lock()
            .expect("in-memory durable store poisoned")
            .snapshot()
    }

    pub fn load_history(&self) -> Vec<Message> {
        self.inner
            .lock()
            .expect("in-memory durable store poisoned")
            .persisted_messages
            .clone()
    }

    pub fn replace_history(&self, history: Vec<Message>) {
        self.inner
            .lock()
            .expect("in-memory durable store poisoned")
            .persisted_messages = history;
    }

    pub fn insert_tool_result(&self, key: ToolResultKey, result: ToolResult) -> Option<ToolResult> {
        self.inner
            .lock()
            .expect("in-memory durable store poisoned")
            .tool_results
            .insert(key, result)
    }
}

impl InMemoryDurableAgentStoreState {
    fn snapshot(&self) -> InMemoryDurableAgentStoreSnapshot {
        let acked_ids = self
            .acked_inbox_entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<BTreeSet<_>>();
        let pending_inbox_entries = self
            .submitted_inbox_entries
            .iter()
            .filter(|entry| !acked_ids.contains(&entry.id))
            .cloned()
            .collect();

        InMemoryDurableAgentStoreSnapshot {
            submitted_inbox_entries: self.submitted_inbox_entries.clone(),
            pending_inbox_entries,
            acked_inbox_entries: self.acked_inbox_entries.clone(),
            acked_inbox_ids: self.acked_inbox_ids.clone(),
            persisted_messages: self.persisted_messages.clone(),
            transactions: self.transactions.clone(),
            turn_outcomes: self.turn_outcomes.clone(),
            tool_results: self.tool_results.clone(),
        }
    }
}

impl IncrementalToolResultPersistence for InMemoryDurableAgentStore {
    fn persist_tool_result(
        &self,
        key: ToolResultKey,
        result: ToolResult,
    ) -> ToolResultPersistenceFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            store
                .inner
                .lock()
                .expect("in-memory durable store poisoned")
                .tool_results
                .insert(key, result);
            Ok(())
        })
    }

    fn load_tool_result(
        &self,
        key: ToolResultKey,
    ) -> ToolResultPersistenceFuture<Option<ToolResult>> {
        let store = self.clone();
        Box::pin(async move {
            Ok(store
                .inner
                .lock()
                .expect("in-memory durable store poisoned")
                .tool_results
                .get(&key)
                .cloned())
        })
    }
}

impl DurableAgentStore for InMemoryDurableAgentStore {
    fn submit_inbox_entry(&self, entry: DurableInboxEntry) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            store
                .inner
                .lock()
                .expect("in-memory durable store poisoned")
                .submitted_inbox_entries
                .push(entry);
            Ok(())
        })
    }

    fn persist_messages_and_ack(&self, args: PersistMessagesArgs) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            let mut state = store
                .inner
                .lock()
                .expect("in-memory durable store poisoned");
            state.transactions.push(args.clone());
            if let Some(turn_outcome) = &args.turn_outcome {
                state.turn_outcomes.push(turn_outcome.clone());
            }
            state
                .acked_inbox_ids
                .extend(args.ack_inbox_entry_ids.clone());

            let mut already_acked = state
                .acked_inbox_entries
                .iter()
                .map(|entry| entry.id.clone())
                .collect::<BTreeSet<_>>();
            for inbox_entry_id in &args.ack_inbox_entry_ids {
                if already_acked.contains(inbox_entry_id) {
                    continue;
                }

                if let Some(entry) = state
                    .submitted_inbox_entries
                    .iter()
                    .find(|entry| &entry.id == inbox_entry_id)
                {
                    let mut entry = entry.clone();
                    entry.status = DurableInboxStatus::Acked;
                    already_acked.insert(entry.id.clone());
                    state.acked_inbox_entries.push(entry);
                }
            }

            for message in &args.messages {
                if let Message::User { content } = message {
                    for item in content.iter() {
                        if let UserContent::ToolResult(tool_result) = item {
                            state.tool_results.insert(
                                ToolResultKey::from_tool_result(tool_result),
                                tool_result.clone(),
                            );
                        }
                    }
                }
            }

            state.persisted_messages.extend(args.messages);
            Ok(())
        })
    }
}

#[derive(Debug)]
pub enum DurableAgentError {
    InvalidSignalMessage(String),
    CommandChannelClosed,
    AlreadyStarted,
    NotStarted,
    Store(DurableAgentStoreError),
    Callback(DurableAgentCallbackError),
    AgentLoop(AgentLoopError),
    TaskJoin(tokio::task::JoinError),
    ManagerFailed(String),
}

impl fmt::Display for DurableAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSignalMessage(message) => {
                write!(f, "invalid signal message: {message}")
            }
            Self::CommandChannelClosed => f.write_str("durable agent command channel is closed"),
            Self::AlreadyStarted => f.write_str("durable agent is already started"),
            Self::NotStarted => f.write_str("durable agent has not been started"),
            Self::Store(err) => write!(f, "durable store failed: {err}"),
            Self::Callback(err) => write!(f, "durable callback failed: {err}"),
            Self::AgentLoop(err) => write!(f, "{err}"),
            Self::TaskJoin(err) => write!(f, "{err}"),
            Self::ManagerFailed(message) => write!(f, "durable manager failed: {message}"),
        }
    }
}

impl Error for DurableAgentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(err) => Some(err.as_ref()),
            Self::Callback(err) => Some(err.as_ref()),
            Self::AgentLoop(err) => Some(err),
            Self::TaskJoin(err) => Some(err),
            Self::InvalidSignalMessage(_)
            | Self::CommandChannelClosed
            | Self::AlreadyStarted
            | Self::NotStarted
            | Self::ManagerFailed(_) => None,
        }
    }
}

impl From<AgentLoopError> for DurableAgentError {
    fn from(err: AgentLoopError) -> Self {
        Self::AgentLoop(err)
    }
}

impl From<DurableAgentStoreError> for DurableAgentError {
    fn from(err: DurableAgentStoreError) -> Self {
        Self::Store(err)
    }
}

#[derive(Clone)]
struct QueuedSignal {
    kind: DurableInboxKind,
    message: Message,
}

enum ManagerCommand {
    Signal(Box<QueuedSignal>),
    Abort,
    Wait(oneshot::Sender<Result<EndReason, DurableAgentError>>),
}

#[derive(Clone)]
pub struct DurableAgentControl {
    commands_tx: mpsc::UnboundedSender<ManagerCommand>,
}

impl DurableAgentControl {
    pub fn abort(&self) -> Result<(), DurableAgentError> {
        self.commands_tx
            .send(ManagerCommand::Abort)
            .map_err(|_| DurableAgentError::CommandChannelClosed)
    }
}

struct DurableCheckpointer<D, R>
where
    D: DurableAgentStore,
    R: Clone,
{
    store: Arc<D>,
    persisted_history: AsyncMutex<Vec<Message>>,
    repair_message: Option<String>,
    commands_tx: mpsc::UnboundedSender<ManagerCommand>,
    event_observer: Option<EventObserver<R>>,
    checkpoint_handler: Option<CheckpointHandler>,
    abort_reason: Mutex<Option<String>>,
}

struct CheckpointResult {
    replacement_history: Option<Vec<Message>>,
}

impl<D, R> DurableCheckpointer<D, R>
where
    D: DurableAgentStore,
    R: Clone + Send + 'static,
{
    fn new(
        store: Arc<D>,
        persisted_history: Vec<Message>,
        repair_message: Option<String>,
        commands_tx: mpsc::UnboundedSender<ManagerCommand>,
        event_observer: Option<EventObserver<R>>,
        checkpoint_handler: Option<CheckpointHandler>,
    ) -> Self {
        Self {
            store,
            persisted_history: AsyncMutex::new(persisted_history),
            repair_message,
            commands_tx,
            event_observer,
            checkpoint_handler,
            abort_reason: Mutex::new(None),
        }
    }

    async fn checkpoint_history(
        &self,
        history: &[Message],
        turn_outcome: Option<DurableTurnOutcome>,
    ) -> Result<CheckpointResult, DurableAgentError> {
        let mut persisted_history = self.persisted_history.lock().await;

        if history.starts_with(&persisted_history) {
            let new_messages = history[persisted_history.len()..].to_vec();
            self.persist_delta(new_messages, turn_outcome).await?;
            *persisted_history = history.to_vec();
            return Ok(CheckpointResult {
                replacement_history: None,
            });
        }

        let common_prefix_len = common_prefix_len(&persisted_history, history);
        let repaired_history = self.repair_persisted_history(&persisted_history).await?;
        let mut replacement_history = repaired_history.clone();
        replacement_history.extend_from_slice(&history[common_prefix_len..]);

        validate_message_history(&replacement_history)
            .map_err(AgentLoopError::InvalidMessageHistory)?;

        let mut new_messages = repaired_history[persisted_history.len()..].to_vec();
        new_messages.extend_from_slice(&history[common_prefix_len..]);
        self.persist_delta(new_messages, turn_outcome).await?;
        *persisted_history = replacement_history.clone();

        Ok(CheckpointResult {
            replacement_history: Some(replacement_history),
        })
    }

    async fn persist_delta(
        &self,
        messages: Vec<Message>,
        turn_outcome: Option<DurableTurnOutcome>,
    ) -> Result<(), DurableAgentError> {
        if messages.is_empty() && turn_outcome.is_none() {
            return Ok(());
        }

        let ack_inbox_entry_ids = inbox_ids_from_messages(&messages);
        self.store
            .persist_messages_and_ack(PersistMessagesArgs {
                messages,
                ack_inbox_entry_ids,
                turn_outcome,
            })
            .await
            .map_err(DurableAgentError::Store)?;
        Ok(())
    }

    async fn repair_persisted_history(
        &self,
        history: &[Message],
    ) -> Result<Vec<Message>, DurableAgentError> {
        let Some(repair_message) = &self.repair_message else {
            return Ok(history.to_vec());
        };

        repair_unanswered_tool_calls_with_persistence(
            history.to_vec(),
            repair_message,
            Some(self.store.as_ref()),
        )
        .await
        .map_err(DurableAgentError::AgentLoop)
    }

    async fn observe_event(&self, event: AgentLoopEvent<R>) -> Result<(), DurableAgentError> {
        if let Some(observer) = &self.event_observer {
            observer(event).await.map_err(DurableAgentError::Callback)?;
        }
        Ok(())
    }

    async fn handle_checkpoint(
        &self,
        checkpoint: DurableCheckpoint,
    ) -> Result<DurableCheckpointAction, DurableAgentError> {
        let Some(handler) = &self.checkpoint_handler else {
            return Ok(DurableCheckpointAction::Continue);
        };

        handler(checkpoint)
            .await
            .map_err(DurableAgentError::Callback)
    }

    fn request_abort(&self, reason: String) -> Result<(), DurableAgentError> {
        *self
            .abort_reason
            .lock()
            .expect("checkpoint abort reason poisoned") = Some(reason);
        self.commands_tx
            .send(ManagerCommand::Abort)
            .map_err(|_| DurableAgentError::CommandChannelClosed)
    }

    fn take_abort_reason(&self) -> Option<String> {
        self.abort_reason
            .lock()
            .expect("checkpoint abort reason poisoned")
            .take()
    }

    async fn assistant_finished(
        &self,
        context: AssistantMessageContext,
    ) -> Result<(), DurableAgentError> {
        self.checkpoint_history(&context.history, None).await?;
        match self
            .handle_checkpoint(DurableCheckpoint::AssistantMessageFinished(context.clone()))
            .await?
        {
            DurableCheckpointAction::Continue => {}
            DurableCheckpointAction::Abort { reason } => self.request_abort(reason)?,
        }
        self.observe_event(AgentLoopEvent::AssistantMessageFinished {
            message: context.message,
            messages: context.new_messages,
        })
        .await
    }

    async fn turn_committed(
        &self,
        turn: TurnHookContext,
    ) -> Result<TurnHookAction, DurableAgentError> {
        let turn_outcome = DurableTurnOutcome {
            kind: turn.outcome_kind.clone(),
            end_reason: turn.end_reason.clone(),
            usage: turn.usage,
        };
        let checkpoint = self
            .checkpoint_history(&turn.history, Some(turn_outcome))
            .await?;
        let action = self
            .handle_checkpoint(DurableCheckpoint::TurnBoundary(turn.clone()))
            .await?;
        let abort_reason = match action {
            DurableCheckpointAction::Continue => self.take_abort_reason(),
            DurableCheckpointAction::Abort { reason } => Some(reason),
        };

        match (checkpoint.replacement_history, abort_reason) {
            (Some(history), Some(reason)) => {
                Ok(TurnHookAction::ReplaceHistoryAndAbort { history, reason })
            }
            (Some(history), None) => Ok(TurnHookAction::ReplaceHistory(history)),
            (None, Some(reason)) => Ok(TurnHookAction::Abort { reason }),
            (None, None) => Ok(TurnHookAction::Continue),
        }
    }
}

pub struct DurableAgentHarness<M, P, D>
where
    M: CompletionModel,
    P: PromptHook<M>,
    D: DurableAgentStore,
{
    loop_config: Arc<Mutex<Option<AgentLoop<M, P>>>>,
    store: Arc<D>,
    commands_tx: mpsc::UnboundedSender<ManagerCommand>,
    commands_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<ManagerCommand>>>>,
    manager: Arc<Mutex<Option<JoinHandle<()>>>>,
    next_inbox_id: AtomicUsize,
    inbox_id_generator: Arc<Mutex<InboxIdGenerator>>,
    event_observer: Arc<Mutex<Option<EventObserver<M::StreamingResponse>>>>,
    checkpoint_handler: Arc<Mutex<Option<CheckpointHandler>>>,
}

impl<M, P, D> DurableAgentHarness<M, P, D>
where
    M: CompletionModel,
    P: PromptHook<M>,
    D: DurableAgentStore,
{
    pub fn new(agent: Agent<M, P>, store: D) -> Self {
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        Self {
            loop_config: Arc::new(Mutex::new(Some(AgentLoop::new(agent)))),
            store: Arc::new(store),
            commands_tx,
            commands_rx: Arc::new(Mutex::new(Some(commands_rx))),
            manager: Arc::default(),
            next_inbox_id: AtomicUsize::new(0),
            inbox_id_generator: Arc::new(Mutex::new(Arc::new(|index| format!("inbox-{index}")))),
            event_observer: Arc::default(),
            checkpoint_handler: Arc::default(),
        }
    }

    pub fn with_history<H, T>(self, history: H) -> Self
    where
        H: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.update_loop_config(|loop_| loop_.with_history(history));
        self
    }

    pub fn max_turns(self, max_turns: usize) -> Self {
        self.update_loop_config(|loop_| loop_.max_turns(max_turns));
        self
    }

    pub fn turn_timeout(self, timeout: Duration) -> Self {
        self.update_loop_config(|loop_| loop_.turn_timeout(timeout));
        self
    }

    pub fn loop_timeout(self, timeout: Duration) -> Self {
        self.update_loop_config(|loop_| loop_.loop_timeout(timeout));
        self
    }

    pub fn with_unanswered_tool_call_repair_message(self, message: impl Into<String>) -> Self {
        self.update_loop_config(|loop_| loop_.with_unanswered_tool_call_repair_message(message));
        self
    }

    pub fn with_inbox_id_generator<F>(self, generator: F) -> Self
    where
        F: Fn(usize) -> String + Send + Sync + 'static,
    {
        *self
            .inbox_id_generator
            .lock()
            .expect("inbox id generator poisoned") = Arc::new(generator);
        self
    }

    pub fn control(&self) -> DurableAgentControl {
        DurableAgentControl {
            commands_tx: self.commands_tx.clone(),
        }
    }

    pub fn abort(&self) -> Result<(), DurableAgentError> {
        self.control().abort()
    }

    /// Observe raw loop events after managed durability work has handled them.
    ///
    /// This is for logging, metrics, UI streaming, or forwarding lifecycle events.
    /// Use [`Self::with_checkpoint_handler`] when the callback needs to make a
    /// durable history decision such as requesting compaction.
    pub fn with_event_observer<F, Fut, E>(self, observer: F) -> Self
    where
        F: Fn(AgentLoopEvent<M::StreamingResponse>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        *self.event_observer.lock().expect("event observer poisoned") =
            Some(Arc::new(move |event| {
                let future = observer(event);
                Box::pin(async move {
                    future
                        .await
                        .map_err(|err| Box::new(err) as DurableAgentCallbackError)
                })
            }));
        self
    }

    /// Handle durable checkpoint boundaries after internal persistence and ACKs.
    ///
    /// The handler receives assistant-message snapshots and turn boundaries only
    /// after the harness has persisted the relevant transcript delta. Assistant
    /// snapshots are for early persistence/recovery and may be temporarily
    /// provider-invalid when tool calls are waiting for results. Turn boundaries
    /// are provider-valid and are the right default for compaction.
    ///
    /// Each checkpoint context includes `end_reason`; `Some(reason)` means the
    /// current run is expected to end at this boundary, while `None` means the
    /// turn is still active or another turn is already queued. That distinction
    /// lets compaction code avoid rewriting history when no next turn is
    /// planned.
    ///
    /// Return [`DurableCheckpointAction::Continue`] to keep running, or
    /// [`DurableCheckpointAction::Abort`] to stop the active run at the next safe
    /// commit boundary. If aborting after a partial assistant snapshot requires
    /// unanswered-tool-call repair, the harness repairs/replaces in-memory
    /// history before the run exits. Handler errors fail the managed run and are
    /// returned by [`Self::wait_for_idle`].
    pub fn with_checkpoint_handler<F, Fut, E>(self, handler: F) -> Self
    where
        F: Fn(DurableCheckpoint) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<DurableCheckpointAction, E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        *self
            .checkpoint_handler
            .lock()
            .expect("checkpoint handler poisoned") = Some(Arc::new(move |checkpoint| {
            let future = handler(checkpoint);
            Box::pin(async move {
                future
                    .await
                    .map_err(|err| Box::new(err) as DurableAgentCallbackError)
            })
        }));
        self
    }

    fn update_loop_config<F>(&self, update: F)
    where
        F: FnOnce(AgentLoop<M, P>) -> AgentLoop<M, P>,
    {
        let mut config = self.loop_config.lock().expect("loop config poisoned");
        let loop_ = config
            .take()
            .expect("durable agent configuration cannot be changed after start");
        *config = Some(update(loop_));
    }
}

impl<M, P, D> DurableAgentHarness<M, P, D>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
    D: DurableAgentStore,
{
    pub fn start(&self) -> Result<(), DurableAgentError> {
        let mut manager = self.manager.lock().expect("manager lock poisoned");
        if manager.is_some() {
            return Ok(());
        }

        let mut loop_config = self.loop_config.lock().expect("loop config poisoned");
        let loop_ = loop_config
            .take()
            .ok_or(DurableAgentError::AlreadyStarted)?;
        drop(loop_config);

        let mut commands_rx_guard = self.commands_rx.lock().expect("commands rx poisoned");
        let commands_rx = commands_rx_guard
            .take()
            .ok_or(DurableAgentError::AlreadyStarted)?;
        drop(commands_rx_guard);

        let event_observer = self
            .event_observer
            .lock()
            .expect("event observer poisoned")
            .clone();
        let checkpoint_handler = self
            .checkpoint_handler
            .lock()
            .expect("checkpoint handler poisoned")
            .clone();
        let store = self.store.clone();
        let initial_history = loop_.initial_history.clone();
        let repair_message = loop_.unanswered_tool_call_repair.clone();
        let checkpoint = Arc::new(DurableCheckpointer::new(
            store.clone(),
            initial_history,
            repair_message,
            self.commands_tx.clone(),
            event_observer.clone(),
            checkpoint_handler,
        ));

        let assistant_checkpoint = checkpoint.clone();
        let turn_checkpoint = checkpoint.clone();
        let loop_ = loop_
            .with_incremental_tool_result_persistence(store)
            .on_assistant_message_finished(move |_state, context| {
                let checkpoint = assistant_checkpoint.clone();
                async move { checkpoint.assistant_finished(context).await }
            })
            .with_turn_hook(move |_state, turn| {
                let checkpoint = turn_checkpoint.clone();
                async move { checkpoint.turn_committed(turn).await }
            });

        let mut durable_manager = DurableManager {
            loop_,
            commands_rx,
            pending: VecDeque::new(),
            waiters: Vec::new(),
            checkpoint,
            last_end_reason: EndReason::NoRun,
            failed_message: None,
            did_continue_initial_state: false,
        };
        *manager = Some(tokio::spawn(async move {
            durable_manager.run().await;
        }));
        Ok(())
    }

    pub async fn wait_for_idle(&self) -> Result<EndReason, DurableAgentError> {
        self.start()?;
        let (tx, rx) = oneshot::channel();
        self.commands_tx
            .send(ManagerCommand::Wait(tx))
            .map_err(|_| DurableAgentError::CommandChannelClosed)?;
        rx.await
            .map_err(|_| DurableAgentError::CommandChannelClosed)?
    }

    pub async fn steer(
        &self,
        message: impl Into<Message>,
    ) -> Result<DurableInboxEntry, DurableAgentError> {
        self.submit_signal(DurableInboxKind::Steer, message.into())
            .await
    }

    pub async fn follow_up(
        &self,
        message: impl Into<Message>,
    ) -> Result<DurableInboxEntry, DurableAgentError> {
        self.submit_signal(DurableInboxKind::FollowUp, message.into())
            .await
    }

    pub async fn interrupt(
        &self,
        message: impl Into<Message>,
    ) -> Result<DurableInboxEntry, DurableAgentError> {
        self.submit_signal(DurableInboxKind::Interrupt, message.into())
            .await
    }

    async fn submit_signal(
        &self,
        kind: DurableInboxKind,
        message: Message,
    ) -> Result<DurableInboxEntry, DurableAgentError> {
        let index = self.next_inbox_id.fetch_add(1, Ordering::SeqCst) + 1;
        let id = {
            let generator = self
                .inbox_id_generator
                .lock()
                .expect("inbox id generator poisoned")
                .clone();
            generator(index)
        };
        let message = tag_message_with_inbox_id(message, &id)?;
        let entry = DurableInboxEntry {
            id,
            kind: kind.clone(),
            message: message.clone(),
            status: DurableInboxStatus::Submitted,
            submitted_at: Instant::now(),
        };
        self.store
            .submit_inbox_entry(entry.clone())
            .await
            .map_err(DurableAgentError::Store)?;
        self.commands_tx
            .send(ManagerCommand::Signal(Box::new(QueuedSignal {
                kind,
                message,
            })))
            .map_err(|_| DurableAgentError::CommandChannelClosed)?;
        Ok(entry)
    }
}

impl<M, P, D> Drop for DurableAgentHarness<M, P, D>
where
    M: CompletionModel,
    P: PromptHook<M>,
    D: DurableAgentStore,
{
    fn drop(&mut self) {
        if let Some(manager) = self.manager.lock().expect("manager lock poisoned").take() {
            manager.abort();
        }
    }
}

struct DurableManager<M, P, D>
where
    M: CompletionModel,
    P: PromptHook<M>,
    D: DurableAgentStore,
{
    loop_: AgentLoop<M, P>,
    commands_rx: mpsc::UnboundedReceiver<ManagerCommand>,
    pending: VecDeque<QueuedSignal>,
    waiters: Vec<oneshot::Sender<Result<EndReason, DurableAgentError>>>,
    checkpoint: Arc<DurableCheckpointer<D, M::StreamingResponse>>,
    last_end_reason: EndReason,
    failed_message: Option<String>,
    did_continue_initial_state: bool,
}

impl<M, P, D> DurableManager<M, P, D>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
    D: DurableAgentStore,
{
    async fn run(&mut self) {
        loop {
            if let Some(message) = self.failed_message.clone() {
                if !self.waiters.is_empty() {
                    self.finish_waiters(Err(DurableAgentError::ManagerFailed(message)));
                    continue;
                }

                let Some(command) = self.commands_rx.recv().await else {
                    return;
                };
                self.handle_idle_command(command);
                continue;
            }

            if let Some(signal) = self.pending.pop_front() {
                match self.run_prompt(signal).await {
                    Ok(end_reason) => {
                        self.last_end_reason = end_reason;
                        continue;
                    }
                    Err(err) => {
                        self.fail(err);
                        continue;
                    }
                }
            }

            if self.should_resume_initial_state() {
                self.did_continue_initial_state = true;
                match self.run_resume().await {
                    Ok(end_reason) => {
                        self.last_end_reason = end_reason;
                        continue;
                    }
                    Err(err) => {
                        self.fail(err);
                        continue;
                    }
                }
            }

            if !self.waiters.is_empty() {
                self.finish_idle_waiters();
            }

            let Some(command) = self.commands_rx.recv().await else {
                return;
            };
            self.handle_idle_command(command);
        }
    }

    fn handle_idle_command(&mut self, command: ManagerCommand) {
        match command {
            ManagerCommand::Signal(signal) => self.pending.push_back(*signal),
            ManagerCommand::Abort => {
                self.pending.clear();
                self.last_end_reason = EndReason::Aborted;
            }
            ManagerCommand::Wait(waiter) => self.waiters.push(waiter),
        }
    }

    fn should_resume_initial_state(&self) -> bool {
        if self.did_continue_initial_state {
            return false;
        }

        match self.loop_.initial_history.last() {
            Some(Message::User { .. }) => true,
            Some(Message::Assistant { content, .. }) => content
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            Some(Message::System { .. }) | None => false,
        }
    }

    async fn run_prompt(&mut self, signal: QueuedSignal) -> Result<EndReason, DurableAgentError> {
        let handle = self.loop_.prompt(signal.message);
        self.run_active(handle).await
    }

    async fn run_resume(&mut self) -> Result<EndReason, DurableAgentError> {
        let handle = self.loop_.resume()?;
        self.run_active(handle).await
    }

    async fn run_active(
        &mut self,
        handle: AgentLoopHandle<M::StreamingResponse>,
    ) -> Result<EndReason, DurableAgentError> {
        let signal_tx = handle
            .commands_tx
            .as_ref()
            .ok_or(DurableAgentError::CommandChannelClosed)?
            .clone();
        handle.finish_when_idle()?;
        while let Some(signal) = self.pending.pop_front() {
            send_signal_to_active(&signal_tx, signal)?;
        }

        let mut events = handle.subscribe();
        let wait = handle.wait();
        tokio::pin!(wait);

        loop {
            tokio::select! {
                command = self.commands_rx.recv() => {
                    let Some(command) = command else {
                        return Err(DurableAgentError::CommandChannelClosed);
                    };
                    match command {
                        ManagerCommand::Signal(signal) => {
                            send_signal_to_active(&signal_tx, *signal)?;
                        }
                        ManagerCommand::Abort => {
                            signal_tx
                                .send(Command::Abort)
                                .map_err(|_| DurableAgentError::CommandChannelClosed)?;
                        }
                        ManagerCommand::Wait(waiter) => {
                            self.waiters.push(waiter);
                        }
                    }
                }
                event = events.recv() => {
                    match event {
                        Ok(AgentLoopEvent::AssistantMessageFinished { .. }) => {}
                        Ok(event) => self.checkpoint.observe_event(event).await?,
                        Err(broadcast::error::RecvError::Lagged(_)) => {}
                        Err(broadcast::error::RecvError::Closed) => {}
                    }
                }
                result = &mut wait => {
                    let result = result?;
                    self.loop_.initial_history = result.history;
                    return Ok(result.end_reason);
                }
            }
        }
    }

    fn finish_waiters(&mut self, result: Result<EndReason, DurableAgentError>) {
        let waiters = std::mem::take(&mut self.waiters);
        match result {
            Ok(end_reason) => {
                for waiter in waiters {
                    let _ = waiter.send(Ok(end_reason.clone()));
                }
            }
            Err(err) => {
                let mut first = Some(err);
                let message = first.as_ref().map(ToString::to_string).unwrap_or_default();
                for waiter in waiters {
                    let result = if let Some(err) = first.take() {
                        Err(err)
                    } else {
                        Err(DurableAgentError::ManagerFailed(message.clone()))
                    };
                    let _ = waiter.send(result);
                }
            }
        }
    }

    fn finish_idle_waiters(&mut self) {
        let end_reason = std::mem::replace(&mut self.last_end_reason, EndReason::NoRun);
        self.finish_waiters(Ok(end_reason));
    }

    fn fail(&mut self, err: DurableAgentError) {
        self.failed_message = Some(err.to_string());
        self.finish_waiters_with_error(err);
    }

    fn finish_waiters_with_error(&mut self, err: DurableAgentError) {
        let message = err.to_string();
        let waiters = std::mem::take(&mut self.waiters);
        let mut first = Some(err);
        for waiter in waiters {
            let result = if let Some(err) = first.take() {
                Err(err)
            } else {
                Err(DurableAgentError::ManagerFailed(message.clone()))
            };
            let _ = waiter.send(result);
        }
    }
}

fn send_signal_to_active(
    signal_tx: &mpsc::UnboundedSender<Command>,
    signal: QueuedSignal,
) -> Result<(), DurableAgentError> {
    let command = match signal.kind {
        DurableInboxKind::Steer => Command::Steer(signal.message),
        DurableInboxKind::FollowUp => Command::FollowUp(signal.message),
        DurableInboxKind::Interrupt => Command::Interrupt(signal.message),
    };
    signal_tx
        .send(command)
        .map_err(|_| DurableAgentError::CommandChannelClosed)
}

pub fn tag_message_with_inbox_id(
    message: Message,
    inbox_id: &str,
) -> Result<Message, DurableAgentError> {
    let Message::User { content } = message else {
        return Err(DurableAgentError::InvalidSignalMessage(
            "durable signals must be user messages".to_string(),
        ));
    };

    let mut tagged = false;
    let mut items = content.iter().cloned().collect::<Vec<_>>();
    for item in &mut items {
        let UserContent::Text(text) = item else {
            continue;
        };

        tag_text_content(text, inbox_id)?;
        tagged = true;
        break;
    }

    if !tagged {
        return Err(DurableAgentError::InvalidSignalMessage(
            "durable signal user messages must contain text content".to_string(),
        ));
    }

    Ok(Message::User {
        content: OneOrMany::many(items).expect("durable signal user content is non-empty"),
    })
}

fn tag_text_content(text: &mut Text, inbox_id: &str) -> Result<(), DurableAgentError> {
    let mut params = match text.additional_params.take() {
        Some(Value::Object(params)) => params,
        None => Map::new(),
        Some(_) => {
            return Err(DurableAgentError::InvalidSignalMessage(
                "text additional_params must be a JSON object to carry an inbox id".to_string(),
            ));
        }
    };
    params.insert(
        INBOX_ENTRY_ID_PARAM.to_string(),
        Value::String(inbox_id.to_string()),
    );
    text.additional_params = Some(Value::Object(params));
    Ok(())
}

pub fn inbox_ids_from_messages(messages: &[Message]) -> Vec<String> {
    messages.iter().filter_map(inbox_id_from_message).collect()
}

pub fn inbox_id_from_message(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };
    content.iter().find_map(|item| {
        let UserContent::Text(text) = item else {
            return None;
        };
        text.additional_params
            .as_ref()?
            .get(INBOX_ENTRY_ID_PARAM)?
            .as_str()
            .map(ToOwned::to_owned)
    })
}

fn common_prefix_len(left: &[Message], right: &[Message]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}
