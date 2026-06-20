use std::{
    collections::VecDeque,
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
    completion::{CompletionModel, GetTokenUsage},
    message::{AssistantContent, Message, Text, UserContent},
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
    EndReason, IncrementalToolResultPersistence, TurnHookAction, TurnHookContext,
    repair_unanswered_tool_calls_with_persistence, validate_message_history,
};

pub const INBOX_ENTRY_ID_PARAM: &str = "rigloop_inbox_entry_id";

pub type DurableAgentStoreError = Box<dyn Error + Send + Sync + 'static>;
pub type DurableAgentObserverError = Box<dyn Error + Send + Sync + 'static>;
pub type DurableAgentFuture<T> =
    Pin<Box<dyn Future<Output = Result<T, DurableAgentStoreError>> + Send>>;

type ObserverFuture = Pin<Box<dyn Future<Output = Result<(), DurableAgentObserverError>> + Send>>;
type EventObserver<R> = Arc<dyn Fn(AgentLoopEvent<R>) -> ObserverFuture + Send + Sync>;
type AssistantObserver = Arc<dyn Fn(AssistantMessageContext) -> ObserverFuture + Send + Sync>;
type TurnObserver = Arc<dyn Fn(TurnHookContext) -> ObserverFuture + Send + Sync>;
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
    pub messages: Vec<Message>,
    pub ack_inbox_entry_ids: Vec<String>,
}

pub trait DurableAgentStore: IncrementalToolResultPersistence {
    fn submit_inbox_entry(&self, entry: DurableInboxEntry) -> DurableAgentFuture<()>;

    fn persist_messages_and_ack(&self, args: PersistMessagesArgs) -> DurableAgentFuture<()>;
}

#[derive(Debug)]
pub enum DurableAgentError {
    InvalidSignalMessage(String),
    CommandChannelClosed,
    AlreadyStarted,
    NotStarted,
    Store(DurableAgentStoreError),
    Observer(DurableAgentObserverError),
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
            Self::Observer(err) => write!(f, "durable observer failed: {err}"),
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
            Self::Observer(err) => Some(err.as_ref()),
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
    Signal(QueuedSignal),
    Wait(oneshot::Sender<Result<EndReason, DurableAgentError>>),
}

struct DurableCheckpoint<D, R>
where
    D: DurableAgentStore,
    R: Clone,
{
    store: Arc<D>,
    persisted_history: AsyncMutex<Vec<Message>>,
    repair_message: Option<String>,
    event_observer: Option<EventObserver<R>>,
    assistant_observer: Option<AssistantObserver>,
    turn_observer: Option<TurnObserver>,
}

struct CheckpointResult {
    replacement_history: Option<Vec<Message>>,
}

impl<D, R> DurableCheckpoint<D, R>
where
    D: DurableAgentStore,
    R: Clone + Send + 'static,
{
    fn new(
        store: Arc<D>,
        persisted_history: Vec<Message>,
        repair_message: Option<String>,
        event_observer: Option<EventObserver<R>>,
        assistant_observer: Option<AssistantObserver>,
        turn_observer: Option<TurnObserver>,
    ) -> Self {
        Self {
            store,
            persisted_history: AsyncMutex::new(persisted_history),
            repair_message,
            event_observer,
            assistant_observer,
            turn_observer,
        }
    }

    async fn checkpoint_history(
        &self,
        history: &[Message],
    ) -> Result<CheckpointResult, DurableAgentError> {
        let mut persisted_history = self.persisted_history.lock().await;

        if history.starts_with(&persisted_history) {
            let new_messages = history[persisted_history.len()..].to_vec();
            self.persist_delta(new_messages).await?;
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
        self.persist_delta(new_messages).await?;
        *persisted_history = replacement_history.clone();

        Ok(CheckpointResult {
            replacement_history: Some(replacement_history),
        })
    }

    async fn persist_delta(&self, messages: Vec<Message>) -> Result<(), DurableAgentError> {
        if messages.is_empty() {
            return Ok(());
        }

        let ack_inbox_entry_ids = inbox_ids_from_messages(&messages);
        self.store
            .persist_messages_and_ack(PersistMessagesArgs {
                messages,
                ack_inbox_entry_ids,
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
            observer(event).await.map_err(DurableAgentError::Observer)?;
        }
        Ok(())
    }

    async fn assistant_finished(
        &self,
        context: AssistantMessageContext,
    ) -> Result<(), DurableAgentError> {
        self.checkpoint_history(&context.history).await?;
        if let Some(observer) = &self.assistant_observer {
            observer(context.clone())
                .await
                .map_err(DurableAgentError::Observer)?;
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
        let checkpoint = self.checkpoint_history(&turn.history).await?;
        if let Some(observer) = &self.turn_observer {
            observer(turn.clone())
                .await
                .map_err(DurableAgentError::Observer)?;
        }
        if let Some(history) = checkpoint.replacement_history {
            return Ok(TurnHookAction::ReplaceHistory(history));
        }
        Ok(TurnHookAction::Continue)
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
    assistant_observer: Arc<Mutex<Option<AssistantObserver>>>,
    turn_observer: Arc<Mutex<Option<TurnObserver>>>,
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
            assistant_observer: Arc::default(),
            turn_observer: Arc::default(),
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

    pub fn on_event_observer<F, Fut, E>(self, observer: F) -> Self
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
                        .map_err(|err| Box::new(err) as DurableAgentObserverError)
                })
            }));
        self
    }

    pub fn on_assistant_message_finished_observer<F, Fut, E>(self, observer: F) -> Self
    where
        F: Fn(AssistantMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        *self
            .assistant_observer
            .lock()
            .expect("assistant observer poisoned") = Some(Arc::new(move |context| {
            let future = observer(context);
            Box::pin(async move {
                future
                    .await
                    .map_err(|err| Box::new(err) as DurableAgentObserverError)
            })
        }));
        self
    }

    pub fn on_turn_committed_observer<F, Fut, E>(self, observer: F) -> Self
    where
        F: Fn(TurnHookContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        *self.turn_observer.lock().expect("turn observer poisoned") = Some(Arc::new(move |turn| {
            let future = observer(turn);
            Box::pin(async move {
                future
                    .await
                    .map_err(|err| Box::new(err) as DurableAgentObserverError)
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
        let assistant_observer = self
            .assistant_observer
            .lock()
            .expect("assistant observer poisoned")
            .clone();
        let turn_observer = self
            .turn_observer
            .lock()
            .expect("turn observer poisoned")
            .clone();
        let store = self.store.clone();
        let initial_history = loop_.initial_history.clone();
        let repair_message = loop_.unanswered_tool_call_repair.clone();
        let checkpoint = Arc::new(DurableCheckpoint::new(
            store.clone(),
            initial_history,
            repair_message,
            event_observer.clone(),
            assistant_observer,
            turn_observer,
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
            .send(ManagerCommand::Signal(QueuedSignal { kind, message }))
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
    checkpoint: Arc<DurableCheckpoint<D, M::StreamingResponse>>,
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
            ManagerCommand::Signal(signal) => self.pending.push_back(signal),
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
                            send_signal_to_active(&signal_tx, signal)?;
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
