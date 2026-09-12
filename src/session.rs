use crate::{
    Commit, InboxEntry, InboxStatus, SignalKind, Store, ToolKey,
    journal::{calls, check_result, validate_append, validate_journal, validate_signal},
};
use futures::{FutureExt, StreamExt};
use rig::{
    agent::{
        Agent, AgentHook, MultiTurnStreamItem, StreamingError,
        hook::{
            CompletionCall, CompletionCallAction, HookContext, ModelTurnAction, ModelTurnFinished,
            ToolCall, ToolCallAction, ToolResultAction, ToolResultEvent,
        },
    },
    completion::{FinishReason, PromptError, Usage},
    message::{Message, ToolResult, ToolResultContent, UserContent},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    panic::AssertUnwindSafe,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{
    sync::{broadcast, mpsc, oneshot, watch},
    task::AbortHandle,
    time::{Instant, timeout_at},
};
use uuid::Uuid;

const RECOVERY: &str = "Recovery: no accepted result was recorded. This tool may or may not have completed. Do not assume it is safe to repeat its side effects.";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndReason {
    Completed,
    Idle,
    Paused,
    Aborted,
    Interrupted,
    TurnTimedOut,
    LoopTimedOut,
    MaxTurns,
    Length,
    ContentFilter,
    HookStopped(String),
    ProviderError(String),
    RunError(String),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub run_id: Uuid,
    pub reason: EndReason,
    /// None if any completed call lacked usage, or a provider call failed.
    pub usage: Option<Usage>,
    pub finished_at: SystemTime,
}
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid history: {0}")]
    InvalidHistory(String),
    #[error("session is closed")]
    Closed,
    #[error("session queue is full")]
    QueueFull,
    #[error("invalid configuration: {0}")]
    Configuration(String),
    #[error("{operation}: {message}")]
    Store {
        operation: &'static str,
        message: String,
    },
    #[error("commit {id} is uncertain: {message}", id = .commit.id)]
    CommitUncertain {
        commit: Arc<Commit>,
        message: String,
    },
    #[error("tool receipt write is uncertain: {message}")]
    ToolWriteUncertain {
        key: ToolKey,
        result: Arc<ToolResult>,
        message: String,
    },
    #[error("session task panicked")]
    TaskPanicked,
}

/// Observation only: use `Receiver::recv`'s explicit `Lagged` error to detect
/// dropped events. Slow observers never block execution or durable commits.
#[derive(Clone, Debug)]
pub enum Event {
    Rig(Arc<MultiTurnStreamItem>),
    RigError(Arc<StreamingError>),
    Committed(Arc<Commit>),
    Idle,
    Stopped(Result<EndReason, Error>),
}

/// Configuration is immutable once started. Rig's agent retains its tools,
/// model configuration, cache settings and hooks. The durable hook runs last.
pub struct Session {
    agent: Agent,
    store: Arc<dyn Store>,
    history: Vec<Message>,
    pending: Vec<InboxEntry>,
    max_turns: usize,
    turn_timeout: Option<Duration>,
    loop_timeout: Option<Duration>,
    io_timeout: Duration,
    capacity: usize,
}
impl Session {
    pub fn new(agent: Agent, store: impl Store) -> Self {
        Self {
            agent,
            store: Arc::new(store),
            history: Vec::new(),
            pending: Vec::new(),
            max_turns: 1_000_000,
            turn_timeout: None,
            loop_timeout: None,
            io_timeout: Duration::from_secs(30),
            capacity: 256,
        }
    }
    /// Caller-loaded canonical journal. Must match the store's current position.
    /// Only a trailing unanswered assistant batch is repaired automatically.
    pub fn history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }
    /// Caller-loaded pending entries, preserving their identity, kind and order.
    /// These must already exist in the store; they are not submitted again.
    pub fn pending(mut self, pending: Vec<InboxEntry>) -> Self {
        self.pending = pending;
        self
    }
    /// Model-call budget per turn (default 1,000,000), overriding the agent default.
    pub fn max_turns(mut self, max: usize) -> Self {
        self.max_turns = max;
        self
    }
    pub fn turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = Some(timeout);
        self
    }
    /// Whole session lifetime, including idle time and startup recovery.
    pub fn loop_timeout(mut self, timeout: Duration) -> Self {
        self.loop_timeout = Some(timeout);
        self
    }
    /// Maximum storage wait and total graceful-finalization allowance. Writes
    /// interrupted by this bound are reported as uncertain, never successful.
    pub fn io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }
    /// Bound both command intake and pending work; overload is explicit.
    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
    pub fn start(self) -> Result<SessionHandle, Error> {
        if self.capacity == 0
            || self.max_turns == 0
            || self.max_turns == usize::MAX
            || self.io_timeout.is_zero()
        {
            return Err(Error::Configuration(
                "capacity, max_turns and io_timeout must be positive; max_turns must be finite"
                    .into(),
            ));
        }
        for duration in self
            .turn_timeout
            .into_iter()
            .chain(self.loop_timeout)
            .chain([self.io_timeout])
        {
            if Instant::now().checked_add(duration).is_none() {
                return Err(Error::Configuration(
                    "timeout exceeds the clock range".into(),
                ));
            }
        }
        validate_journal(&self.history)?;
        if self.pending.len() > self.capacity {
            return Err(Error::QueueFull);
        }
        let mut ids = HashSet::new();
        for entry in &self.pending {
            validate_signal(&entry.message)?;
            if entry.id.is_empty() || !ids.insert(entry.id.clone()) {
                return Err(Error::Configuration(
                    "pending inbox IDs must be nonempty and unique".into(),
                ));
            }
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|e| Error::Configuration(e.to_string()))?;
        let (tx, rx) = mpsc::channel(self.capacity);
        let (events, _) = broadcast::channel(1024);
        let (status_tx, status) = watch::channel(None);
        let mut actor = Actor::new(self, rx, events.clone());
        let task = runtime.spawn(async move {
            let result = AssertUnwindSafe(actor.run())
                .catch_unwind()
                .await
                .unwrap_or(Err(Error::TaskPanicked));
            actor.emit(Event::Stopped(result.clone()));
            let _ = status_tx.send(Some(result));
        });
        Ok(SessionHandle {
            inner: Arc::new(HandleInner {
                tx,
                events,
                status,
                abort: task.abort_handle(),
            }),
        })
    }
}

#[derive(Clone)]
pub struct SessionHandle {
    inner: Arc<HandleInner>,
}
struct HandleInner {
    tx: mpsc::Sender<Command>,
    events: broadcast::Sender<Event>,
    status: watch::Receiver<Option<Result<EndReason, Error>>>,
    abort: AbortHandle,
}
impl Drop for HandleInner {
    fn drop(&mut self) {
        self.abort.abort();
    }
}
impl SessionHandle {
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.inner.events.subscribe()
    }
    /// Cancellation after enqueue does not cancel submission: the actor owns it.
    /// For retries retain the entry and its ID, or reload the pending inbox.
    pub async fn submit(&self, entry: InboxEntry) -> Result<InboxStatus, Error> {
        validate_signal(&entry.message)?;
        if entry.id.is_empty() {
            return Err(Error::Configuration("inbox ID must not be empty".into()));
        }
        let (tx, rx) = oneshot::channel();
        self.send(Command::Submit(entry, tx)).await?;
        self.reply(rx).await?
    }
    async fn signal(&self, kind: SignalKind, message: impl Into<Message>) -> Result<String, Error> {
        let entry = InboxEntry::new(kind, message);
        let id = entry.id.clone();
        self.submit(entry).await?;
        Ok(id)
    }
    pub async fn follow_up(&self, message: impl Into<Message>) -> Result<String, Error> {
        self.signal(SignalKind::FollowUp, message).await
    }
    pub async fn steer(&self, message: impl Into<Message>) -> Result<String, Error> {
        self.signal(SignalKind::Steer, message).await
    }
    pub async fn interrupt(&self, message: impl Into<Message>) -> Result<String, Error> {
        self.signal(SignalKind::Interrupt, message).await
    }
    pub async fn pause(&self) -> Result<(), Error> {
        self.send(Command::Pause).await
    }
    pub async fn abort(&self) -> Result<(), Error> {
        self.send(Command::Abort).await
    }
    /// Resume the existing last conversation message without appending it again.
    /// System-only history cannot resume. Repeated resumes coalesce; resume during an active turn is a no-op.
    pub async fn resume(&self) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::Resume(tx)).await?;
        self.reply(rx).await?
    }
    /// Barrier for accepted work. Idle does not stop the session or close intake.
    pub async fn wait_for_idle(&self) -> Result<EndReason, Error> {
        if let Some(result) = self.inner.status.borrow().clone() {
            return result;
        }
        let (tx, rx) = oneshot::channel();
        if self.send(Command::Wait(tx)).await.is_err() {
            return self.stopped().await;
        }
        match rx.await {
            Ok(result) => result,
            Err(_) => self.stopped().await,
        }
    }
    /// Construct a snapshot only when requested. May include a recoverable tail
    /// while a tool batch is active; validate before using it as a provider input.
    pub async fn history(&self) -> Result<Vec<Message>, Error> {
        let (tx, rx) = oneshot::channel();
        self.send(Command::History(tx)).await?;
        self.reply(rx).await
    }
    pub async fn stopped(&self) -> Result<EndReason, Error> {
        let mut status = self.inner.status.clone();
        loop {
            if let Some(result) = status.borrow().clone() {
                return result;
            }
            status.changed().await.map_err(|_| Error::Closed)?;
        }
    }
    async fn send(&self, command: Command) -> Result<(), Error> {
        self.inner.tx.send(command).await.map_err(|_| Error::Closed)
    }
    async fn reply<T>(&self, rx: oneshot::Receiver<T>) -> Result<T, Error> {
        match rx.await {
            Ok(value) => Ok(value),
            Err(_) => Err(self.stopped().await.err().unwrap_or(Error::Closed)),
        }
    }
}

enum Command {
    Submit(InboxEntry, oneshot::Sender<Result<InboxStatus, Error>>),
    Resume(oneshot::Sender<Result<(), Error>>),
    Pause,
    Abort,
    Wait(oneshot::Sender<Result<EndReason, Error>>),
    History(oneshot::Sender<Vec<Message>>),
}
#[derive(Default)]
struct Queue {
    immediate: VecDeque<InboxEntry>,
    steers: VecDeque<InboxEntry>,
    follow_ups: VecDeque<InboxEntry>,
    resume: bool,
}
impl Queue {
    fn len(&self) -> usize {
        self.immediate.len() + self.steers.len() + self.follow_ups.len() + usize::from(self.resume)
    }
    fn contains(&self, id: &str) -> bool {
        self.immediate
            .iter()
            .chain(&self.steers)
            .chain(&self.follow_ups)
            .any(|e| e.id == id)
    }
    fn push(&mut self, entry: InboxEntry) {
        match entry.kind {
            SignalKind::Interrupt => self.immediate.push_front(entry),
            SignalKind::Steer => self.steers.push_back(entry),
            SignalKind::FollowUp => self.follow_ups.push_back(entry),
        }
    }
    fn next(&mut self) -> Option<Vec<InboxEntry>> {
        if let Some(entry) = self.immediate.pop_front() {
            Some(vec![entry])
        } else if !self.steers.is_empty() {
            Some(self.steers.drain(..).collect())
        } else if std::mem::take(&mut self.resume) {
            Some(Vec::new())
        } else {
            self.follow_ups.pop_front().map(|e| vec![e])
        }
    }
}

struct Actor {
    config: Session,
    commands: mpsc::Receiver<Command>,
    events: broadcast::Sender<Event>,
    queue: Queue,
    waiters: Vec<oneshot::Sender<Result<EndReason, Error>>>,
    paused: bool,
    loop_deadline: Option<Instant>,
    receipts: BTreeMap<String, ToolResult>,
}
impl Actor {
    fn new(
        mut config: Session,
        commands: mpsc::Receiver<Command>,
        events: broadcast::Sender<Event>,
    ) -> Self {
        let mut queue = Queue::default();
        // Restored interrupts retain their persisted order (live interrupts have
        // newest-first priority). Do not reverse them by replaying push_front.
        for entry in std::mem::take(&mut config.pending) {
            if entry.kind == SignalKind::Interrupt {
                queue.immediate.push_back(entry);
            } else {
                queue.push(entry);
            }
        }
        let loop_deadline = config.loop_timeout.map(|d| Instant::now() + d);
        Self {
            config,
            commands,
            events,
            queue,
            waiters: Vec::new(),
            paused: false,
            loop_deadline,
            receipts: BTreeMap::new(),
        }
    }
    fn emit(&self, event: Event) {
        let _ = self.events.send(event);
    }
    fn emit_rig(&self, item: MultiTurnStreamItem) {
        if self.events.receiver_count() > 0 {
            self.emit(Event::Rig(Arc::new(item)));
        }
    }
    fn io_deadline(&self, execution: Option<Instant>) -> Instant {
        let io = Instant::now() + self.config.io_timeout;
        execution.map_or(io, |d| d.min(io))
    }
    async fn commit(
        &mut self,
        messages: Vec<Message>,
        ack_inbox_ids: Vec<String>,
        outcome: Option<Outcome>,
        deadline: Instant,
    ) -> Result<(), Error> {
        validate_append(&self.config.history, &messages)?;
        let commit = Arc::new(Commit {
            id: Uuid::new_v4(),
            expected_messages: self.config.history.len(),
            messages,
            ack_inbox_ids,
            outcome,
        });
        let error = match timeout_at(deadline, self.config.store.commit((*commit).clone())).await {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(e.to_string()),
            Err(_) => Some("storage deadline elapsed".into()),
        };
        if let Some(message) = error {
            return Err(Error::CommitUncertain { commit, message });
        }
        self.config.history.extend(commit.messages.clone());
        self.emit(Event::Committed(commit));
        Ok(())
    }
    async fn repair_tail(&mut self, deadline: Instant) -> Result<(), Error> {
        let Some(last) = self.config.history.last() else {
            return Ok(());
        };
        let calls: Vec<_> = calls(last).into_iter().cloned().collect();
        if calls.is_empty() {
            return Ok(());
        }
        let position = self.config.history.len() - 1;
        let mut results = Vec::new();
        for call in calls {
            let key = ToolKey {
                assistant_position: position,
                call_id: call.id.to_string(),
            };
            let result = match self.receipts.remove(call.id.as_str()) {
                Some(result) => Some(result),
                None => timeout_at(deadline, self.config.store.load_tool_result(key))
                    .await
                    .map_err(|_| Error::Store {
                        operation: "load tool receipt",
                        message: "storage deadline elapsed".into(),
                    })?
                    .map_err(|e| Error::Store {
                        operation: "load tool receipt",
                        message: e.to_string(),
                    })?,
            }
            .unwrap_or(ToolResult {
                call: call.id.clone(),
                provider: call.provider.clone(),
                name: call.function.name.clone(),
                content: vec![ToolResultContent::text(RECOVERY)],
            });
            check_result(&call, &result)?;
            results.push(UserContent::ToolResult(result));
        }
        self.commit(
            vec![Message::User { content: results }],
            Vec::new(),
            None,
            deadline,
        )
        .await
    }
    async fn run(&mut self) -> Result<EndReason, Error> {
        self.repair_tail(self.io_deadline(self.loop_deadline))
            .await?;
        let mut idle = false;
        loop {
            if self.loop_deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(EndReason::LoopTimedOut);
            }
            // Drain a bounded snapshot, so continuous senders cannot starve work.
            for _ in 0..self.config.capacity {
                let Ok(command) = self.commands.try_recv() else {
                    break;
                };
                if let Some(reason) = self.command(command, false, self.loop_deadline).await? {
                    return Ok(reason);
                }
            }
            if self.paused {
                return Ok(EndReason::Paused);
            }
            if let Some(entries) = self.queue.next() {
                idle = false;
                let reason = self.turn(entries).await?;
                if !matches!(reason, EndReason::Completed | EndReason::Interrupted) {
                    return Ok(reason);
                }
                continue;
            }
            for waiter in self.waiters.drain(..) {
                let _ = waiter.send(Ok(EndReason::Idle));
            }
            if !idle {
                self.emit(Event::Idle);
                idle = true;
            }
            tokio::select! {
                command = self.commands.recv() => {
                    let Some(command) = command else { return Ok(EndReason::Aborted); };
                    if let Some(reason) = self.command(command, false, self.loop_deadline).await? { return Ok(reason); }
                }
                _ = wait_deadline(self.loop_deadline) => return Ok(EndReason::LoopTimedOut),
            }
        }
    }
    async fn command(
        &mut self,
        command: Command,
        active: bool,
        deadline: Option<Instant>,
    ) -> Result<Option<EndReason>, Error> {
        match command {
            Command::Submit(entry, reply) => {
                if self.queue.len() >= self.config.capacity && !self.queue.contains(&entry.id) {
                    let _ = reply.send(Err(Error::QueueFull));
                    return Ok(None);
                }
                let status = timeout_at(
                    self.io_deadline(deadline),
                    self.config.store.submit(entry.clone()),
                )
                .await
                .map_err(|_| Error::Store {
                    operation: "submit inbox (completion unknown; retry the same entry)",
                    message: "storage deadline elapsed".into(),
                })?
                .map_err(|e| Error::Store {
                    operation: "submit inbox (retry the same entry)",
                    message: e.to_string(),
                });
                match status {
                    Ok(status) => {
                        let interrupt = status == InboxStatus::Pending
                            && !self.queue.contains(&entry.id)
                            && entry.kind == SignalKind::Interrupt
                            && active
                            && !self.paused;
                        if status == InboxStatus::Pending && !self.queue.contains(&entry.id) {
                            self.queue.push(entry);
                        }
                        let _ = reply.send(Ok(status));
                        if interrupt {
                            return Ok(Some(EndReason::Interrupted));
                        }
                    }
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Resume(reply) => {
                let result = if active {
                    Ok(())
                } else if !matches!(
                    self.config.history.last(),
                    Some(Message::User { .. } | Message::Assistant { .. })
                ) {
                    Err(Error::InvalidHistory(
                        "resume requires a user or assistant message".into(),
                    ))
                } else if self.queue.len() >= self.config.capacity && !self.queue.resume {
                    Err(Error::QueueFull)
                } else {
                    self.queue.resume = true;
                    Ok(())
                };
                let _ = reply.send(result);
            }
            Command::Pause => self.paused = true,
            Command::Abort => return Ok(Some(EndReason::Aborted)),
            Command::History(reply) => {
                let _ = reply.send(self.config.history.clone());
            }
            Command::Wait(reply) => {
                self.waiters.retain(|waiter| !waiter.is_closed());
                if self.waiters.len() >= self.config.capacity {
                    let _ = reply.send(Err(Error::QueueFull));
                } else {
                    self.waiters.push(reply);
                }
            }
        }
        Ok(None)
    }
    async fn checkpoint(&mut self, checkpoint: Checkpoint, deadline: Instant) -> Result<(), Error> {
        match checkpoint {
            Checkpoint::BeforeRequest(expected_len) => {
                if expected_len != self.config.history.len() {
                    return Err(Error::InvalidHistory("Rig changed transcript without a canonical checkpoint; invalid-tool recovery and retry feedback require upstream checkpoint support".into()));
                }
                if self
                    .config
                    .history
                    .last()
                    .is_some_and(|m| !calls(m).is_empty())
                {
                    Err(Error::InvalidHistory(
                        "Rig requested a new model turn before the durable tool batch completed"
                            .into(),
                    ))
                } else {
                    Ok(())
                }
            }
            Checkpoint::BeforeTool(id) => {
                if self
                    .config
                    .history
                    .last()
                    .is_some_and(|m| calls(m).iter().any(|c| c.id.as_str() == id))
                {
                    Ok(())
                } else {
                    Err(Error::InvalidHistory("tool dispatch without a durable canonical assistant checkpoint; recovered invalid tool calls are not supported".into()))
                }
            }
            Checkpoint::Assistant(message) => {
                self.receipts.clear();
                self.commit(vec![message], Vec::new(), None, deadline).await
            }
            Checkpoint::Result { id, name, content } => {
                let Some(last) = self.config.history.last() else {
                    return Err(Error::InvalidHistory(
                        "tool result before assistant checkpoint".into(),
                    ));
                };
                let pending = calls(last);
                let Some(call) = pending.iter().find(|call| call.id.as_str() == id) else {
                    return Err(Error::InvalidHistory(
                        "tool result outside its pending assistant batch".into(),
                    ));
                };
                let result = ToolResult {
                    call: call.id.clone(),
                    provider: call.provider.clone(),
                    name,
                    content,
                };
                let key = ToolKey {
                    assistant_position: self.config.history.len() - 1,
                    call_id: id.clone(),
                };
                let error = match timeout_at(
                    deadline,
                    self.config
                        .store
                        .save_tool_result(key.clone(), result.clone()),
                )
                .await
                {
                    Ok(Ok(())) => None,
                    Ok(Err(e)) => Some(e.to_string()),
                    Err(_) => Some("storage deadline elapsed".into()),
                };
                if let Some(message) = error {
                    return Err(Error::ToolWriteUncertain {
                        key,
                        result: Arc::new(result),
                        message,
                    });
                }
                self.receipts.insert(id, result);
                if self.receipts.len() == pending.len() {
                    let results = pending
                        .iter()
                        .map(|c| UserContent::ToolResult(self.receipts[c.id.as_str()].clone()))
                        .collect();
                    self.commit(
                        vec![Message::User { content: results }],
                        Vec::new(),
                        None,
                        deadline,
                    )
                    .await?;
                    self.receipts.clear();
                }
                Ok(())
            }
        }
    }
    async fn turn(&mut self, entries: Vec<InboxEntry>) -> Result<EndReason, Error> {
        let turn_deadline = self.config.turn_timeout.map(|d| Instant::now() + d);
        let deadline = match (turn_deadline, self.loop_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let timeout_reason = if self.loop_deadline.is_some() && deadline == self.loop_deadline {
            EndReason::LoopTimedOut
        } else {
            EndReason::TurnTimedOut
        };
        if !entries.is_empty() {
            let messages = entries.iter().map(|e| e.message.clone()).collect();
            let ids = entries.into_iter().map(|e| e.id).collect();
            self.commit(messages, ids, None, self.io_deadline(deadline))
                .await?;
        }
        let Some((prompt, history)) = self.config.history.split_last() else {
            return Err(Error::InvalidHistory("cannot run empty history".into()));
        };
        let (check_tx, mut check_rx) = mpsc::channel(1);
        let mut stream = self
            .config
            .agent
            .runner(prompt.clone())
            .history(history.to_vec())
            .without_memory()
            .max_turns(self.config.max_turns)
            .add_hook(DurableHook(check_tx))
            .stream()
            .await;
        let mut usage = Usage::new();
        let mut usage_known = true;
        let mut calls_seen = 0;
        let mut finish_reason = None;
        let reason = loop {
            // A delivered durable checkpoint wins over cancellation: it is a
            // completed boundary and must either persist or report uncertainty.
            tokio::select! {
                biased;
                Some((checkpoint, reply)) = check_rx.recv() => {
                    self.checkpoint(checkpoint, self.io_deadline(deadline)).await?;
                    let _ = reply.send(());
                }
                _ = wait_deadline(deadline) => break timeout_reason,
                next = async {
                    tokio::select! {
                        command = self.commands.recv() => Next::Command(command),
                        item = stream.next() => Next::Item(item),
                    }
                } => {
                    let item = match next {
                        Next::Command(command) => {
                            let Some(command) = command else { break EndReason::Aborted; };
                            if let Some(reason) = self.command(command, true, deadline).await? { break reason; }
                            continue;
                        }
                        Next::Item(item) => item,
                    };
                    match item {
                        Some(Ok(item)) => {
                            match &item {
                                MultiTurnStreamItem::CompletionCall(call) => { calls_seen += 1; usage_known &= call.usage.has_values(); usage += call.usage; }
                                MultiTurnStreamItem::FinalResponse(_) => { self.emit_rig(item); break finish_reason.unwrap_or(EndReason::Completed); }
                                _ => {}
                            }
                            // Canonical finish reason is carried by the boundary hook
                            // via the stream's completion-call accounting below.
                            if let MultiTurnStreamItem::CompletionCall(call) = &item {
                                finish_reason = call.finish_reason.as_ref().and_then(|r| match r { FinishReason::Length => Some(EndReason::Length), FinishReason::ContentFilter => Some(EndReason::ContentFilter), _ => None });
                            }
                            self.emit_rig(item);
                        }
                        Some(Err(error)) => { usage_known = false; let reason = classify(&error); self.emit(Event::RigError(Arc::new(error))); break reason; }
                        None => { usage_known = false; break EndReason::ProviderError("Rig stream ended without a final response".into()); }
                    }
                }
            }
        };
        // A hook may have queued its checkpoint during the same poll that
        // selected a control command. Save that accepted boundary before drop.
        let finalization = self.io_deadline(None);
        while let Ok((checkpoint, reply)) = check_rx.try_recv() {
            self.checkpoint(checkpoint, finalization).await?;
            let _ = reply.send(());
        }
        // Drop all model/tool/hook futures before recovering the journal tail.
        drop(stream);
        drop(check_rx);
        self.repair_tail(finalization).await?;
        let outcome = Outcome {
            run_id: Uuid::new_v4(),
            reason: reason.clone(),
            usage: (usage_known && calls_seen > 0).then_some(usage),
            finished_at: SystemTime::now(),
        };
        self.commit(Vec::new(), Vec::new(), Some(outcome), finalization)
            .await?;
        Ok(reason)
    }
}
enum Next {
    Command(Option<Command>),
    Item(Option<Result<MultiTurnStreamItem, StreamingError>>),
}
async fn wait_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}
fn classify(error: &StreamingError) -> EndReason {
    match error {
        StreamingError::Prompt(error) => match error.as_ref() {
            PromptError::MaxTurnsError { .. } => EndReason::MaxTurns,
            PromptError::PromptCancelled { reason, .. } => EndReason::HookStopped(reason.clone()),
            PromptError::CompletionError(error) => EndReason::ProviderError(error.to_string()),
            other => EndReason::RunError(other.to_string()),
        },
        other => EndReason::ProviderError(other.to_string()),
    }
}

enum Checkpoint {
    BeforeRequest(usize),
    BeforeTool(String),
    Assistant(Message),
    Result {
        id: String,
        name: String,
        content: Vec<ToolResultContent>,
    },
}
struct DurableHook(mpsc::Sender<(Checkpoint, oneshot::Sender<()>)>);
impl DurableHook {
    async fn checkpoint(&self, checkpoint: Checkpoint) -> bool {
        let (tx, rx) = oneshot::channel();
        self.0.send((checkpoint, tx)).await.is_ok() && rx.await.is_ok()
    }
}
impl AgentHook for DurableHook {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        if self
            .checkpoint(Checkpoint::BeforeRequest(event.history.len() + 1))
            .await
        {
            CompletionCallAction::Continue
        } else {
            CompletionCallAction::Stop("session stopped".into())
        }
    }
    async fn on_model_turn_finished(
        &self,
        _: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let message = Message::Assistant {
            id: event.identity.message_id.clone(),
            content: event.content.clone(),
        };
        if self.checkpoint(Checkpoint::Assistant(message)).await {
            ModelTurnAction::Continue
        } else {
            ModelTurnAction::Stop("session stopped".into())
        }
    }
    async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        let Some(id) = event.tool_call_id else {
            return ToolCallAction::Stop("tool call has no ID".into());
        };
        if self.checkpoint(Checkpoint::BeforeTool(id.into())).await {
            ToolCallAction::Run
        } else {
            ToolCallAction::Stop("session stopped".into())
        }
    }
    async fn on_tool_result(
        &self,
        _: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let Some(id) = event.tool_call_id else {
            return ToolResultAction::Stop("tool result has no call ID".into());
        };
        let checkpoint = Checkpoint::Result {
            id: id.into(),
            name: event.tool_name.into(),
            content: event.presentation.as_content().to_vec(),
        };
        if self.checkpoint(checkpoint).await {
            ToolResultAction::Keep
        } else {
            ToolResultAction::Stop("session stopped".into())
        }
    }
}
