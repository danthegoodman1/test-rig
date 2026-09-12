//! Application-owned storage for one conversation. See [`Store`] for the contract.
use crate::{Outcome, journal::validate_append};
use rig::message::{Message, ToolResult};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::SystemTime,
};
use uuid::Uuid;

pub type StoreError = Box<dyn StdError + Send + Sync>;
pub type StoreFuture<T> = Pin<Box<dyn Future<Output = Result<T, StoreError>> + Send>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalKind {
    FollowUp,
    Steer,
    Interrupt,
}

/// A stable inbox identity. Retain the same entry when retrying a submission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InboxEntry {
    pub id: String,
    pub kind: SignalKind,
    pub message: Message,
    pub submitted_at: SystemTime,
}
impl InboxEntry {
    pub fn new(kind: SignalKind, message: impl Into<Message>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kind,
            message: message.into(),
            submitted_at: SystemTime::now(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InboxStatus {
    Pending,
    Acked,
}

/// Receipt identity scoped to an assistant's journal position, so providers may
/// reuse call IDs in later turns without overwriting earlier results.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ToolKey {
    pub assistant_position: usize,
    pub call_id: String,
}

/// One append-only transaction. Its ID and payload must survive uncertain writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Commit {
    pub id: Uuid,
    pub expected_messages: usize,
    pub messages: Vec<Message>,
    pub ack_inbox_ids: Vec<String>,
    pub outcome: Option<Outcome>,
}

/// Storage is scoped to **one conversation with one active writer**.
///
/// - `submit` inserts once, returns the existing status on identical retry, and
///   rejects reuse of an ID with different contents.
/// - `commit` atomically appends, ACKs and records the outcome. The same commit
///   ID/payload must be safe to retry even after a write-then-error or cancellation.
///   Check the ID before the expected position; conflicting retries must fail.
/// - `save_tool_result` is idempotent for the same key/result; conflicting results
///   must fail. Receipts are saved before the driver may advance past that result.
///
/// A trailing assistant tool batch is a recoverable journal checkpoint, not yet
/// a provider request. Loading history/inbox and resolving uncertain writes are
/// caller-owned. An error or dropped write future does **not** mean rollback.
pub trait Store: Send + Sync + 'static {
    fn submit(&self, entry: InboxEntry) -> StoreFuture<InboxStatus>;
    fn commit(&self, commit: Commit) -> StoreFuture<()>;
    fn save_tool_result(&self, key: ToolKey, result: ToolResult) -> StoreFuture<()>;
    fn load_tool_result(&self, key: ToolKey) -> StoreFuture<Option<ToolResult>>;
}

/// Reference implementation with the production contract, but no disk durability.
#[derive(Clone, Debug, Default)]
pub struct MemoryStore {
    inner: Arc<Mutex<MemoryState>>,
}
#[derive(Clone, Debug, Default)]
struct MemoryState {
    snapshot: Snapshot,
    inbox_positions: BTreeMap<String, usize>,
    acked: BTreeSet<String>,
}
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub history: Vec<Message>,
    pub inbox: Vec<InboxEntry>,
    pub acked_ids: Vec<String>,
    pub commits: BTreeMap<Uuid, Commit>,
    pub tool_results: BTreeMap<ToolKey, ToolResult>,
}
impl Snapshot {
    pub fn pending(&self) -> Vec<InboxEntry> {
        let acked: BTreeSet<_> = self.acked_ids.iter().collect();
        self.inbox
            .iter()
            .filter(|e| !acked.contains(&e.id))
            .cloned()
            .collect()
    }
}
impl MemoryStore {
    pub fn snapshot(&self) -> Snapshot {
        self.inner
            .lock()
            .expect("memory store poisoned")
            .snapshot
            .clone()
    }
}
impl Store for MemoryStore {
    fn submit(&self, entry: InboxEntry) -> StoreFuture<InboxStatus> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let mut state = inner.lock().map_err(|_| "memory store poisoned")?;
            if let Some(index) = state.inbox_positions.get(&entry.id) {
                let existing = &state.snapshot.inbox[*index];
                if existing != &entry {
                    return Err("inbox ID reused with different contents".into());
                }
            } else {
                let index = state.snapshot.inbox.len();
                state.inbox_positions.insert(entry.id.clone(), index);
                state.snapshot.inbox.push(entry.clone());
            }
            Ok(if state.acked.contains(&entry.id) {
                InboxStatus::Acked
            } else {
                InboxStatus::Pending
            })
        })
    }
    fn commit(&self, commit: Commit) -> StoreFuture<()> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let mut state = inner.lock().map_err(|_| "memory store poisoned")?;
            if let Some(existing) = state.snapshot.commits.get(&commit.id) {
                return if existing == &commit {
                    Ok(())
                } else {
                    Err("commit ID reused with different contents".into())
                };
            }
            if state.snapshot.history.len() != commit.expected_messages {
                return Err("stale journal position".into());
            }
            validate_append(&state.snapshot.history, &commit.messages)
                .map_err(|e| -> StoreError { Box::new(e) })?;
            let mut seen_ids = BTreeSet::new();
            let mut used_messages = BTreeSet::new();
            for id in &commit.ack_inbox_ids {
                let Some(index) = state.inbox_positions.get(id) else {
                    return Err("ACK refers to unknown inbox entry".into());
                };
                let entry = &state.snapshot.inbox[*index];
                if state.acked.contains(id) || !seen_ids.insert(id) {
                    return Err("duplicate inbox ACK".into());
                }
                let Some(index) =
                    commit
                        .messages
                        .iter()
                        .enumerate()
                        .find_map(|(index, message)| {
                            (message == &entry.message && !used_messages.contains(&index))
                                .then_some(index)
                        })
                else {
                    return Err("ACK requires the inbox message in this transaction".into());
                };
                used_messages.insert(index);
            }
            state.snapshot.history.extend(commit.messages.clone());
            state.acked.extend(commit.ack_inbox_ids.clone());
            state
                .snapshot
                .acked_ids
                .extend(commit.ack_inbox_ids.clone());
            state.snapshot.commits.insert(commit.id, commit);
            Ok(())
        })
    }
    fn save_tool_result(&self, key: ToolKey, result: ToolResult) -> StoreFuture<()> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let mut state = inner.lock().map_err(|_| "memory store poisoned")?;
            if key.call_id != result.call.as_str() {
                return Err("receipt key does not match result".into());
            }
            if let Some(existing) = state.snapshot.tool_results.get(&key) {
                if existing != &result {
                    return Err("conflicting tool receipt".into());
                }
            } else {
                state.snapshot.tool_results.insert(key, result);
            }
            Ok(())
        })
    }
    fn load_tool_result(&self, key: ToolKey) -> StoreFuture<Option<ToolResult>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            Ok(inner
                .lock()
                .map_err(|_| "memory store poisoned")?
                .snapshot
                .tool_results
                .get(&key)
                .cloned())
        })
    }
}
