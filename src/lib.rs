mod api;
mod engine;
mod history;

pub mod durable;

pub use api::{
    AgentLoop, AgentLoopError, AgentLoopErrorInfo, AgentLoopEvent, AgentLoopHandle,
    AgentLoopResult, ApiErrorInfo, ApiErrorKind, AssistantMessageContext,
    AssistantMessageHookError, DEFAULT_MAX_TURNS, DEFAULT_UNANSWERED_TOOL_CALL_REPAIR_MESSAGE,
    EndReason, IncrementalToolResultPersistence, InvalidMessageHistoryError, QueueKind,
    ToolResultKey, ToolResultPersistenceError, ToolResultPersistenceFuture, TurnHookAction,
    TurnHookContext, TurnHookError, TurnOutcomeKind,
};
pub use durable::{
    DurableAgentCallbackError, DurableAgentControl, DurableAgentError, DurableAgentFuture,
    DurableAgentHarness, DurableAgentStore, DurableAgentStoreError, DurableCheckpoint,
    DurableCheckpointAction, DurableInboxEntry, DurableInboxKind, DurableInboxStatus,
    DurableTurnOutcome, INBOX_ENTRY_ID_PARAM, InMemoryDurableAgentStore,
    InMemoryDurableAgentStoreSnapshot, PersistMessagesArgs, inbox_id_from_message,
    inbox_ids_from_messages, tag_message_with_inbox_id,
};
pub use history::{repair_unanswered_tool_calls, repair_unanswered_tool_calls_with_persistence};

pub(crate) use api::Command;
#[cfg(test)]
pub(crate) use api::{
    DEFAULT_TOOL_REPAIR_MESSAGE, EVENT_BUFFER_SIZE, SharedMessages, lock_messages,
};
#[cfg(test)]
pub(crate) use engine::{PendingTurn, Runner, RunnerInit};
pub(crate) use history::validate_message_history;
#[cfg(test)]
pub(crate) use history::{PartialTurn, assistant_tool_call_ids, validate_resume_history};

#[cfg(test)]
mod tests;
