//! Thin, fallible integration with Rig memory policies. Custom async compaction
//! can use `AgentHook::on_completion_call`, `Compactor` and `RequestPatch` directly.
use crate::validate_history;
use rig::{
    agent::{
        AgentHook,
        hook::{CompletionCall, CompletionCallAction, HookContext, RequestPatch},
    },
    message::Message,
};
use rig_memory::MemoryPolicy;

/// Apply a Rig memory policy at every model request, including tool rounds.
/// The current prompt must survive unchanged; the durable journal is untouched.
/// Policies must budget for the preamble, tools and output headroom separately.
pub struct ContextPolicy<P>(pub P);
impl<P: MemoryPolicy> AgentHook for ContextPolicy<P> {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        let mut messages = event.history.to_vec();
        messages.push(event.prompt.clone());
        match self.0.apply(messages).and_then(|mut messages| {
            validate_history(&messages)
                .map_err(|e| rig_memory::MemoryError::Internal(e.to_string()))?;
            if messages.pop().as_ref() != Some(event.prompt) {
                return Err(rig_memory::MemoryError::Internal(
                    "context policy must retain the current prompt unchanged".into(),
                ));
            }
            Ok(messages)
        }) {
            Ok(history) => CompletionCallAction::Patch(RequestPatch::new().history(history)),
            Err(error) => CompletionCallAction::Stop(format!("context policy: {error}")),
        }
    }
}

/// Validate a custom compactor's request view before returning its history patch.
/// Useful for async hooks that implement application-specific summary persistence.
pub fn history_patch(
    history: Vec<Message>,
    prompt: &Message,
) -> Result<RequestPatch, crate::Error> {
    let mut full = history.clone();
    full.push(prompt.clone());
    validate_history(&full)?;
    Ok(RequestPatch::new().history(history))
}

use rig::wasm_compat::WasmBoxedFuture;
use rig_memory::{CompactingMemory, Compactor, ConversationMemory, MemoryError};
use std::sync::{Arc, Mutex};

/// Rig's rolling compaction adapter, applied before each request rather than
/// only on session load. Create one instance per conversation; do not share an
/// agent carrying this hook between conversations.
///
/// Summary state is a derived, process-local cache. A fresh instance rebuilds
/// it from the full journal on restart. Side-effectful compactors should
/// deduplicate by conversation ID and input content, as Rig's `Compactor`
/// contract requires. Bound the summary separately from the retained window.
pub struct CompactingContext<P, C: Compactor> {
    conversation_id: String,
    source: ContextSource,
    memory: CompactingMemory<ContextSource, P, C>,
}
impl<P: MemoryPolicy, C: Compactor> CompactingContext<P, C> {
    pub fn new(conversation_id: impl Into<String>, policy: P, compactor: C) -> Self {
        let source = ContextSource::default();
        Self {
            conversation_id: conversation_id.into(),
            memory: CompactingMemory::new(source.clone(), policy, compactor),
            source,
        }
    }
}
impl<P: MemoryPolicy, C: Compactor> AgentHook for CompactingContext<P, C> {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        let mut messages = event.history.to_vec();
        messages.push(event.prompt.clone());
        // Release the synchronous lock before compaction/LLM IO.
        *self.source.0.lock().expect("context source poisoned") = messages;
        match self
            .memory
            .load(&self.conversation_id)
            .await
            .and_then(|mut messages| {
                if messages.pop().as_ref() != Some(event.prompt) {
                    return Err(MemoryError::Internal(
                        "compaction must retain the current prompt unchanged".into(),
                    ));
                }
                history_patch(messages, event.prompt)
                    .map_err(|e| MemoryError::Internal(e.to_string()))
            }) {
            Ok(patch) => CompletionCallAction::Patch(patch),
            Err(error) => CompletionCallAction::Stop(format!("compaction: {error}")),
        }
    }
}

// Read-only request input for upstream CompactingMemory. No duplicate durable
// store, append path or custom summary/watermark state machine.
#[derive(Clone, Default)]
struct ContextSource(Arc<Mutex<Vec<Message>>>);
impl ConversationMemory for ContextSource {
    fn load<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            self.0
                .lock()
                .map(|m| m.clone())
                .map_err(|_| MemoryError::Internal("context source poisoned".into()))
        })
    }
    fn append<'a>(
        &'a self,
        _: &'a str,
        _: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async { Err(MemoryError::Internal("context source is read-only".into())) })
    }
    fn clear<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async { Err(MemoryError::Internal("context source is read-only".into())) })
    }
}
