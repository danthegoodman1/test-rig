//! Request-only context views over Rig's memory windows. Each hook patches what
//! one model request sends, including tool rounds; the durable journal is untouched.
use crate::validate_history;
use rig::{
    agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch},
    message::Message,
};
use rig_memory::recent::{Recent, view};
use std::{error::Error as StdError, future::Future, sync::Arc};
use tokio::sync::Mutex;

/// Send only the newest messages a Rig [`Recent`] rule keeps, such as
/// `LastMessages(20)` or `TokenBudget::new(8_000)`. The window never starts on
/// a result whose call it cut off. The current prompt must survive. Budget the
/// preamble, tools and output headroom separately.
pub struct ContextPolicy<R>(pub R);
impl<R: Recent> AgentHook for ContextPolicy<R> {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let messages = request_messages(&event);
        match view(&messages, &self.0).split_last() {
            Some((_, history)) => {
                CompletionCallAction::Patch(RequestPatch::new().history(history.to_vec()))
            }
            None => CompletionCallAction::Stop(
                "context policy: the window must retain the current prompt".into(),
            ),
        }
    }
}

/// Validate a custom context view before returning its history patch.
/// Useful for async hooks that implement application-specific summaries.
pub fn history_patch(
    history: Vec<Message>,
    prompt: &Message,
) -> Result<RequestPatch, crate::Error> {
    let mut full = history.clone();
    full.push(prompt.clone());
    validate_history(&full)?;
    Ok(RequestPatch::new().history(history))
}

/// Summarizes messages as they leave the request window.
pub trait Compactor: Send + Sync {
    /// Fold `evicted`, oldest first, into the `previous` summary and return
    /// the new summary message.
    fn compact(
        &self,
        evicted: &[Message],
        previous: Option<&Message>,
    ) -> impl Future<Output = Result<Message, Box<dyn StdError + Send + Sync>>> + Send;
}

/// Rolling compaction: a summary of everything before a [`Recent`] window,
/// followed by the window. Create one instance per conversation; do not share
/// an agent carrying this hook between conversations.
///
/// The summary is a derived, process-local cache. A fresh instance rebuilds it
/// from the full journal on restart. Side-effectful compactors should
/// deduplicate by conversation and input content. Bound the summary separately
/// from the retained window.
pub struct CompactingContext<R, C> {
    recent: R,
    compactor: C,
    /// How many leading messages the summary covers, and the summary.
    summary: Mutex<Option<(usize, Message)>>,
}
impl<R: Recent, C: Compactor> CompactingContext<R, C> {
    pub fn new(recent: R, compactor: C) -> Self {
        Self {
            recent,
            compactor,
            summary: Mutex::new(None),
        }
    }
}
impl<R: Recent, C: Compactor> AgentHook for CompactingContext<R, C> {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let messages = request_messages(&event);
        let start = messages.len() - view(&messages, &self.recent).len();
        let mut summary = self.summary.lock().await;
        // The journal only grows; a shorter one is another conversation.
        if summary
            .as_ref()
            .is_some_and(|(covered, _)| *covered > messages.len())
        {
            *summary = None;
        }
        let covered = summary.as_ref().map_or(0, |(covered, _)| *covered);
        if start > covered {
            let evicted: Vec<Message> = messages[covered..start]
                .iter()
                .map(|m| (**m).clone())
                .collect();
            let previous = summary.as_ref().map(|(_, message)| message);
            match self.compactor.compact(&evicted, previous).await {
                Ok(message) => *summary = Some((start, message)),
                Err(error) => return CompletionCallAction::Stop(format!("compaction: {error}")),
            }
        }
        let start = start.max(covered);
        if start == messages.len() {
            return CompletionCallAction::Stop(
                "compaction: the window must retain the current prompt".into(),
            );
        }
        let window = &messages[start..messages.len() - 1];
        let history = summary
            .as_ref()
            .map(|(_, message)| message.clone())
            .into_iter()
            .chain(window.iter().map(|m| (**m).clone()))
            .collect();
        match history_patch(history, event.prompt) {
            Ok(patch) => CompletionCallAction::Patch(patch),
            Err(error) => CompletionCallAction::Stop(format!("compaction: {error}")),
        }
    }
}

fn request_messages(event: &CompletionCallEvent<'_>) -> Vec<Arc<Message>> {
    let mut messages = event.history.to_vec();
    messages.push(Arc::new(event.prompt.clone()));
    messages
}
