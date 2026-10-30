use crate::Error;
use rig::{
    conversation::pairing,
    message::{CallId, Message, ToolCall, ToolResult, UserContent},
};
use std::collections::HashSet;

pub(crate) fn calls(message: &Message) -> Vec<&ToolCall> {
    match message {
        Message::Assistant(turn) => turn.tool_calls().collect(),
        _ => Vec::new(),
    }
}
pub(crate) fn check_result(call: &ToolCall, result: &ToolResult) -> Result<(), Error> {
    if call.id != result.call || result.content.is_empty() {
        return Err(Error::InvalidHistory(
            "tool result identity/content does not match its call".into(),
        ));
    }
    Ok(())
}
/// Validate provider ordering and identity. Each tool batch must be answered
/// exactly once by the next message, a user message holding only its results.
/// Partial batches and orphans are rejected; this function never discards or
/// rewrites caller data.
pub fn validate_history(history: &[Message]) -> Result<(), Error> {
    validate(history.iter(), false)
}

pub(crate) fn validate_journal(history: &[Message]) -> Result<(), Error> {
    validate(history.iter(), true)
}
pub(crate) fn validate_append(history: &[Message], delta: &[Message]) -> Result<(), Error> {
    // Earlier history was validated on load/append. Only the seam can change.
    let previous = history.last().filter(|m| !calls(m).is_empty());
    validate(previous.into_iter().chain(delta.iter()), true)
}
fn validate<'a>(
    messages: impl Iterator<Item = &'a Message>,
    allow_tail: bool,
) -> Result<(), Error> {
    let invalid = |s: &str| Error::InvalidHistory(s.into());
    let mut pending: Vec<CallId> = Vec::new();
    for message in messages {
        let empty = match message {
            Message::User { content } => content.is_empty(),
            Message::Assistant(turn) => turn.content.is_empty(),
            Message::System { .. } => false,
        };
        if empty {
            return Err(invalid("empty message content"));
        }
        let content = match message {
            Message::User { content } => Some(content),
            _ => None,
        };
        let results = || {
            content.into_iter().flatten().filter_map(|c| match c {
                UserContent::ToolResult(r) => Some(r),
                _ => None,
            })
        };
        if !pending.is_empty() {
            let Some(content) = content else {
                return Err(invalid(
                    "tool calls require an immediately following user result batch",
                ));
            };
            // Rig matches repeated IDs as a multiset; IDs are unique here, so
            // this is an exact one-to-one answer.
            pairing::answers(&pending, content).map_err(|e| {
                Error::InvalidHistory(format!("{e}; normalize stored history before starting"))
            })?;
            if results().any(|r| r.content.is_empty()) {
                return Err(invalid("empty tool result content"));
            }
            pending.clear();
        } else {
            if results().next().is_some() {
                return Err(invalid("orphan tool result"));
            }
            pending = calls(message).into_iter().map(|c| c.id.clone()).collect();
            let mut seen = HashSet::new();
            if pending.iter().any(|id| !seen.insert(id)) {
                return Err(invalid("duplicate tool call"));
            }
        }
    }
    if !allow_tail && !pending.is_empty() {
        return Err(invalid("unanswered tool calls"));
    }
    Ok(())
}
pub(crate) fn validate_signal(message: &Message) -> Result<(), Error> {
    match message {
        Message::User { content }
            if !content.is_empty()
                && !content
                    .iter()
                    .any(|c| matches!(c, UserContent::ToolResult(_))) =>
        {
            Ok(())
        }
        _ => Err(Error::InvalidHistory(
            "signals must be nonempty user messages without tool results".into(),
        )),
    }
}
