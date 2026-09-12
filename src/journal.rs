use crate::Error;
use rig::message::{AssistantContent, Message, ToolCall, ToolResult, UserContent};
use std::collections::HashSet;

pub(crate) fn calls(message: &Message) -> Vec<&ToolCall> {
    match message {
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
pub(crate) fn check_result(call: &ToolCall, result: &ToolResult) -> Result<(), Error> {
    if call.id != result.call || call.provider != result.provider || result.content.is_empty() {
        return Err(Error::InvalidHistory(
            "tool result identity/content does not match its call".into(),
        ));
    }
    Ok(())
}
/// Validate provider ordering and identity. Partial result batches and orphans
/// are rejected; this function never discards or rewrites caller data.
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
    let mut pending: Vec<&ToolCall> = Vec::new();
    for message in messages {
        if matches!(message, Message::User { content } if content.is_empty())
            || matches!(message, Message::Assistant { content, .. } if content.is_empty())
        {
            return Err(invalid("empty message content"));
        }
        if !pending.is_empty() {
            let Message::User { content } = message else {
                return Err(invalid(
                    "tool calls require an immediately following user result batch",
                ));
            };
            let results: Vec<_> = content
                .iter()
                .filter_map(|c| match c {
                    UserContent::ToolResult(r) => Some(r),
                    _ => None,
                })
                .collect();
            if results.len() != pending.len() {
                return Err(invalid(
                    "incomplete or extra tool results; normalize stored history before starting",
                ));
            }
            let mut seen = HashSet::new();
            for result in results {
                if !seen.insert(result.call.as_str()) {
                    return Err(invalid("duplicate tool result"));
                }
                let Some(call) = pending.iter().find(|c| c.id == result.call) else {
                    return Err(invalid("orphan tool result"));
                };
                check_result(call, result)?;
            }
            pending.clear();
        } else {
            if let Message::User { content } = message
                && content
                    .iter()
                    .any(|c| matches!(c, UserContent::ToolResult(_)))
            {
                return Err(invalid("orphan tool result"));
            }
            pending = calls(message);
            let mut seen = HashSet::new();
            if pending.iter().any(|c| !seen.insert(c.id.as_str())) {
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
