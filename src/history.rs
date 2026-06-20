use std::collections::HashSet;

use rig::{
    OneOrMany,
    message::{AssistantContent, Message, ToolCall, ToolResult, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent},
};

use crate::api::{
    AgentLoopError, IncrementalToolResultPersistence, InvalidMessageHistoryError, ToolResultKey,
};

#[derive(Default)]
pub(crate) struct PartialTurn {
    pending_tool_calls: Vec<PendingToolCall>,
    assistant_content: Vec<AssistantContent>,
    finished_assistant_message: Option<Message>,
    active_tool_results: Vec<UserContent>,
    pub(crate) completed_messages: Vec<Message>,
}

struct PendingToolCall {
    internal_call_id: String,
    tool_call: ToolCall,
}

impl PartialTurn {
    pub(crate) fn note_assistant_item<R>(&mut self, item: StreamedAssistantContent<R>)
    where
        R: Clone,
    {
        match item {
            StreamedAssistantContent::Text(text) => {
                self.assistant_content.push(AssistantContent::Text(text));
            }
            StreamedAssistantContent::Reasoning(reasoning) => {
                self.assistant_content
                    .push(AssistantContent::Reasoning(reasoning));
            }
            StreamedAssistantContent::ToolCall {
                internal_call_id,
                tool_call,
            } => {
                self.pending_tool_calls.push(PendingToolCall {
                    internal_call_id,
                    tool_call,
                });
            }
            StreamedAssistantContent::ToolCallDelta { .. }
            | StreamedAssistantContent::ReasoningDelta { .. }
            | StreamedAssistantContent::Final(_) => {}
        }
    }

    pub(crate) fn assistant_message(&self) -> Option<Message> {
        if self.finished_assistant_message.is_some() || !self.active_tool_results.is_empty() {
            return None;
        }

        let mut content = self.assistant_content.clone();
        content.extend(
            self.pending_tool_calls
                .iter()
                .map(|call| AssistantContent::ToolCall(call.tool_call.clone())),
        );

        if content.is_empty() {
            return None;
        }

        OneOrMany::many(content)
            .ok()
            .map(|content| Message::Assistant { id: None, content })
    }

    pub(crate) fn mark_assistant_message_finished(&mut self, message: Message) {
        self.finished_assistant_message = Some(message);
    }

    pub(crate) fn note_user_item(&mut self, item: StreamedUserContent) {
        let StreamedUserContent::ToolResult {
            internal_call_id,
            tool_result,
        } = item;

        let Some(position) = self
            .pending_tool_calls
            .iter()
            .position(|call| call.internal_call_id == internal_call_id)
        else {
            return;
        };

        if self.active_tool_results.is_empty() {
            let assistant_message = self
                .finished_assistant_message
                .take()
                .or_else(|| self.assistant_message());
            if let Some(message) = assistant_message {
                self.assistant_content.clear();
                self.completed_messages.push(message);
            }
        }

        self.pending_tool_calls.remove(position);
        self.active_tool_results
            .push(UserContent::ToolResult(tool_result));

        if self.pending_tool_calls.is_empty()
            && let Ok(content) = OneOrMany::many(std::mem::take(&mut self.active_tool_results))
        {
            self.completed_messages.push(Message::User { content });
        }
    }

    pub(crate) fn append_messages(&self, prompt: &Message) -> Option<Vec<Message>> {
        if self.completed_messages.is_empty() {
            return None;
        }

        let mut messages = Vec::with_capacity(self.completed_messages.len() + 1);
        messages.push(prompt.clone());
        messages.extend(self.completed_messages.clone());
        Some(messages)
    }
}

pub(crate) fn validate_message_history(
    messages: &[Message],
) -> Result<(), InvalidMessageHistoryError> {
    let mut pending_tool_calls: Option<(usize, Vec<String>)> = None;

    for (index, message) in messages.iter().enumerate() {
        if let Some((assistant_index, expected_ids)) = pending_tool_calls.take() {
            let Message::User { content } = message else {
                return Err(InvalidMessageHistoryError::new(format!(
                    "assistant message {assistant_index} contains tool calls, but message {index} is not the required user tool-result message"
                )));
            };

            let result_ids = tool_result_ids(content);
            if result_ids.is_empty() {
                return Err(InvalidMessageHistoryError::new(format!(
                    "assistant message {assistant_index} contains tool calls, but user message {index} contains no tool results"
                )));
            }

            ensure_unique_ids(&result_ids, index, "tool result")?;

            for id in &expected_ids {
                if !result_ids.contains(id) {
                    return Err(InvalidMessageHistoryError::new(format!(
                        "assistant tool call `{id}` at message {assistant_index} is missing a matching tool result in message {index}"
                    )));
                }
            }

            for id in &result_ids {
                if !expected_ids.contains(id) {
                    return Err(InvalidMessageHistoryError::new(format!(
                        "tool result `{id}` at message {index} does not match any tool call from message {assistant_index}"
                    )));
                }
            }

            continue;
        }

        match message {
            Message::Assistant { content, .. } => {
                let tool_call_ids = assistant_tool_call_ids(content);
                ensure_unique_ids(&tool_call_ids, index, "tool call")?;
                if !tool_call_ids.is_empty() {
                    pending_tool_calls = Some((index, tool_call_ids));
                }
            }
            Message::User { content } => {
                let result_ids = tool_result_ids(content);
                if let Some(id) = result_ids.first() {
                    return Err(InvalidMessageHistoryError::new(format!(
                        "tool result `{id}` at message {index} has no immediately preceding assistant tool call"
                    )));
                }
            }
            Message::System { .. } => {}
        }
    }

    if let Some((assistant_index, expected_ids)) = pending_tool_calls {
        return Err(InvalidMessageHistoryError::new(format!(
            "assistant message {assistant_index} contains unanswered tool calls: {}",
            expected_ids.join(", ")
        )));
    }

    Ok(())
}

pub(crate) fn validate_resume_history(
    messages: &[Message],
) -> Result<(), InvalidMessageHistoryError> {
    validate_message_history(messages)?;

    match messages.last() {
        Some(Message::User { .. } | Message::Assistant { .. }) => Ok(()),
        Some(Message::System { .. }) => Err(InvalidMessageHistoryError::new(
            "resume requires the last committed message to be user or assistant content",
        )),
        None => Err(InvalidMessageHistoryError::new(
            "resume requires at least one committed message",
        )),
    }
}

/// Repair assistant tool-call messages that are missing matching user tool results.
///
/// The returned history preserves existing messages, inserts a synthetic user
/// tool-result message after any unanswered assistant tool-call message, and
/// adds missing tool results to partial result batches. Histories that are
/// invalid for reasons other than unanswered tool calls are not normalized.
pub fn repair_unanswered_tool_calls<H, T>(
    history: H,
    repair_message: impl AsRef<str>,
) -> Vec<Message>
where
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let mut messages = history.into_iter().map(Into::into).collect::<Vec<_>>();
    let repair_message = repair_message.as_ref();

    let mut index = 0;
    while index < messages.len() {
        let expected_ids = match &messages[index] {
            Message::Assistant { content, .. } => assistant_tool_call_ids(content),
            Message::User { .. } | Message::System { .. } => {
                index += 1;
                continue;
            }
        };

        if expected_ids.is_empty() {
            index += 1;
            continue;
        }

        let next_index = index + 1;
        let Some(next_message) = messages.get_mut(next_index) else {
            messages.push(repair_tool_result_message(&expected_ids, repair_message));
            index += 2;
            continue;
        };

        let Message::User { content } = next_message else {
            messages.insert(
                next_index,
                repair_tool_result_message(&expected_ids, repair_message),
            );
            index += 2;
            continue;
        };

        let result_ids = tool_result_ids(content);
        if result_ids.is_empty() {
            messages.insert(
                next_index,
                repair_tool_result_message(&expected_ids, repair_message),
            );
            index += 2;
            continue;
        }

        let missing_ids = expected_ids
            .iter()
            .filter(|id| !result_ids.contains(id))
            .cloned()
            .collect::<Vec<_>>();

        if missing_ids.is_empty() {
            index += 2;
            continue;
        }

        let mut repaired_content = content.iter().cloned().collect::<Vec<_>>();
        repaired_content.extend(repair_tool_result_contents(&missing_ids, repair_message));
        if let Ok(next_content) = OneOrMany::many(repaired_content) {
            *content = next_content;
        }
        index += 2;
    }

    messages
}

/// Repair unanswered assistant tool calls, preferring persisted tool results when available.
///
/// For each missing tool result, this first calls
/// [`IncrementalToolResultPersistence::load_tool_result`]. If no persisted
/// result exists, it falls back to the supplied synthetic repair text.
pub async fn repair_unanswered_tool_calls_with_persistence<H, T>(
    history: H,
    repair_message: impl AsRef<str>,
    persistence: Option<&dyn IncrementalToolResultPersistence>,
) -> Result<Vec<Message>, AgentLoopError>
where
    H: IntoIterator<Item = T>,
    T: Into<Message>,
{
    let mut messages = history.into_iter().map(Into::into).collect::<Vec<_>>();
    let repair_message = repair_message.as_ref();

    let mut index = 0;
    while index < messages.len() {
        let expected_calls = match &messages[index] {
            Message::Assistant { content, .. } => assistant_tool_calls(content),
            Message::User { .. } | Message::System { .. } => {
                index += 1;
                continue;
            }
        };

        if expected_calls.is_empty() {
            index += 1;
            continue;
        }

        let next_index = index + 1;
        match messages.get(next_index) {
            Some(Message::User { content }) if !tool_result_ids(content).is_empty() => {
                let result_ids = tool_result_ids(content);
                if expected_calls
                    .iter()
                    .any(|tool_call| !result_ids.contains(&tool_call.id))
                {
                    let repaired_content = repair_tool_result_content(
                        content.iter().cloned().collect(),
                        &expected_calls,
                        repair_message,
                        persistence,
                    )
                    .await?;
                    let repaired_content = OneOrMany::many(repaired_content)
                        .expect("repair tool result content is non-empty");
                    if let Some(Message::User { content }) = messages.get_mut(next_index) {
                        *content = repaired_content;
                    }
                }
            }
            _ => {
                let repaired_content = repair_tool_result_content(
                    Vec::new(),
                    &expected_calls,
                    repair_message,
                    persistence,
                )
                .await?;
                let message = Message::User {
                    content: OneOrMany::many(repaired_content)
                        .expect("repair tool result content is non-empty"),
                };
                if next_index < messages.len() {
                    messages.insert(next_index, message);
                } else {
                    messages.push(message);
                }
            }
        }

        index += 2;
    }

    Ok(messages)
}

async fn repair_tool_result_content(
    existing_content: Vec<UserContent>,
    expected_calls: &[ToolCall],
    repair_message: &str,
    persistence: Option<&dyn IncrementalToolResultPersistence>,
) -> Result<Vec<UserContent>, AgentLoopError> {
    let mut repaired = existing_content
        .iter()
        .filter(|item| !matches!(item, UserContent::ToolResult(_)))
        .cloned()
        .collect::<Vec<_>>();

    for tool_call in expected_calls {
        let existing_result = existing_content.iter().find_map(|item| match item {
            UserContent::ToolResult(tool_result) if tool_result.id == tool_call.id => {
                Some(tool_result.clone())
            }
            _ => None,
        });

        let mut tool_result = match (existing_result, persistence) {
            (Some(tool_result), _) => tool_result,
            (None, Some(persistence)) => persistence
                .load_tool_result(ToolResultKey::from_tool_call(tool_call))
                .await
                .map_err(AgentLoopError::ToolResultPersistence)?
                .unwrap_or_else(|| ToolResult {
                    id: tool_call.id.clone(),
                    call_id: tool_call.call_id.clone(),
                    content: ToolResultContent::from_tool_output(repair_message.to_string()),
                }),
            (None, None) => ToolResult {
                id: tool_call.id.clone(),
                call_id: tool_call.call_id.clone(),
                content: ToolResultContent::from_tool_output(repair_message.to_string()),
            },
        };

        tool_result.id = tool_call.id.clone();
        if tool_result.call_id.is_none() {
            tool_result.call_id = tool_call.call_id.clone();
        }
        repaired.push(UserContent::ToolResult(tool_result));
    }

    Ok(repaired)
}

fn repair_tool_result_message(ids: &[String], repair_message: &str) -> Message {
    Message::User {
        content: OneOrMany::many(repair_tool_result_contents(ids, repair_message))
            .expect("repair tool result content is non-empty"),
    }
}

fn repair_tool_result_contents(ids: &[String], repair_message: &str) -> Vec<UserContent> {
    ids.iter()
        .map(|id| {
            UserContent::ToolResult(ToolResult {
                id: id.clone(),
                call_id: None,
                content: ToolResultContent::from_tool_output(repair_message.to_string()),
            })
        })
        .collect()
}

pub(crate) fn assistant_tool_call_ids(content: &OneOrMany<AssistantContent>) -> Vec<String> {
    content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.id.clone()),
            _ => None,
        })
        .collect()
}

fn assistant_tool_calls(content: &OneOrMany<AssistantContent>) -> Vec<ToolCall> {
    content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids(content: &OneOrMany<UserContent>) -> Vec<String> {
    content
        .iter()
        .filter_map(|item| match item {
            UserContent::ToolResult(tool_result) => Some(tool_result.id.clone()),
            _ => None,
        })
        .collect()
}

fn ensure_unique_ids(
    ids: &[String],
    message_index: usize,
    kind: &str,
) -> Result<(), InvalidMessageHistoryError> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(InvalidMessageHistoryError::new(format!(
                "duplicate {kind} id `{id}` at message {message_index}"
            )));
        }
    }

    Ok(())
}
