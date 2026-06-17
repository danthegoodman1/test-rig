use rig::{
    agent::CompletionCall,
    message::{AssistantContent, Message, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, ToolCallDeltaContent},
};

pub(crate) fn render_assistant_item<R>(item: &StreamedAssistantContent<R>) -> bool {
    match item {
        StreamedAssistantContent::Text(text) => {
            println!("assistant delta> {}", text.text);
            false
        }
        StreamedAssistantContent::ToolCallDelta {
            internal_call_id,
            content,
            ..
        } => match content {
            ToolCallDeltaContent::Name(name) => {
                println!("tool delta> {internal_call_id} name={name}");
                false
            }
            ToolCallDeltaContent::Delta(delta) => {
                println!("tool delta> {internal_call_id} args+={delta:?}");
                true
            }
        },
        StreamedAssistantContent::ToolCall {
            internal_call_id,
            tool_call,
        } => {
            println!(
                "tool complete> {internal_call_id} {}({})",
                tool_call.function.name, tool_call.function.arguments
            );
            false
        }
        StreamedAssistantContent::Reasoning(_)
        | StreamedAssistantContent::ReasoningDelta { .. } => false,
        StreamedAssistantContent::Final(_) => false,
    }
}

pub(crate) fn render_user_item(item: &StreamedUserContent) {
    let StreamedUserContent::ToolResult {
        internal_call_id,
        tool_result,
    } = item;

    let rendered = render_tool_result_content(&tool_result.content).join(", ");
    println!("tool result> {internal_call_id} {rendered}");
}

pub(crate) fn record_usage(call: CompletionCall) {
    let total_tokens = call
        .usage
        .map(|usage| usage.total_tokens)
        .unwrap_or_default();
    println!(
        "usage> call={} total_tokens={}",
        call.call_index, total_tokens
    );
}

pub(crate) fn print_user_prompt(message: &Message) {
    if let Some(text) = first_text(message) {
        println!("user> {text}");
    }
}

pub(crate) fn print_history(label: &str, history: &[Message]) {
    println!("{label}: {} message(s)", history.len());
    for (idx, message) in history.iter().enumerate() {
        println!("  {idx}: {}", describe_message(message));
    }
}

fn describe_message(message: &Message) -> String {
    match message {
        Message::User { content } => content
            .iter()
            .map(|item| match item {
                UserContent::Text(text) => format!("user {:?}", text.text),
                UserContent::ToolResult(tool_result) => format!(
                    "tool result {} {:?}",
                    tool_result.id,
                    render_tool_result_content(&tool_result.content)
                ),
                other => format!("user {other:?}"),
            })
            .collect::<Vec<_>>()
            .join(", "),
        Message::Assistant { content, .. } => content
            .iter()
            .map(|item| match item {
                AssistantContent::Text(text) => format!("assistant {:?}", text.text),
                AssistantContent::ToolCall(tool_call) => format!(
                    "assistant tool call {}({})",
                    tool_call.function.name, tool_call.function.arguments
                ),
                other => format!("assistant {other:?}"),
            })
            .collect::<Vec<_>>()
            .join(", "),
        Message::System { content } => format!("system {content:?}"),
    }
}

fn render_tool_result_content(content: &rig::OneOrMany<ToolResultContent>) -> Vec<String> {
    content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => text.text.clone(),
            other => format!("{other:?}"),
        })
        .collect()
}

pub(crate) fn first_text(message: &Message) -> Option<String> {
    match message {
        Message::User { content } => content.iter().find_map(|item| match item {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }),
        Message::Assistant { content, .. } => content.iter().find_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }),
        Message::System { content } => Some(content.clone()),
    }
}
