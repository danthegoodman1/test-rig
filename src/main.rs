use std::{error::Error, fmt};

use agentloop::AgentLoop;
use rig::{
    agent::AgentBuilder,
    completion::ToolDefinition,
    message::Message,
    test_utils::{MockCompletionModel, MockStreamEvent},
    tool::Tool,
};
use serde_json::{Value, json};

struct EchoTool;

#[derive(Debug)]
struct EchoError;

impl fmt::Display for EchoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("echo tool error")
    }
}

impl Error for EchoError {}

impl Tool for EchoTool {
    const NAME: &'static str = "echo";

    type Error = EchoError;
    type Args = Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Echo a text field back to the model".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let text = args
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("<missing text>");
        Ok(format!("echo tool saw: {text}"))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::new(scripted_model()).tool(EchoTool).build();
    let agent_loop = AgentLoop::new(agent).with_persistence_hook(|messages| async move {
        let rendered = serde_json::to_string_pretty(&messages)?;
        println!(
            "persist> appending {} message(s):\n{rendered}",
            messages.len()
        );
        Ok::<(), serde_json::Error>(())
    });
    let handle = agent_loop.prompt(Message::user(
        "Start by calling the echo tool with a draft input.",
    ));

    handle.steer(Message::user(
        "Steer: stop building that tool input and answer directly.",
    ))?;
    handle.follow_up(Message::user("Also summarize what happened."))?;

    let result = handle.wait().await?;
    println!("end_reason: {:?}", result.end_reason);
    println!(
        "last_response: {}",
        result.last_response.unwrap_or_default()
    );
    println!("history messages: {}", result.history.len());

    Ok(())
}

fn scripted_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("call_1", "internal_1", "echo"),
            MockStreamEvent::tool_call_arguments_delta("call_1", "internal_1", r#"{"text":"draft"#),
            MockStreamEvent::tool_call_arguments_delta("call_1", "internal_1", r#" input"}"#),
            MockStreamEvent::tool_call("call_1", "echo", json!({"text": "draft input"})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("Original turn finished after the echo tool result."),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("Queued steer handled by the harness."),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("Follow-up handled after steering."),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ])
}
