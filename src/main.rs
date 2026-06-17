use std::{
    collections::VecDeque,
    error::Error,
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

use futures::StreamExt;
use rig::{
    agent::{Agent, AgentBuilder, MultiTurnStreamItem},
    completion::ToolDefinition,
    message::Message,
    streaming::StreamingPrompt,
    test_utils::{MockCompletionModel, MockStreamEvent},
    tool::Tool,
};
use serde_json::{Value, json};

mod printing;

use printing::{
    first_text, print_history, print_user_prompt, record_usage, render_assistant_item,
    render_user_item,
};

#[derive(Clone, Default)]
struct SteerQueue {
    queued: Arc<Mutex<VecDeque<Message>>>,
}

impl SteerQueue {
    fn steer(&self, message: impl Into<Message>) {
        lock(&self.queued).push_back(message.into());
    }

    fn pop(&self) -> Option<Message> {
        lock(&self.queued).pop_front()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

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
    let steer_queue = SteerQueue::default();

    run_streaming_with_steer(
        &agent,
        steer_queue,
        Message::user("Start by calling the echo tool with a draft input."),
    )
    .await
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
            MockStreamEvent::text("Queued steer handled on the next outer-loop prompt."),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ])
}

async fn run_streaming_with_steer(
    agent: &Agent<MockCompletionModel>,
    steer_queue: SteerQueue,
    first_prompt: Message,
) -> anyhow::Result<()> {
    let mut history = Vec::<Message>::new();
    let mut next_prompt = first_prompt;
    let mut simulated_user_steer_sent = false;

    loop {
        println!("\n=== starting stream ===");
        print_user_prompt(&next_prompt);

        let mut completed = false;

        let mut stream = agent
            .stream_prompt(next_prompt.clone())
            .with_history(history.clone())
            .multi_turn(100)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(item)) => {
                    if render_assistant_item(&item) && !simulated_user_steer_sent {
                        let steer = Message::user(
                            "Steer: stop building that tool input and answer directly.",
                        );
                        println!(
                            "ui> queued steer: {}",
                            first_text(&steer).unwrap_or_default()
                        );
                        steer_queue.steer(steer);
                        simulated_user_steer_sent = true;
                    }
                }
                Ok(MultiTurnStreamItem::StreamUserItem(item)) => {
                    render_user_item(&item);
                }
                Ok(MultiTurnStreamItem::CompletionCall(call)) => {
                    record_usage(call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(final_response)) => {
                    println!("final> {}", final_response.response());
                    if let Some(messages) = final_response.history() {
                        history.extend(messages.iter().cloned());
                    }
                    completed = true;
                }
                Ok(_) => {}
                Err(err) => return Err(err.into()),
            }
        }

        if completed {
            print_history("history after completed stream", &history);
        }

        if let Some(steer_message) = steer_queue.pop() {
            next_prompt = steer_message;
            continue;
        }

        break;
    }

    Ok(())
}
