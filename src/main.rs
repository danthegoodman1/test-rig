use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    sync::{Arc, Mutex},
};

use agentloop::{AgentLoop, TurnHookAction};
use rig::{
    agent::AgentBuilder,
    completion::ToolDefinition,
    message::Message,
    test_utils::{MockCompletionModel, MockStreamEvent},
    tool::Tool,
};
use serde_json::{Value, json};

type AgentId = String;

#[derive(Clone, Debug)]
struct DemoState {
    cursor: Arc<Mutex<PersistCursor>>,
    store: Arc<Mutex<MemoryStore>>,
}

#[derive(Clone, Debug)]
struct PersistCursor {
    agent_id: AgentId,
    generation_id: u64,
    next_message_index: usize,
}

#[derive(Default, Debug)]
struct MemoryStore {
    sessions: BTreeMap<(AgentId, u64), Vec<Message>>,
}

impl MemoryStore {
    fn append(&mut self, agent_id: &str, generation_id: u64, messages: Vec<Message>) -> usize {
        let session = self
            .sessions
            .entry((agent_id.to_string(), generation_id))
            .or_default();
        let start_index = session.len();
        session.extend(messages);
        start_index
    }

    fn print_sessions(&self) {
        println!("persisted sessions:");
        for ((agent_id, generation_id), messages) in &self.sessions {
            println!(
                "  agent={agent_id} generation={generation_id} messages={}",
                messages.len()
            );
        }
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
    let cursor = Arc::new(Mutex::new(PersistCursor {
        agent_id: "demo-agent".to_string(),
        generation_id: 0,
        next_message_index: 0,
    }));
    let store = Arc::new(Mutex::new(MemoryStore::default()));
    let app_state = DemoState {
        cursor,
        store: store.clone(),
    };

    let agent_loop = AgentLoop::new(agent)
        .with_app_state(app_state)
        .with_turn_hook(|state, turn| async move {
            if !turn.new_messages.is_empty() {
                let mut cursor = state.cursor.lock().unwrap();
                let mut store = state.store.lock().unwrap();
                let start_index = store.append(
                    &cursor.agent_id,
                    cursor.generation_id,
                    turn.new_messages.clone(),
                );

                println!(
                    "persist> agent={} generation={} append index={} count={}",
                    cursor.agent_id,
                    cursor.generation_id,
                    start_index,
                    turn.new_messages.len()
                );

                cursor.next_message_index = start_index + turn.new_messages.len();
            }

            if turn.history.len() < 4 {
                return Ok::<_, std::convert::Infallible>(TurnHookAction::Continue);
            }

            let summary = vec![Message::system(format!(
                "Compacted summary: previous generation contained {} messages. \
                     The user asked to start with an echo tool call; the tool returned \
                     a draft input result; the agent completed that turn.",
                turn.history.len()
            ))];

            let mut cursor = state.cursor.lock().unwrap();
            let mut store = state.store.lock().unwrap();
            let next_generation_id = cursor.generation_id + 1;
            let start_index = store.append(&cursor.agent_id, next_generation_id, summary.clone());

            println!(
                "compact> agent={} generation {} -> {} summary index={} count={}",
                cursor.agent_id,
                cursor.generation_id,
                next_generation_id,
                start_index,
                summary.len()
            );

            cursor.generation_id = next_generation_id;
            cursor.next_message_index = start_index + summary.len();

            Ok(TurnHookAction::ReplaceHistory(summary))
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
    store.lock().unwrap().print_sessions();

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
