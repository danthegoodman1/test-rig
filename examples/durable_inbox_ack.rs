//! Demonstrates ACKing a simulated durable inbox only after rigloop's
//! persistence hook sees the submitted user message in committed history.

use std::sync::{Arc, Mutex};

use rig::{
    OneOrMany,
    agent::AgentBuilder,
    message::{Message, Text, UserContent},
    test_utils::{MockCompletionModel, MockStreamEvent},
};
use rigloop::{AgentLoop, AgentLoopHandle, TurnHookAction};
use serde_json::json;

const INBOX_ID_KEY: &str = "demo_inbox_id";

#[derive(Clone, Debug)]
struct Inbox {
    inner: Arc<Mutex<InboxState>>,
}

#[derive(Debug, Default)]
struct InboxState {
    next_id: usize,
    pending: Vec<Entry>,
    acked: Vec<Entry>,
    persisted: Vec<Message>,
}

#[derive(Clone, Debug)]
struct Entry {
    id: String,
    text: String,
}

impl Inbox {
    fn new() -> Self {
        Self {
            inner: Arc::default(),
        }
    }

    fn steer<R>(
        &self,
        handle: &AgentLoopHandle<R>,
        text: impl Into<String>,
    ) -> anyhow::Result<Entry>
    where
        R: Clone,
    {
        let (entry, message) = self.enqueue(text);
        handle.steer(message)?;
        println!("submit> kind=steer id={} text={:?}", entry.id, entry.text);
        Ok(entry)
    }

    fn follow_up<R>(
        &self,
        handle: &AgentLoopHandle<R>,
        text: impl Into<String>,
    ) -> anyhow::Result<Entry>
    where
        R: Clone,
    {
        let (entry, message) = self.enqueue(text);
        handle.follow_up(message)?;
        println!(
            "submit> kind=follow_up id={} text={:?}",
            entry.id, entry.text
        );
        Ok(entry)
    }

    fn enqueue(&self, text: impl Into<String>) -> (Entry, Message) {
        let mut state = self.inner.lock().unwrap();
        state.next_id += 1;

        let entry = Entry {
            id: format!("inbox-{}", state.next_id),
            text: text.into(),
        };
        let message = tagged_user_message(&entry.id, &entry.text);
        state.pending.push(entry.clone());
        (entry, message)
    }

    fn checkpoint_from_history(&self, history: &[Message]) {
        let mut state = self.inner.lock().unwrap();
        let start = state.persisted.len();
        assert!(
            start <= history.len(),
            "persisted cursor is ahead of candidate history"
        );
        let new_messages = &history[start..];

        // In a real store, this is one transaction:
        // slice unread history, append it, then ACK matching inbox rows.
        println!(
            "persist> history[{}..{}] ({} new messages)",
            start,
            history.len(),
            new_messages.len()
        );
        state.persisted.extend(new_messages.iter().cloned());

        for inbox_id in new_messages.iter().filter_map(inbox_id) {
            let Some(index) = state.pending.iter().position(|entry| entry.id == inbox_id) else {
                continue;
            };
            let entry = state.pending.remove(index);
            println!("ack> id={} text={:?}", entry.id, entry.text);
            state.acked.push(entry);
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::new(scripted_model()).build();
    let inbox = Inbox::new();

    let inbox_for_hook = inbox.clone();
    let loop_ = AgentLoop::new(agent).with_turn_hook(move |_state, turn| {
        let inbox = inbox_for_hook.clone();
        async move {
            inbox.checkpoint_from_history(&turn.history);
            Ok::<_, std::convert::Infallible>(TurnHookAction::Continue)
        }
    });

    let handle = loop_.prompt(Message::user("start"));

    let steer_entries: Vec<_> = (0..10)
        .map(|index| inbox.steer(&handle, format!("steer-{index}")))
        .collect::<Result<_, _>>()?;
    let follow_up_entries: Vec<_> = (0..3)
        .map(|index| inbox.follow_up(&handle, format!("follow-up-{index}")))
        .collect::<Result<_, _>>()?;

    let result = handle.wait().await?;
    let state = inbox.inner.lock().unwrap();

    assert!(state.pending.is_empty());
    assert_eq!(
        state
            .acked
            .iter()
            .map(|entry| &entry.id)
            .collect::<Vec<_>>(),
        steer_entries
            .iter()
            .chain(follow_up_entries.iter())
            .map(|entry| &entry.id)
            .collect::<Vec<_>>()
    );

    println!("end_reason: {:?}", result.end_reason);
    println!("persisted messages: {}", state.persisted.len());
    println!(
        "acked inbox rows: {:?}",
        state
            .acked
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<Vec<_>>()
    );

    Ok(())
}

fn tagged_user_message(inbox_id: &str, text: &str) -> Message {
    Message::User {
        content: OneOrMany::one(UserContent::Text(Text {
            text: text.to_string(),
            additional_params: Some(json!({ INBOX_ID_KEY: inbox_id })),
        })),
    }
}

fn inbox_id(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };

    content.iter().find_map(|content| match content {
        UserContent::Text(text) => text
            .additional_params
            .as_ref()?
            .get(INBOX_ID_KEY)?
            .as_str()
            .map(ToOwned::to_owned),
        _ => None,
    })
}

fn scripted_model() -> MockCompletionModel {
    // Extra turns keep the example deterministic whether queued steers batch
    // together immediately or drain across several loop turns.
    MockCompletionModel::from_stream_turns((0..32).map(|index| {
        vec![
            MockStreamEvent::text(format!("synthetic response {index}")),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    }))
}
