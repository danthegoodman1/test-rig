//! Demonstrates the managed durable harness: submit durable signals, start the
//! agent manager, then wait until queued work is idle.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use rig::{
    agent::AgentBuilder,
    message::{Message, ToolResult},
    test_utils::{MockCompletionModel, MockStreamEvent},
};
use rigloop::{
    DurableAgentFuture, DurableAgentHarness, DurableAgentStore, DurableInboxEntry,
    IncrementalToolResultPersistence, PersistMessagesArgs, ToolResultKey,
};

#[derive(Clone, Debug, Default)]
struct MemoryDurableStore {
    inner: Arc<Mutex<MemoryDurableState>>,
}

#[derive(Debug, Default)]
struct MemoryDurableState {
    inbox: Vec<DurableInboxEntry>,
    acked_inbox_ids: Vec<String>,
    persisted: Vec<Message>,
    tool_results: HashMap<ToolResultKey, ToolResult>,
}

impl IncrementalToolResultPersistence for MemoryDurableStore {
    fn persist_tool_result(
        &self,
        key: ToolResultKey,
        result: ToolResult,
    ) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            store.inner.lock().unwrap().tool_results.insert(key, result);
            Ok(())
        })
    }

    fn load_tool_result(&self, key: ToolResultKey) -> DurableAgentFuture<Option<ToolResult>> {
        let store = self.clone();
        Box::pin(async move { Ok(store.inner.lock().unwrap().tool_results.get(&key).cloned()) })
    }
}

impl DurableAgentStore for MemoryDurableStore {
    fn submit_inbox_entry(&self, entry: DurableInboxEntry) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            println!("submit> kind={:?} id={}", entry.kind, entry.id);
            store.inner.lock().unwrap().inbox.push(entry);
            Ok(())
        })
    }

    fn persist_messages_and_ack(&self, args: PersistMessagesArgs) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            println!(
                "persist> messages={} ack_ids={:?}",
                args.messages.len(),
                args.ack_inbox_entry_ids
            );
            let mut state = store.inner.lock().unwrap();
            state.persisted.extend(args.messages);
            state.acked_inbox_ids.extend(args.ack_inbox_entry_ids);
            Ok(())
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::new(scripted_model()).build();
    let store = MemoryDurableStore::default();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let start = harness.follow_up("start").await?;
    for index in 0..10 {
        harness.steer(format!("steer-{index}")).await?;
    }
    for index in 0..3 {
        harness.follow_up(format!("follow-up-{index}")).await?;
    }

    harness.start()?;
    let end_reason = harness.wait_for_idle().await?;
    let state = store.inner.lock().unwrap();

    assert!(state.acked_inbox_ids.contains(&start.id));
    assert_eq!(state.inbox.len(), state.acked_inbox_ids.len());

    println!("end_reason: {:?}", end_reason);
    println!("persisted messages: {}", state.persisted.len());
    println!("acked inbox rows: {:?}", state.acked_inbox_ids);

    Ok(())
}

fn scripted_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns((0..32).map(|index| {
        vec![
            MockStreamEvent::text(format!("synthetic response {index}")),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    }))
}
