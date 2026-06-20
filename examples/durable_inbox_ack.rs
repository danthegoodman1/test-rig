//! Demonstrates the managed durable harness: submit durable signals, start the
//! agent manager, then wait until queued work is idle.

use rig::{
    agent::AgentBuilder,
    test_utils::{MockCompletionModel, MockStreamEvent},
};
use rigloop::{DurableAgentHarness, InMemoryDurableAgentStore};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::new(scripted_model()).build();
    let store = InMemoryDurableAgentStore::default();
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
    let state = store.snapshot();

    assert!(state.acked_inbox_ids.contains(&start.id));
    assert!(state.pending_inbox_entries.is_empty());

    println!("end_reason: {:?}", end_reason);
    println!("persisted messages: {}", state.persisted_messages.len());
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
