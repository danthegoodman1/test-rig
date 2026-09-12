//! Durable intake and caller-owned restart, using only scripted model responses.
use rig::{
    agent::AgentBuilder,
    test_utils::{MockCompletionModel, MockStreamEvent as E},
};
use rigloop::{EndReason, MemoryStore, Session};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = MemoryStore::default();
    let agent = AgentBuilder::new(model()).build();
    let session = Session::new(agent, store.clone()).start()?;
    let id = session.follow_up("start").await?;
    assert_eq!(session.wait_for_idle().await?, EndReason::Idle);
    assert!(store.snapshot().acked_ids.contains(&id));
    session.pause().await?;
    session.stopped().await?;

    // A real application loads this snapshot from its own database. Explicit
    // restart preserves pending entry IDs and does not resubmit or retag them.
    let snapshot = store.snapshot();
    let resumed = Session::new(AgentBuilder::new(model()).build(), store.clone())
        .pending(snapshot.pending())
        .history(snapshot.history)
        .start()?;
    resumed.follow_up("continue after restart").await?;
    assert_eq!(resumed.wait_for_idle().await?, EndReason::Idle);
    assert!(store.snapshot().pending().is_empty());
    println!(
        "{} messages, {} ACKs",
        store.snapshot().history.len(),
        store.snapshot().acked_ids.len()
    );
    Ok(())
}
fn model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([vec![
        E::text("done"),
        E::final_response_with_default_usage(),
    ]])
}
