//! Demonstrates app-owned compaction with the managed durable harness.
//!
//! The harness persists appendable transcript snapshots while it runs. When an
//! checkpoint handler decides compaction would be useful. If another turn is
//! already queued, it requests an abort; otherwise the application defers the
//! rewrite until it actually decides to continue.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use rig::{
    agent::AgentBuilder,
    message::{AssistantContent, Message, UserContent},
    test_utils::{MockCompletionModel, MockStreamEvent},
};
use rigloop::{
    DurableAgentHarness, DurableCheckpoint, DurableCheckpointAction, EndReason,
    InMemoryDurableAgentStore,
};

const COMPACTION_INPUT_TOKEN_THRESHOLD: u64 = 100;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = InMemoryDurableAgentStore::default();
    let compaction_recommended = Arc::new(AtomicBool::new(false));
    let agent = AgentBuilder::new(pre_compaction_model()).build();
    let handler_compaction_recommended = compaction_recommended.clone();
    let harness =
        DurableAgentHarness::new(agent, store.clone()).with_checkpoint_handler(move |checkpoint| {
            let handler_compaction_recommended = handler_compaction_recommended.clone();
            async move {
                if let DurableCheckpoint::TurnBoundary(turn) = checkpoint {
                    let input_tokens = turn.usage.map(|usage| usage.input_tokens).unwrap_or(0);
                    if input_tokens >= COMPACTION_INPUT_TOKEN_THRESHOLD {
                        handler_compaction_recommended.store(true, Ordering::SeqCst);
                        if turn.end_reason.is_none() {
                            println!(
                                "checkpoint> compaction requested at {input_tokens} input tokens"
                            );
                            return Ok::<_, rigloop::DurableAgentError>(
                                DurableCheckpointAction::Abort {
                                    reason: "compaction requested".to_string(),
                                },
                            );
                        }
                        println!(
                            "checkpoint> compaction deferred at terminal boundary ({input_tokens} input tokens)"
                        );
                    }
                }

                Ok(DurableCheckpointAction::Continue)
            }
        });

    harness.follow_up("start a long session").await?;
    harness.start()?;
    let end_reason = harness.wait_for_idle().await?;

    println!("first end_reason: {end_reason:?}");
    assert_eq!(end_reason, EndReason::Idle);
    assert!(compaction_recommended.load(Ordering::SeqCst));
    drop(harness);

    let compacted = if compaction_recommended.load(Ordering::SeqCst) {
        let compacted = compact_history(&store.load_history());
        store.replace_history(compacted.clone());
        println!("compacted history messages: {}", compacted.len());
        compacted
    } else {
        store.load_history()
    };

    let agent = AgentBuilder::new(post_compaction_model()).build();
    let harness = DurableAgentHarness::new(agent, store.clone())
        .with_history(compacted)
        .with_inbox_id_generator(|index| format!("after-compaction-{index}"));

    harness
        .follow_up("continue from the compacted context")
        .await?;
    harness.start()?;
    let end_reason = harness.wait_for_idle().await?;

    println!("second end_reason: {end_reason:?}");
    println!("final durable messages: {}", store.load_history().len());

    Ok(())
}

fn compact_history(history: &[Message]) -> Vec<Message> {
    let user_summary = user_texts(history).join(" | ");
    let assistant_summary = assistant_texts(history).join(" | ");

    vec![Message::system(format!(
        "Conversation summary:\nUser asked: {user_summary}\nAssistant answered: {assistant_summary}"
    ))]
}

fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| {
            let Message::User { content } = message else {
                return None;
            };

            content.iter().find_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
        })
        .collect()
}

fn assistant_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| {
            let Message::Assistant { content, .. } = message else {
                return None;
            };

            content.iter().find_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
        })
        .collect()
}

fn pre_compaction_model() -> MockCompletionModel {
    let mut usage = rig::completion::Usage::new();
    usage.input_tokens = 125;
    usage.output_tokens = 16;
    usage.total_tokens = usage.input_tokens + usage.output_tokens;

    MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text(
            "This answer is intentionally long enough to cross the example compaction threshold.",
        ),
        MockStreamEvent::final_response(usage),
    ]])
}

fn post_compaction_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("continued from compacted history"),
        MockStreamEvent::final_response_with_default_usage(),
    ]])
}
