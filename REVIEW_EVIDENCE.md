# Codebase review evidence

Companion to [REVIEW_PLAN.md](REVIEW_PLAN.md). Baseline: `29474e14876e1be51a8cfbef4b8deab30c9b6dad`, rigloop 0.2.1, Rig 0.38.2. This document preserves the pre-implementation review. Current implementation evidence is in [IMPLEMENTATION_EVIDENCE.md](IMPLEMENTATION_EVIDENCE.md). The test snippets below were appended to `src/tests.rs` in a temporary copy so they could reuse the existing imports and helpers.

## Baseline checks

| Command | Observed result |
| --- | --- |
| `cargo test --all-targets` | 70 tests passed; both examples compiled as test targets. |
| `cargo fmt --check` | Passed. |
| `cargo clippy --all-targets --all-features -- -D warnings` | Passed. |
| `cargo test --doc` | Passed, but there are zero doc tests. README snippets are not verified by this command. |

No paid/live provider requests, database integration tests, or performance benchmarks were run. Performance claims in the plan are source-level complexity findings, with measurement deferred to its benchmark gate. Published upstream 0.42 source and official documentation were inspected at review time; subsequent adoption is recorded separately. The ten probes below compile against the locked local dependency graph and all expose failures on the reviewed code.

## Upstream 0.42 source inspection

Inspected the published `rig-core`, `rig-agent` and `rig-memory` 0.42.0 crate archives. These observations establish API candidates and limitations, not integration-test results.

| Boundary | Source evidence | Implication |
| --- | --- | --- |
| Canonical model turn | `rig-agent/src/agent/hook.rs:584`: `ModelTurnFinished` carries canonical assistant blocks, identity, usage and finish reason before tools/finalization. | Replace delta reconstruction; verify acceptance/retry hook ordering before durable writes. |
| Request-only context | `rig-agent/src/agent/hook.rs:838`: `RequestPatch::history`; `runner.rs` test `history_patch_changes_sent_messages_not_transcript_on_both_surfaces`. | Reuse per-call history projection without rewriting durable history. |
| Tool batch failure | `rig-agent/src/agent/prompt_request/streaming.rs:650`: `drive_tool_calls` holds results until the whole batch succeeds; failure publishes/commits none and drains started siblings. | The result stream alone cannot preserve each completed tool across later batch failure or external cancellation. Durable per-result hooks/recovery need a spike. |
| Memory errors | `rig-agent/src/agent/runner.rs:559`: `append_run_messages` logs append errors and proceeds. | Built-in conversation memory cannot directly satisfy failure-propagating transcript/ACK commits. |
| Custom summary | `rig-core/src/memory.rs:298`: async `Compactor`; `rig-memory/src/lib.rs:1222`: `TemplateCompactor`. | Custom code can choose its own LLM prompt/model using evicted messages plus carry-over. The bundled template implementation concatenates text and optionally truncates bytes. |
| Cache policies | `rig-core/src/providers/anthropic/completion.rs:1670–1783`: manual, automatic, one-hour and static-prefix TTL setters. | Standard prefix/tail caching is built in. Explicit provider-specific tool markers are preserved within the breakpoint budget. |
| Arbitrary cache markers | Same provider source: `anthropic_text_content_from_message_text` sets `cache_control: None` for ordinary text (`:805–826`); raw text escape hatch permits only server-tool blocks (`:828–850`); manual caching clears message markers (`:2580–2604`). | No ordinary message-level API for arbitrary text breakpoints was found. Use request fixtures to validate a narrow provider adapter/upstream addition if exact placement is required; do not assume text `additional_params` is forwarded. |

Published sources: [agent hooks](https://docs.rs/crate/rig-agent/0.42.0/source/src/agent/hook.rs), [shared driver](https://docs.rs/crate/rig-agent/0.42.0/source/src/agent/prompt_request/streaming.rs), [runner](https://docs.rs/crate/rig-agent/0.42.0/source/src/agent/runner.rs), [memory contract](https://docs.rs/crate/rig-core/0.42.0/source/src/memory.rs), [memory adapters](https://docs.rs/crate/rig-memory/0.42.0/source/src/lib.rs), [Anthropic conversion/cache policy](https://docs.rs/crate/rig-core/0.42.0/source/src/providers/anthropic/completion.rs).

## Regression observations

| ID | Probe | Observed failure | Finding |
| --- | --- | --- | --- |
| E1 | `review_chunked_text_durable_is_not_duplicated` | Expected two messages, got three: user, assistant with two text blocks, then assistant with joined text. | F1 |
| E2 | `review_two_tool_calls_durable_are_not_duplicated` | Expected two tool results, got four. Separate per-call records and the final grouped records are both persisted. This is transcript duplication, not evidence that the tools executed twice. | F1 |
| E3 | `review_sync_repair_preserves_provider_call_id` | Expected `Some("call-1")`, got `None`. | F2 |
| E4 | `review_async_repair_does_not_hide_orphan_result` | Repair deletes the unrelated result and validation succeeds, hiding the malformed input. | F2 |
| E5 | `review_signal_at_loop_end_starts_next_run` | A durably accepted second signal produces `ManagerFailed("durable agent command channel is closed")`. The first terminal-event callback provides a deterministic handoff barrier. | F4 |
| E6 | `review_default_inbox_ids_survive_restart` | Both entries receive `inbox-1`; the new submission is already absent from `pending_inbox_entries` before being run. | F5 |
| E7 | `review_turn_timeout_includes_assistant_hook` | An outer 100 ms wait expires although the configured turn timeout is 10 ms. The assistant hook never resolves. | F6 |
| E8 | `review_provider_error_keeps_completed_tool_results` | After a successful tool result and a provider error on the next call, returned history is empty instead of retaining the result. | F3 |
| E9 | `review_restarting_partial_tool_batch_is_append_safe` | Managed restart fails with `tool result a at message 2 has no immediately preceding assistant tool call`. The repaired user message is incorrectly appended again. | F2 |
| E10 | `review_usage_absence_is_not_zero` | Missing usage is reported as `Some(Usage { ... all zero ... })` instead of `None`. | F7 |

These are assertions of intended behavior, so the review run is expected to exit unsuccessfully. E9 assumes automatic recovery; if the chosen API instead requires caller-owned normalization, replace its success assertion with a rejection-before-request assertion and a successful normalized-restart case. The implementation plan records that decision explicitly.

## Reproduce without modifying the checkout

Run this Python snippet from the repository root. It extracts the Rust block below, copies the baseline source/manifest/lockfile to a new temporary directory, and appends the probes to the copied test module. It reuses the checkout's build cache; it does not edit the checkout's source or manifests. The script reads the exact reviewed revision from Git, so subsequent implementation edits do not change these probes.

```python
from pathlib import Path
import os
import subprocess
import tempfile

root = Path.cwd()
evidence = (root / "REVIEW_EVIDENCE.md").read_text()
probes = evidence.split("```rust\n", 1)[1].split("\n```", 1)[0]
scratch = Path(tempfile.mkdtemp(prefix="rigloop-review-"))
baseline = "29474e14876e1be51a8cfbef4b8deab30c9b6dad"
source_files = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", baseline, "src"], text=True,
).splitlines()
for name in [*source_files, "Cargo.toml", "Cargo.lock", "README.md"]:
    destination = scratch / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(subprocess.check_output(["git", "show", f"{baseline}:{name}"]))
with (scratch / "src/tests.rs").open("a") as output:
    output.write("\n" + probes + "\n")
env = dict(os.environ, CARGO_TARGET_DIR=str(root / "target"))
print("Scratch copy:", scratch, flush=True)
result = subprocess.run(
    ["cargo", "test", "--locked", "--manifest-path", str(scratch / "Cargo.toml"),
     "review_", "--", "--nocapture"],
    env=env,
)
raise SystemExit(result.returncode)
```

## Probe source

```rust
#[tokio::test]
async fn review_chunked_text_durable_is_not_duplicated() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("hello"),
        MockStreamEvent::text(" world"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = InMemoryDurableAgentStore::default();
    let harness = DurableAgentHarness::new(AgentBuilder::new(model).build(), store.clone());
    harness.follow_up("start").await.unwrap();
    assert_eq!(harness.wait_for_idle().await.unwrap(), EndReason::Idle);
    let h = store.load_history();
    assert_eq!(h.len(), 2, "duplicated assistant: {h:?}");
}

#[tokio::test]
async fn review_two_tool_calls_durable_are_not_duplicated() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("a", "add", serde_json::json!({"x":1,"y":2})),
            MockStreamEvent::tool_call("b", "add", serde_json::json!({"x":3,"y":4})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = InMemoryDurableAgentStore::default();
    let harness = DurableAgentHarness::new(
        AgentBuilder::new(model).tool(MockAddTool).build(),
        store.clone(),
    );
    harness.follow_up("start").await.unwrap();
    let end = harness.wait_for_idle().await;
    assert!(end.is_ok(), "{end:?}");
    let h = store.load_history();
    assert_eq!(
        tool_results(h.iter()).len(),
        2,
        "duplicated tool results: {h:?}"
    );
}

#[test]
fn review_sync_repair_preserves_provider_call_id() {
    let mut call = test_tool_call("item-1");
    call.call_id = Some("call-1".into());
    let h = repair_unanswered_tool_calls(
        [Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(call)),
        }],
        "recovered",
    );
    assert_eq!(tool_results(h.iter())[0].call_id.as_deref(), Some("call-1"));
}

#[tokio::test]
async fn review_async_repair_does_not_hide_orphan_result() {
    let h = vec![
        Message::Assistant {
            id: None,
            content: OneOrMany::many([
                AssistantContent::ToolCall(test_tool_call("a")),
                AssistantContent::ToolCall(test_tool_call("b")),
            ])
            .unwrap(),
        },
        Message::User {
            content: OneOrMany::many([
                UserContent::ToolResult(test_tool_result("a")),
                UserContent::ToolResult(test_tool_result("orphan")),
            ])
            .unwrap(),
        },
    ];
    let h = repair_unanswered_tool_calls_with_persistence(h, "recovered", None)
        .await
        .unwrap();
    assert!(
        validate_message_history(&h).is_err(),
        "invalid result silently removed: {h:?}"
    );
}

#[tokio::test]
async fn review_signal_at_loop_end_starts_next_run() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("second"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = InMemoryDurableAgentStore::default();
    let e = entered.clone();
    let r = release.clone();
    let first_end = Arc::new(AtomicBool::new(true));
    let harness = DurableAgentHarness::new(AgentBuilder::new(model).build(), store.clone())
        .with_event_observer(move |event| {
            let e = e.clone();
            let r = r.clone();
            let first_end = first_end.clone();
            async move {
                if matches!(event, AgentLoopEvent::LoopEnded { .. })
                    && first_end.swap(false, Ordering::SeqCst)
                {
                    e.notify_one();
                    r.notified().await;
                }
                Ok::<_, std::io::Error>(())
            }
        });
    harness.follow_up("first").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), entered.notified())
        .await
        .unwrap();
    harness.follow_up("second").await.unwrap();
    release.notify_one();
    let outcome = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap();
    assert!(
        outcome.is_ok(),
        "accepted durable signal killed manager: {outcome:?}"
    );
}

#[tokio::test]
async fn review_default_inbox_ids_survive_restart() {
    let store = InMemoryDurableAgentStore::default();
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let harness = DurableAgentHarness::new(AgentBuilder::new(model).build(), store.clone());
    let first = harness.follow_up("first").await.unwrap();
    harness.wait_for_idle().await.unwrap();
    drop(harness);
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("second"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let harness = DurableAgentHarness::new(AgentBuilder::new(model).build(), store.clone())
        .with_history(store.load_history());
    let second = harness.follow_up("second").await.unwrap();
    assert_ne!(
        first.id,
        second.id,
        "new submission is already considered ACKed: {:?}",
        store.snapshot().pending_inbox_entries
    );
}

#[tokio::test]
async fn review_turn_timeout_includes_assistant_hook() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let h = AgentLoop::new(AgentBuilder::new(model).build())
        .turn_timeout(Duration::from_millis(10))
        .on_assistant_message_finished(|_, _| async {
            std::future::pending::<()>().await;
            Ok::<_, std::io::Error>(())
        })
        .prompt("start");
    let outcome = timeout(Duration::from_millis(100), h.wait()).await;
    assert!(
        outcome.is_ok(),
        "turn timeout cannot end an unresponsive hook"
    );
}

#[tokio::test]
async fn review_provider_error_keeps_completed_tool_results() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("a", "add", serde_json::json!({"x":1,"y":2})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![MockStreamEvent::error("provider failed")],
    ]);
    let result = AgentLoop::new(AgentBuilder::new(model).tool(MockAddTool).build())
        .prompt("start")
        .wait()
        .await
        .unwrap();
    assert!(matches!(result.end_reason, EndReason::ApiError { .. }));
    assert_eq!(
        tool_results(result.history.iter()).len(),
        1,
        "completed results lost: {:?}",
        result.history
    );
}

#[tokio::test]
async fn review_restarting_partial_tool_batch_is_append_safe() {
    let history = vec![
        Message::Assistant {
            id: None,
            content: OneOrMany::many([
                AssistantContent::ToolCall(test_tool_call("a")),
                AssistantContent::ToolCall(test_tool_call("b")),
            ])
            .unwrap(),
        },
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(test_tool_result("a"))),
        },
    ];
    let store = InMemoryDurableAgentStore::default();
    store.replace_history(history.clone());
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let harness = DurableAgentHarness::new(AgentBuilder::new(model).build(), store.clone())
        .with_history(history);
    let result = harness.wait_for_idle().await;
    assert!(result.is_ok(), "partial batch restart failed: {result:?}");
}

#[tokio::test]
async fn review_usage_absence_is_not_zero() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::FinalResponse(MockResponse::new()),
    ]]);
    let observed = Arc::new(Mutex::new(None));
    let observed_hook = observed.clone();
    let _ = AgentLoop::new(AgentBuilder::new(model).build())
        .with_turn_hook(move |_, t| {
            let observed = observed_hook.clone();
            async move {
                *observed.lock().unwrap() = Some(t.usage);
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        })
        .prompt("start")
        .wait()
        .await
        .unwrap();
    assert_eq!(*observed.lock().unwrap(), Some(None));
}
```
