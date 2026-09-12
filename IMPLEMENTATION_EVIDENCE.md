# Implementation evidence

Current checkout: Rig/rig-core/rig-agent/rig-memory 0.42.0. The previous reviewed
revision was `29474e14876e1be51a8cfbef4b8deab30c9b6dad` on Rig 0.38.2.

## Implementation size and ownership

`wc -l` over production Rust source, including comments and blank lines:

| Version | Files | Lines |
| --- | --- | ---: |
| Review baseline | lib, api, engine, history, durable | 3,781 |
| Current | lib, session, journal, store, context | 1,404 |

That is 2,377 fewer lines (62.9%). Tests and examples are excluded on both sides.
The custom runtime modules were deleted. Rig supplies canonical model messages and turn
execution. The session retains intake, durability, cancellation and a narrow
append validator/recovery path. Memory policies and rolling compaction are
upstream implementations behind small request-only hooks.

## Verification

| Check | Result |
| --- | --- |
| `cargo test --all-targets --all-features --offline` | 36 behavioral tests passed; 1 timing probe intentionally ignored in the standard suite. Both examples compile. |
| `cargo fmt --check` | Passed. |
| `cargo clippy --all-targets --all-features --offline -- -D warnings` | Passed. |
| `cargo test --doc --offline` | 1 public API doctest compiled successfully. |
| `cargo run --example durable_inbox_ack --offline` | Passed; 4 messages and 2 ACKs after caller-owned restart. |
| `cargo run --example durable_compaction --offline` | Passed; custom LLM compaction prompt, 2-message request view and unchanged 4-message journal in the same session. |
| `cargo test --release --lib --offline append_bookkeeping_scaling -- --ignored --nocapture` | Passed; results below. |
| `cargo tree -i rig-core --offline` / `cargo tree -d --offline` | One coherent Rig 0.42.0 message/runtime graph; unrelated transitive platform/proc-macro version duplicates remain. |
| `git diff --check` | Passed. |

Dependencies were changed with `cargo add`/`cargo rm`; manifest and lockfile diffs
were inspected. Production additions are rig-memory, serde, uuid and thiserror;
serde_json is used only by tests/examples. Bytes is a dev dependency for captured
streaming HTTP fixtures, and Tokio test-util supplies paused clocks. No live
provider calls, credentials, database driver or publication were involved.

The old 70-test suite targeted the removed API and its internal assembler/queue
structures; it was not run unchanged against the new implementation. The new
suite tests the retained contracts directly through Rig 0.42 and the session.
The original failing probes remain reproducible against the exact old revision
using the script in REVIEW_EVIDENCE.md.

## Review regression mapping

Test names below are in `src/tests.rs`.

| Review probe | Replacement evidence |
| --- | --- |
| E1: duplicated chunked answer | `canonical_chunks_metadata_and_no_duplicate_commits` checks exact count, joined text and provider message ID. |
| E2: duplicated tool results | `grouped_calls_results_and_provider_identity_round_trip` compares the journal with Rig's next request and checks both provider IDs. |
| E3: lost provider call ID | Same identity test, plus strict result identity validation and real-receipt restart. The old synchronous repair API is removed. |
| E4: silently dropped orphan result | `validator_rejects_orphans_duplicate_results_and_provider_metadata_conflicts`; corrupted input returns an error rather than being rewritten. |
| E5: closed runner at idle handoff | `accepting_work_after_idle_never_closes_the_session`: 60 transitions on a multi-thread runtime with concurrent waiters. There is no runner handoff. |
| E6: restart inbox-ID collision | `drop_and_restart_restores_receipt_and_pending_identity`, `submissions_are_idempotent_and_conflicting_ids_fail`, and repeated-call-ID receipt scoping. |
| E7: timeout bypassed by hung hook | `turn_deadline_bounds_a_hung_user_hook`; paused-clock recovery-read, commit and tool-write bounds. |
| E8: provider failure erases completed work | `provider_failure_keeps_prior_completed_round_trip` checks retained results and persisted failure outcome. |
| E9: unsafe append after partial stored batch | `incomplete_or_corrupt_stored_result_batch_is_rejected_without_model_io`; the selected contract requires caller normalization of this case. Tail-only repair is exercised by restart tests. |
| E10: missing usage becomes zero | `missing_usage_stays_unknown`. |

Additional tests cover accepted result rewrites; discarded retry attempts;
assistant-write failure before tool dispatch; sibling stop/abort/interrupt;
paused pending intake; duplicate/stale commit rejection; submit-caller cancellation;
resume; bounded queue overload; explicit observer lag; reasoning assembly;
custom compactor restart and intake while compaction waits; intra-tool context
policies; preserved preamble; and captured Anthropic automatic/manual cache
markers with independent static-prefix TTLs.

The `unsupported_invalid_tool_repair_fails_before_execution` test verifies the
chosen upstream fallback: a recovery path that suppresses the canonical assistant
callback cannot execute a tool without a durable assistant checkpoint. Ordinary
configured hooks remain active and tested.

## Release bookkeeping measurement

One local run, mean microseconds per iteration over 25 iterations. The append
column performs actual `MemoryStore::commit` calls with a fixed one-message delta,
including UUID creation, validation, idempotency indexing and stored transaction
copies. Setup and initial journal loading are outside the timed region.

The comparison reproduces the removed checkpointer's successful-prefix pattern:
compare a complete history against a separately owned copy, then clone the full
history. It is a reference algorithm measurement, **not** a benchmark of the old
runtime as a whole.

| Retained messages | Body bytes/message | Append commit µs | Prior prefix/copy pattern µs |
| ---: | ---: | ---: | ---: |
| 100 | 64 | 3.39 | 8.27 |
| 1,000 | 64 | 2.11 | 75.89 |
| 10,000 | 64 | 2.02 | 677.54 |
| 100 | 4,096 | 1.58 | 33.48 |
| 1,000 | 4,096 | 1.68 | 328.85 |
| 10,000 | 4,096 | 1.85 | 3,592.70 |

These results support the narrow conclusion that fixed append bookkeeping no
longer scales with retained journal length. They do not establish end-to-end
speedups, allocation counts or total retained bytes. Rig still owns/copies data
for requests and full-run state; context adapters also construct a request view.
Observers were not varied in this microbenchmark. No timing assertions are used
as correctness gates.

## Explicit limits

- One active writer per conversation; no database implementation or fencing.
- Side effects of rejected/unobserved tools remain unknown. Graceful interruption
  preserves accepted results; a hard drop can leave a recoverable journal tail.
- Store error/timeout can follow a successful write. The caller reconciles exact
  commit/receipt payloads and reloads; the framework does not silently retry.
- Rig invalid-tool recovery and retry-feedback paths without canonical callbacks
  fail closed. Full support needs upstream checkpoint extensions.
- Compaction summaries are rebuilt from the journal on restart. Applications
  bound summary output and deduplicate costly compactor effects.
- Standard Anthropic prefix/tail policies are verified on captured streaming
  requests. Arbitrary historical text-block markers are not implemented.
- Async timeouts cannot preempt blocking synchronous application code.
- No live-provider or production-database integration was exercised.
