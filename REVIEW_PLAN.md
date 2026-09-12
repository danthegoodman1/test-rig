# Rigloop implementation plan

## Overarching goal

Keep one small durable session around Rig 0.42's native runner. Preserve provider
ordering, accepted tool results and atomic transcript/inbox ACKs. Use upstream
context policies and compaction; keep recovery and uncertain-write reconciliation
caller-owned. The chosen implementation is complete within the boundary below.

The review baseline was `29474e14876e1be51a8cfbef4b8deab30c9b6dad` (Rig 0.38.2).
[REVIEW_EVIDENCE.md](REVIEW_EVIDENCE.md) retains its ten failing probes.
[IMPLEMENTATION_EVIDENCE.md](IMPLEMENTATION_EVIDENCE.md) records current tests,
benchmark results and limits. No release or deployment is part of this plan.

## Implementation principles

- One actor owns queues and journal positions; idle never replaces the actor.
- Commit canonical assistant messages from Rig hooks, never streamed deltas.
- Keep the durable journal separate from the next model request's context view.
- Preserve accepted results individually. A rejected/unobserved tool completion
  remains unknown; never automatically replay side effects.
- Use immutable commit identities, expected append positions and explicit ACKs.
- Bound awaited storage; report uncertain writes instead of assuming rollback.
- Use native hooks and memory policies; remove superseded APIs and state machines.

The explicit upstream fallback is to reject invalid-tool recovery or retry
feedback when Rig omits canonical checkpoints. The configured runner remains the
only execution implementation. Summary state is a rebuildable cache; tools run
serially; the store has one active writer per conversation. Arbitrary historical
Anthropic cache markers remain a provider extension, while standard cache
controls are preserved and tested.

## Testing strategy

Exercise real Rig streaming with mock models and tools, capture Anthropic HTTP
requests, and inject failures before/after durable writes. Use paused Tokio time
for deadlines and barriers for cancellation. Map all ten review probes to the
new API, including deliberate rejection of malformed stored history. Run all
library/example targets, documentation, formatting and strict Clippy. Measure
append bookkeeping separately from provider IO and native request construction.

## Phase 0: Choose the upstream boundary

Goal: Replace custom execution/history machinery with a proven Rig boundary.

Scope: Align Rig crates, compose the final durable hook with configured hooks,
verify request-only policies/cache settings, and choose a bounded fallback for
unsupported callbacks. The deletion map is the removed `api.rs`, `engine.rs`,
`history.rs`, `durable.rs`, replaced by `session.rs`, `store.rs`, `journal.rs` and
small context adapters.

Completion gate: One compiled execution route with explicit retained contracts.

Testing plan: Canonical chunks, metadata, partial results, hook ordering, context
projection, captured cache bodies and a coherent dependency graph.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Decision | 0A: Guarantee and ownership table | README defines journal/request validity, ACK semantics, accepted results, deadlines and single-writer ownership. |
| Complete | Work | 0B: Rig execution integration spike | Native runner integration in `src/session.rs`; grouped calls, interrupted siblings, rewritten results, rejected attempts and unsupported-recovery tests in `src/tests.rs`. |
| Complete | Work | 0C: Memory version/policy spike | Cargo.lock resolves Rig/core/agent/memory 0.42.0; policy and compactor integration tests pass. |
| Complete | Doc | 0D: Deletion map and public API | Removed four custom runtime modules; README documents the session and store APIs. |
| Complete | Test | 0T: Integration scenarios | `anthropic_cache_controls_reach_the_session_wire_request`, canonical/identity tests and `cargo tree -i rig-core`. |
| Complete | Gate | 0G: Select one route | Configured AgentRunner + last durability hook; unsupported recovered turns fail before dispatch/next request. No hand-driven alternate engine. |

## Phase 1: Canonical journal commits and recovery

Goal: Resolve F1–F3 without content-prefix reconciliation.

Scope: Checkpoint complete assistant blocks, persist accepted results by assistant
position/call ID, append result batches in call order and repair only unanswered
journal tails. Reject malformed partial batches before provider IO.

Completion gate: No duplicated assistant/result representations; errors and
interruption retain accepted results and provider metadata.

Testing plan: E1–E4/E8/E9 equivalents, hooks before tool dispatch, receipts on
restart and exact transcript comparisons with Rig's subsequent request.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 1A: Canonical assistant boundary | `canonical_chunks_metadata_and_no_duplicate_commits`, `reasoning_deltas_use_rigs_canonical_assembly`; no PartialTurn assembler. |
| Complete | Work | 1B: Append-only commit bookkeeping | `Actor::commit`, `validate_append`; UUID/position commits with no full-prefix reconciliation. |
| Complete | Work | 1C: One identity-preserving repair path | `repair_tail`, strict identity validation; grouped/provider-ID and corrupted-result tests. |
| Complete | Decision | 1D: Partial stored batch normalization | Rejected on start; caller-owned atomic normalization documented in README. Tail receipt recovery tested across restart. |
| Complete | Work | 1E: Error and interruption preservation | Provider failure, sibling stop, abort and interrupt tests retain real earlier results and close missing calls explicitly. |
| Complete | Test | 1T: Transcript/recovery regression matrix | Review probe mapping in IMPLEMENTATION_EVIDENCE; assistant write failure prevents tools from executing. |
| Complete | Gate | 1G: Canonical provider-safe boundaries | Startup validation, dispatch guard and append validation; accepted-result and recovery tests pass. |

## Phase 2: One lifecycle owner

Goal: Resolve F4 and remove duplicate manager/runner queues.

Scope: One long-lived actor; bounded intake; explicit steering, follow-up,
interrupt, resume, pause, abort and idle-barrier behavior. Configuration is consumed
at start; handles carry commands and observations only.

Completion gate: Normal idle transitions never close intake or poison the session.

Testing plan: Repeated multi-thread idle handoffs, concurrent waiters, batching,
priority, pause/abort boundaries, resume and handle cancellation.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 2A: Single queue/lifecycle owner | `Actor` and `Queue` in session.rs; removed managed runner handoff. |
| Complete | Work | 2B: Accepted-work handoff safety | 60 multi-thread idle transitions with concurrent waiters; accepted submit survives caller cancellation. |
| Complete | Work | 2C: Idle, pause and wait semantics | Queue priority/batching, pause latch, abort, resume and idle-loop deadline tests. |
| Complete | Work | 2D: Builder/session separation | `Session::start(self)` returns `SessionHandle`; invalid resume returns a local error without terminating the actor. |
| Complete | Test | 2T: Scheduling and drop matrix | Single-thread control tests, multi-thread idle race, hard-drop restart, bounded queue and compaction-intake tests. |
| Complete | Gate | 2G: No work lost at idle | Inbox entries remain pending on pause/abort; continuing the same session after idle is tested. |

## Phase 3: Durable identity and replay

Goal: Resolve F5 and make uncertain writes reconcilable.

Scope: UUID/caller IDs; idempotent submit and commits; caller-loaded pending
entries; internal ACK identity; stable receipt positions and persistable timestamps.

Completion gate: Restart preserves entry identity/order and cannot confuse a new
signal with an old ACK. Identical uncertain writes are safe to retry.

Testing plan: Duplicate/conflicting IDs, replay without resubmit, caller-canceled
submission, write-before-error, hung writes, stale positions and duplicate ACKs.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 3A: Stable scoped IDs | UUID defaults; same-ID conflict tests; repeated provider IDs use distinct assistant positions. |
| Complete | Work | 3B: Existing-entry replay | `Session::pending`; drop/restart test and runnable durable_inbox_ack example. |
| Complete | Work | 3C: Internal identity envelope | InboxEntry travels through the actor; Commit ACK IDs are never derived from provider text. |
| Complete | Work | 3D: Idempotent commit/store contract | Store rustdoc; exact uncertain Commit/ToolResult payloads returned; before/after-write retry tests. |
| Complete | Work | 3E: Reference store/timestamps | Indexed identity checks, atomic mutation after validation, SystemTime/serde records; stale/conflicting/double-ACK tests. |
| Complete | Test | 3T: Intake/restart failure matrix | Cancellation-after-insert, partial receipt restart, same-ID replay and unknown-write cases. |
| Complete | Gate | 3G: Durable intake survives restart | Executable restart example and regression tests; caller owns loading, normalization and reconciliation. |

## Phase 4: Bounded waits and truthful observations

Goal: Resolve F6/F7 without adding callback supervisors.

Scope: Execution deadlines plus a bounded finalization allowance; bounded storage
reads/writes; explicit uncertain errors; one broadcast observation path with
visible lag; native structured finish reasons and optional usage.

Completion gate: Configured deadlines cover async hooks/compaction and storage;
unknown writes/usage are never reported as known success/zero.

Testing plan: Paused-clock hook and storage failures, startup reads, idle timeout,
observer overflow, missing usage and durable provider-error outcomes.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Decision | 4A: Deadline/finalization contract | README and Session rustdoc specify execution/IO bounds, cooperative futures and finalization allowance. |
| Complete | Work | 4B: Cancellation and uncertain writes | Paused-clock hung hook, commit, receipt and startup-read tests; exact retry payloads. |
| Complete | Work | 4C: Ordered bounded observations | One actor emits broadcasts; receiver lag remains explicit; no user observer callback in the commit path. |
| Complete | Work | 4D: Usage and outcomes | Missing usage stays None; native CompletionCall finish reasons; provider failure preserves transcript and persists outcome; RigError keeps structured errors. |
| Complete | Test | 4T: Timeout/observation matrix | Deadline, error-injection, subscriber overflow and multi-waiter tests. |
| Complete | Gate | 4G: Bounded truthful control surface | Final standard checks pass; failure/uncertainty semantics documented and tested. |

## Phase 5: Upstream context and scaling

Goal: Resolve F8 through request-only context and delta-based bookkeeping.

Scope: Direct fallible MemoryPolicy adapter and native CompactingMemory;
application Compactor prompts; retained preamble; bounded intake; snapshots only
on request; no per-checkpoint journal-prefix scans/copies.

Completion gate: Context shrinks within the same session/tool loop without
rewriting the journal or dropping accepted work. Bookkeeping stays independent
of retained transcript length for a fixed append.

Testing plan: Intra-tool policies, policy errors, custom compactor/restart,
queued work during compaction, captured Anthropic cache bodies and a release
benchmark at 100/1,000/10,000 messages with 64/4,096-byte bodies.

Status ledger:

| Status | Type | Item | Evidence / Gap |
| --- | --- | --- | --- |
| Complete | Work | 5A: Distinct context view and policy boundary | ContextPolicy and CompactingContext use RequestPatch::history; first/intra-loop request and unchanged-journal tests. |
| Complete | Work | 5B: Upstream policy integration | Custom Compactor example/test, restart rebuild, preamble preservation, policy failure and Anthropic cache wire fixtures. |
| Complete | Work | 5C: Delta bookkeeping/bounded intake | Position-based commits, indexed store IDs, overload tests; no full-prefix checkpointer. |
| Complete | Doc | 5D: Session compaction/restart examples | Both examples execute successfully; library doctest compiles; README explains context budgets and caller-owned restart. |
| Complete | Test | 5T: Selected context/scaling matrix | 36 behavioral tests and isolated release bookkeeping probe; measurements and limits in IMPLEMENTATION_EVIDENCE. |
| Complete | Gate | 5G: Smaller implementation with bounded context | About 63% fewer production source lines; fixed-append measurements remain near-flat as history grows. |

## Deliberate non-goals and remaining upstream limits

No database driver, automatic journal loading, side-effect retries, durable summary
state machine, concurrent tools or multi-writer fencing.
A larger allocation/retention and end-to-end provider benchmark remains optional
profiling work; the measured claim is limited to append bookkeeping. Native Rig
request/history ownership still incurs its own costs. Invalid-tool recovery,
retry feedback without canonical callbacks and arbitrary historical text-cache
markers require an upstream extension before this session can support them.
