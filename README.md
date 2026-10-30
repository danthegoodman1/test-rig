# rigloop

A small durable session around [Rig](https://github.com/0xPlaygrounds/rig).

Rig owns model calls, streaming, canonical assistant messages and tool execution.
Rigloop owns one session queue, durable inbox ACKs, append-only checkpoints and
recovery of accepted tool results. Context windows use Rig's memory crate. The
runtime and memory integrations use Rig 0.45.

## A session

```rust
use rig::agent::Agent;
use rigloop::{EndReason, MemoryStore, Session};

async fn run(agent: Agent) -> Result<(), rigloop::Error> {
    let store = MemoryStore::default();
    let session = Session::new(agent, store.clone()).start()?;

    session.follow_up("Explain the design").await?;
    assert_eq!(session.wait_for_idle().await?, EndReason::Idle);

    // Idle leaves the same session ready for more work.
    session.follow_up("Give me an example").await?;
    session.wait_for_idle().await?;
    session.pause().await?;
    session.stopped().await?;
    Ok(())
}
```

Use a local/path dependency for this checkout, alongside `rig = "0.45"` and
Tokio. `MemoryStore` is the reference implementation for examples
and tests. Implement `Store` over a database for persistence across processes.

`Session` is a builder; `start()` consumes it and returns a cloneable
`SessionHandle`. There is one actor and one queue owner. Dropping the last
handle cancels the task. Use `abort()` or `pause()` followed by `stopped()` for
graceful shutdown.

| Operation | Behavior |
| --- | --- |
| `follow_up(message)` | Queue one prompt after ready steers and interrupts. |
| `steer(message)` | Apply all ready steers together after the active turn. |
| `interrupt(message)` | Persist the signal, cancel the active turn, close its tool batch, then run the interrupt next. Live interrupts have newest-first priority. |
| `resume()` | Run from the last conversation message without appending it again. Coalesces queued resumes; a no-op during an active turn. |
| `wait_for_idle()` | Wait until accepted work drains or the session stops. Multiple waiters are supported. Work accepted later can extend the wait. |
| `pause()` | Finish the active turn, then stop. Queued durable entries remain pending. |
| `abort()` | Cancel the active turn, preserve accepted results, then stop. |
| `history()` | Request a snapshot while the actor is alive. May include a recoverable tool-call tail. |
| `subscribe()` | Receive ordered observations; receiver lag is explicit. |

Only nonempty user messages without tool-result blocks may be submitted as
signals. Multimodal content is supported. Queue capacity defaults to 256;
`queue_capacity` bounds pending work, command intake and idle waiters. Commands
backpressure at intake, and pending-work overload returns `Error::QueueFull`
before submission to storage. Controls are applied between bounded storage
operations. Tools execute serially through Rig's default runner.

## Durability and recovery

A `Store` belongs to **one conversation with one active writer**. Its four
operations are `submit`, `commit`, `save_tool_result` and `load_tool_result`.
Loading the journal and pending inbox remains application-owned.

A commit contains a UUID, expected journal position, appendable messages, inbox
IDs to ACK, and an optional turn outcome. The store must atomically append,
ACK and record the outcome. ACK means the signal is incorporated into the
journal, not that a model completed the user's request. Signal IDs stay outside
provider messages; default IDs are UUIDs, and applications may supply their own
`InboxEntry.id`.

Assistant messages are checkpointed from Rig's canonical model-turn hook before
tools execute. Each tool outcome is persisted as a receipt before Rig advances.
Once Rig commits a batch, the session appends it as one user message in
model-call order, exactly as Rig holds it, including hook skips and Rig's
feedback for malformed arguments. Before every model request the journal must
match Rig's transcript, and each finished run must match Rig's final transcript.
Interruption fills missing results with an explicit unknown-completion error
result. Recovery preserves call IDs and provider items; it never silently drops
unrelated results.

The journal may end with an unanswered assistant batch after a crash or hard
cancellation. A fresh session restores saved receipts and appends the missing
result batch before the next request. Every other tool batch must be answered by
the next message: one user message holding exactly its results and nothing
else. A pre-existing **partial or split result batch**, orphan, duplicate or
empty result is rejected. Normalize such stored history atomically in the
application before starting.

```rust
async fn restart(agent: rig::agent::Agent, store: rigloop::MemoryStore) -> Result<(), rigloop::Error> {
use rigloop::Session;
let saved = store.snapshot(); // Load from your database in production.
let session = Session::new(agent, store)
    .pending(saved.pending()) // Already stored entries; no resubmit or retag.
    .history(saved.history)
    .start()?;
session.wait_for_idle().await?;
Ok(())
}
```

`submit(entry)` is idempotent for the same ID and contents. A conflicting reuse
must fail. Cancellation after command enqueue does not cancel the actor's
submission. Retain the same entry for retries, or reload pending entries from
the store; calling `follow_up` again creates a different signal.

A storage error or timeout can follow a successful durable write.
`Error::CommitUncertain` contains the exact commit to reconcile/retry;
`Error::ToolWriteUncertain` contains the receipt key and payload. Stop and
resolve these against the store, then load a fresh session. The reference store
checks idempotency before position and rejects conflicting retries. Rigloop
never claims exactly-once external tool side effects and never automatically
re-executes missing tools.

## Context and compaction

Use `context::ContextPolicy(recent)` as an agent hook with a Rig memory window
such as `LastMessages(20)` or `TokenBudget::new(8_000)`, or
`context::CompactingContext::new(recent, compactor)` for rolling compaction of
everything before the window. Both run before every model request, including
tool rounds. A window never starts on a result whose call it cut off. Errors
stop the run instead of silently restoring oversized context.

```rust,ignore
use rig_memory::recent::LastMessages;
use rigloop::context::ContextPolicy;

let agent = rig::agent::AgentBuilder::new(model)
    .preamble("Persistent application instructions")
    .add_hook(ContextPolicy(LastMessages(20)))
    .build();
```

The durable journal remains complete. The current prompt must survive unchanged.
Put persistent application instructions in the agent's preamble; system messages
in history are subject to the chosen window. Budget the preamble, tools, summary
and output headroom separately from the retained window.

Implement `context::Compactor` to choose the summarization prompt, model and
summary message; Rig's `rig_memory::Summarizer` can build the request.
[The compaction example](examples/durable_compaction.rs) demonstrates a custom
prompt with a separate summarizer and output limit. The summary is a
process-local derived cache, rebuilt from the full journal on restart. A
side-effectful compactor should deduplicate by conversation and input content.
Use one `CompactingContext` instance per conversation; do not share an agent
carrying that hook between conversations.

Configure Anthropic caching on the Rig model before building the agent. The
session preserves `Messages::with_prompt_caching`,
`with_static_prefix_cache_ttl` and a request cache retention
(`GenerationOptions::cache`), which hands the conversation-tail marker to
Anthropic's automatic caching. Tests capture the actual streaming request and
verify a one-hour tools/system prefix with either a manual or automatic
conversation-tail marker. Arbitrary historical text-block breakpoints still
require upstream provider support or a provider adapter.

## Deadlines, hooks and observations

`turn_timeout` covers the active model/tool/hook work, including compaction.
`loop_timeout` covers the session lifetime, including startup recovery and idle.
Neither is enabled by default. `io_timeout` defaults to 30 seconds and bounds
storage operations. During active work, the earlier execution/storage deadline
wins. Graceful cancellation gets one additional `io_timeout` allowance to repair
and finalize. An unfinished write is reported as uncertain; it is not silently
retried or reported as successful. Async deadlines require cooperative futures;
blocking synchronous hook/tool code must be offloaded by the application.

Configured Rig hooks run before the final durability hook. Result rewrites are
persisted in their accepted presentation. Ordinary hooks, skips, rewrites,
repeat retries and Rig's malformed-argument feedback compose with the session.

Rig 0.45 settles a turn repaired by invalid-tool-name recovery without its
model-turn hooks, and retry feedback and run-start prompt rewrites change the
transcript without a canonical checkpoint. Those paths are rejected before
dispatch or the next model request; they need an upstream checkpoint extension.
The session disables Rig's conversation memory, which saves once per finished
run. Use context hooks and the explicit `Store` transaction contract here.

`Event::Rig` forwards Rig stream items, and `Event::RigError` preserves Rig's
structured `PromptError`. These are observations; use `Event::Committed` as the
durable boundary. Slow consumers cannot stall the session; `recv()` reports
`Lagged` when the 1,024-event buffer overflows. Turn outcomes record structured
finish reasons where Rig supplies them. Usage is `None` when metrics are missing
or a provider call fails; reported aggregates cover completed calls only.

## Validation

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --doc
cargo run --example durable_inbox_ack
cargo run --example durable_compaction
cargo test --release --lib append_bookkeeping_scaling -- --ignored --nocapture
```

[REVIEW_PLAN.md](REVIEW_PLAN.md) tracks the implementation and retained upstream
limits. [IMPLEMENTATION_EVIDENCE.md](IMPLEMENTATION_EVIDENCE.md) records validation
and the deliberately narrow bookkeeping benchmark.
