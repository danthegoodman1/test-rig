# rigloop

[![crates.io](https://img.shields.io/crates/v/rigloop.svg)](https://crates.io/crates/rigloop)
[![docs.rs](https://docs.rs/rigloop/badge.svg)](https://docs.rs/rigloop)

A small agent loop harness for [Rig](https://github.com/0xPlaygrounds/rig).

`rigloop` has two layers:

- `DurableAgentHarness`: the recommended application API. Start a managed agent
  once, signal it over time, wait for idle, and atomically persist transcript
  deltas with durable inbox ACKs.
- `AgentLoop`: the lower-level execution engine. Prompt, resume, steer,
  follow up, interrupt, abort, subscribe to events, and inspect in-memory Rig
  history directly.

Rig still owns model calls and tools. Rigloop owns loop control, provider-safe
history ordering, interruption, partial commits, durable signal ACKs, and the
small amount of recovery needed around unanswered tool calls.

## Contents

- [Install](#install)
- [Recommended API](#recommended-api)
- [Durable Signals](#durable-signals)
- [Low-Level Engine](#low-level-engine)
- [Timeouts](#timeouts)
- [Events and Checkpoints](#events-and-checkpoints)
- [History and Repair](#history-and-repair)
- [End Reasons](#end-reasons)
- [Inspiration](#inspiration)

## Install

```toml
[dependencies]
rigloop = "0.1"
rig = "0.38"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Recommended API

Use `DurableAgentHarness` when building an app or service that needs a
long-lived agent session:

```rust
use rig::{
    client::CompletionClient,
    providers::openai,
};
use rigloop::{DurableAgentHarness, InMemoryDurableAgentStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::Client::from_env()?;
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble("You are concise and practical.")
        .build();

    // Use your own DurableAgentStore in production. The in-memory store is
    // useful for tests, examples, and local prototypes.
    let store = InMemoryDurableAgentStore::default();
    let harness = DurableAgentHarness::new(agent, store);

    harness.follow_up("Explain agent loops in one paragraph.").await?;
    harness.start()?;

    let end_reason = harness.wait_for_idle().await?;
    println!("end_reason: {end_reason:?}");

    Ok(())
}
```

`max_turns(...)` is optional. By default rigloop uses `DEFAULT_MAX_TURNS`
(`1_000_000`) so long tool loops are not cut off by the harness unless you add
your own guardrail.

Once started, the harness owns the manager task. Later calls to `steer`,
`follow_up`, or `interrupt` wake the manager automatically; callers do not call
`start()` again.

```rust
use rig::message::Message;

harness.steer("Keep the next answer shorter.").await?;
harness.follow_up(Message::user("Also give me a checklist.")).await?;
harness.interrupt("Stop that and answer this instead.").await?;
harness.pause()?;

let end_reason = harness.wait_for_idle().await?;
```

`pause()` is process-local control. It stops at the next safe turn boundary
without appending conversation content. Durable inbox entries that have been
submitted but not started remain unacked/submitted so a later process start can
load and resume them; the current harness stays paused until it is dropped.

The store supplies both durable inbox persistence and completed tool-result
persistence:

```rust
use rigloop::{
    DurableAgentFuture, DurableAgentStore, DurableInboxEntry,
    IncrementalToolResultPersistence, PersistMessagesArgs,
};

impl IncrementalToolResultPersistence for MyStore {
    // persist_tool_result(...)
    // load_tool_result(...)
}

impl DurableAgentStore for MyStore {
    fn submit_inbox_entry(&self, entry: DurableInboxEntry) -> DurableAgentFuture<()> {
        // Insert the submitted durable signal.
    }

    fn persist_messages_and_ack(&self, args: PersistMessagesArgs) -> DurableAgentFuture<()> {
        // Atomically append transcript messages, ACK matching inbox rows, and
        // record args.turn_outcome when the checkpoint is a turn boundary.
    }
}
```

For tests, examples, and local prototypes, `InMemoryDurableAgentStore` keeps
the same append-and-ACK shape without the store boilerplate:

```rust
use rigloop::{DurableAgentHarness, InMemoryDurableAgentStore};

let store = InMemoryDurableAgentStore::default();
let harness = DurableAgentHarness::new(agent, store.clone());

harness.follow_up("start").await?;
harness.start()?;
harness.wait_for_idle().await?;

let snapshot = store.snapshot();
assert!(snapshot.pending_inbox_entries.is_empty());
```

See [examples/durable_inbox_ack.rs](examples/durable_inbox_ack.rs)
for a complete in-memory example.

## Durable Signals

Signal methods create a `DurableInboxEntry`, tag the first user text block with
the generated inbox id, then submit the inbox entry before forwarding it to the
running agent or queueing it for the manager.

The tag lives in the text block's `additional_params` under
`rigloop_inbox_entry_id`. When a persisted transcript delta contains tagged
messages, the harness includes those ids in `PersistMessagesArgs` so the store
can atomically persist the messages and ACK the durable inbox rows in the same
transaction.

When a checkpoint is also a turn boundary, `PersistMessagesArgs::turn_outcome`
contains the turn outcome kind, optional terminal loop reason, and provider
usage. The outcome is part of the same durable transaction. It may be present
even when `messages` is empty because assistant snapshots can be persisted
before the final turn boundary records usage.

By default inbox ids are harness-local monotonic strings like `inbox-1`. Use
`with_inbox_id_generator(...)` to provide ids from your own database or id
service. Signal messages must contain a user text block that can be tagged;
otherwise the signal fails with `DurableAgentError::InvalidSignalMessage`.

History loading remains caller-owned. Load persisted transcript messages in
your application, repair them if needed, and pass them through `with_history`.
If a running turn is interrupted after an assistant tool-call snapshot has been
persisted, the harness repairs the trailing unanswered tool calls through the
tool-result store or the default recovery text before appending the interrupt
message, so the durable transcript remains replayable.

## Low-Level Engine

`AgentLoop` is the escape hatch for custom orchestration. `prompt(...)` returns
an `AgentLoopHandle`.

```rust
use rigloop::AgentLoop;

let handle = AgentLoop::new(agent).prompt(Message::user("Start."));

handle.follow_up(Message::user("Also give me a checklist."))?;
handle.steer(Message::user("Keep the next answer shorter."))?;
handle.interrupt(Message::user("Stop that and answer this instead."))?;
handle.resume()?;
handle.abort()?;
handle.pause()?;

let messages = handle.state();
let result = handle.wait().await?;
```

The queue rules are intentionally simple:

| Method | Behavior |
| --- | --- |
| `follow_up` | Runs after the loop would otherwise become idle. Follow-ups apply one at a time. |
| `steer` | Runs after the current agent turn. All ready steers apply together. |
| `interrupt` | Stops the current turn, keeps only valid completed tool results, then runs the new message next. |
| `resume` | Runs another agent turn using the current history as-is. It is a no-op while a turn is running. |
| `abort` | Stops the loop. Completed tool results from the current turn are kept. |
| `pause` | Stops at the next turn boundary without adding conversation content. Queued work is not drained. |
| `state` | Returns a clone of the committed in-memory Rig message history. |
| `wait` | Waits for the loop to finish and returns `AgentLoopResult`. |

Dropping `AgentLoopHandle`, or cancelling a pending `wait()`, aborts the
running task. Use `abort()` followed by `wait()` when you want the loop to stop
through the normal commit path and keep completed tool results from the active
turn.

## Timeouts

Set turn and loop wall-clock timeouts on either layer:

```rust
use std::time::Duration;

let harness = DurableAgentHarness::new(agent, store)
    .turn_timeout(Duration::from_secs(30))
    .loop_timeout(Duration::from_secs(120));

let agent_loop = AgentLoop::new(agent)
    .turn_timeout(Duration::from_secs(30))
    .loop_timeout(Duration::from_secs(120));
```

`turn_timeout` applies to each active agent turn and resets between turns.
`loop_timeout` applies to the whole loop lifetime, including idle time while the
low-level handle is still alive. If both can fire during a turn, the earlier
deadline wins.

Timeouts during an active turn commit only valid completed tool-call/tool-result
pairs. Active-turn timeouts emit `AgentLoopEvent::TurnTimedOut`; loop timeouts
while idle end the loop without an extra transcript commit.

## Events and Checkpoints

On the managed harness, checkpoint handlers run after rigloop's internal
durability work. Handler errors fail the managed run and are returned from
`wait_for_idle()`.

```rust
let harness = DurableAgentHarness::new(agent, store)
    .with_checkpoint_handler(|checkpoint| async move {
        match checkpoint {
            rigloop::DurableCheckpoint::TurnBoundary(turn) => {
                println!(
                    "committed {} messages, input_tokens={:?}",
                    turn.new_messages.len(),
                    turn.usage.map(|usage| usage.input_tokens),
                );
            }
            rigloop::DurableCheckpoint::AssistantMessageFinished(context) => {
                println!(
                    "assistant snapshot: {} messages, ending={}",
                    context.new_messages.len(),
                    context.end_reason.is_some(),
                );
            }
        }

        Ok::<_, std::convert::Infallible>(
            rigloop::DurableCheckpointAction::Continue,
        )
    });
```

Use `context.end_reason` or `turn.end_reason` to avoid maintenance work that is
only useful before another turn. For compaction, prefer
`DurableCheckpoint::TurnBoundary`; assistant snapshots are earlier recovery
points and can temporarily contain unanswered tool calls.

Use low-level event subscription, or `with_event_observer(...)` on the managed
harness, when you need raw stream forwarding:

```rust
let mut events = handle.subscribe();

while let Ok(event) = events.recv().await {
    match event {
        rigloop::AgentLoopEvent::Rig(item) => {
            // Full Rig stream granularity, including text deltas and tool deltas.
        }
        rigloop::AgentLoopEvent::AssistantMessageFinished { messages, .. } => {
            // A partial assistant snapshot is available.
        }
        rigloop::AgentLoopEvent::TurnCommitted { messages } => {
            // A valid append batch was committed.
        }
        rigloop::AgentLoopEvent::LoopEnded { end_reason } => break,
        rigloop::AgentLoopEvent::LoopFailed { error } => break,
        _ => {}
    }
}
```

`LoopEnded` is emitted for normal loop outcomes, including provider/tool
outcomes represented as `EndReason`s such as `ContextFull` or `ApiError`.
`LoopFailed` is emitted when the runner itself returns an `AgentLoopError`,
such as invalid Rig message history shape/order.

## History and Repair

Seed managed or low-level sessions with `with_history(...)`:

```rust
let harness = DurableAgentHarness::new(agent, store)
    .with_history(existing_messages);

let loop_from_history = AgentLoop::new(agent)
    .with_history(existing_messages)
    .resume()?;
```

`resume()` validates that the committed Rig message history is usable before
starting another model call. Invalid tool-call ordering is rejected before it
can be sent to a provider.

Histories loaded from partial assistant-message persistence are repaired by
default when an assistant tool call is missing a matching tool result. Rigloop
inserts a recovery tool result that says `Recovery message: no result was recorded for this tool call. It may or may not have completed.`;
other invalid message ordering still fails validation.

Use the same repair logic directly when you need to normalize persisted history
before handing it to another boundary:

```rust
use rigloop::{
    repair_unanswered_tool_calls,
    DEFAULT_UNANSWERED_TOOL_CALL_REPAIR_MESSAGE,
};

let repaired = repair_unanswered_tool_calls(
    loaded_messages,
    DEFAULT_UNANSWERED_TOOL_CALL_REPAIR_MESSAGE,
);
```

If you record completed tool results separately, use
`repair_unanswered_tool_calls_with_persistence(...)` so repair can restore those
results before falling back to synthetic recovery text.

## End Reasons

`AgentLoopResult` includes `end_reason`, `history`, and `last_response`.
`DurableAgentHarness::wait_for_idle()` returns the latest `EndReason`.

Known end reasons include:

| Reason | Meaning |
| --- | --- |
| `Idle` | The prompt and all queued work completed. |
| `NoRun` | A managed wait reached idle without starting or resuming an agent turn. |
| `Aborted` | The caller aborted the loop. |
| `Paused` | The caller paused the loop at a safe turn boundary. |
| `AbortedByHook` | A turn hook or durable checkpoint handler stopped the loop after a commit boundary. |
| `ContentFilter` | The provider refused or filtered the request/response. |
| `ContextFull` | The provider rejected the request because the context was too large. |
| `Length` | The provider stopped due to an output limit. |
| `MaxTurns` | Rig hit the configured multi-turn tool-call limit. |
| `TurnTimedOut` | The active turn exceeded the configured wall-clock timeout. |
| `LoopTimedOut` | The loop exceeded the configured wall-clock timeout. |
| `ToolError` | Tool execution failed. |
| `ApiError` | The provider/client returned an API error. |

## Inspiration

The managed API shape is inspired by
[`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent):
stateful agent control, event streaming, steering, follow-ups, durable inboxes,
and hooks. This crate keeps that idea intentionally small and Rig-native.
