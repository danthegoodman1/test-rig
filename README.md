# rigloop

A small agent loop harness for [Rig](https://github.com/0xPlaygrounds/rig).

`rigloop` owns the queue around a `rig::agent::Agent`: prompt once, then
steer, follow up, interrupt, resume, inspect state, or abort from a handle.
It keeps Rig in charge of model calls and tools, and adds just enough loop
control for application code.

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Handle Controls](#handle-controls)
- [Timeouts](#timeouts)
- [Turn Hooks](#turn-hooks)
- [Events](#events)
- [History and Resume](#history-and-resume)
- [End Reasons](#end-reasons)
- [Inspiration](#inspiration)

## Install

```toml
[dependencies]
rigloop = "0.1"
rig = "0.38"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Quick Start

```rust
use rig::{
    client::CompletionClient,
    message::Message,
    providers::openai,
};
use rigloop::AgentLoop;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::Client::from_env()?;
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble("You are concise and practical.")
        .build();

    let handle = AgentLoop::new(agent).prompt(Message::user("Explain agent loops in one paragraph."));

    let result = handle.wait().await?;
    println!("end_reason: {:?}", result.end_reason);
    println!("{}", result.last_response.unwrap_or_default());

    Ok(())
}
```

## Handle Controls

`AgentLoop::prompt(...)` returns an `AgentLoopHandle`.

```rust
handle.follow_up(Message::user("Also give me a checklist."))?;
handle.steer(Message::user("Keep the next answer shorter."))?;
handle.interrupt(Message::user("Stop that and answer this instead."))?;
handle.resume()?;
handle.abort()?;

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
| `state` | Returns a clone of the committed in-memory Rig message history. |
| `wait` | Waits for the loop to finish and returns `AgentLoopResult`. |

Dropping `AgentLoopHandle`, or cancelling a pending `wait()`, aborts the
running task. Use `abort()` followed by `wait()` when you want the loop to stop
through the normal commit path and keep completed tool results from the active
turn.

## Timeouts

Set turn and loop wall-clock timeouts independently:

```rust
use std::time::Duration;

let agent_loop = AgentLoop::new(agent)
    .turn_timeout(Duration::from_secs(30))
    .loop_timeout(Duration::from_secs(120));
```

`turn_timeout` applies to each active agent turn and resets between turns.
`loop_timeout` applies to the whole loop lifetime, including idle time while the
handle is still alive. If both can fire during a turn, the earlier deadline wins.

Timeouts during an active turn commit only valid completed tool-call/tool-result
pairs, then run `with_turn_hook` with that append batch, which may be empty.
Active-turn timeouts emit `AgentLoopEvent::TurnTimedOut`. A loop timeout while
idle ends the loop without running the hook.

## Turn Hooks

Use `with_turn_hook` for persistence, compaction, and application-controlled
stops. The hook receives:

- `history`: the candidate full history after appending this turn's messages.
- `new_messages`: only the append batch generated at this turn boundary.

```rust
use rigloop::{AgentLoop, TurnHookAction};

let agent_loop = AgentLoop::new(agent).with_turn_hook(|_app_state, turn| async move {
    println!("persist {} new messages", turn.new_messages.len());

    if turn.history.len() > 100 {
        let summary = rig::message::Message::system("Compacted summary of the previous turns.");
        return Ok::<_, std::convert::Infallible>(TurnHookAction::ReplaceHistory(vec![summary]));
    }

    Ok(TurnHookAction::Continue)
});
```

A hook can return:

| Action | Meaning |
| --- | --- |
| `Continue` | Keep the candidate full history. |
| `ReplaceHistory(messages)` | Replace the in-memory history with this complete history. |
| `Abort { reason }` | Keep the candidate full history, then stop the loop with `EndReason::AbortedByHook`. |

Use `on_assistant_message_finished` when you need to persist an assistant
snapshot before tool results are committed:

```rust
let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(
    |_app_state, context| async move {
        println!("persist assistant snapshot: {} messages", context.new_messages.len());
        Ok::<_, std::convert::Infallible>(())
    },
);
```

For tool-call turns, the hook future is polled while Rig continues tool
execution, then awaited before later commit/error boundaries.
If the snapshot contains tool calls, it is not yet valid provider history until
matching tool results exist. Rigloop repairs loaded histories with unanswered
assistant tool calls by inserting synthetic failed tool results before
validation. Override that synthetic result text with
`with_unanswered_tool_call_repair_message(...)`.

## Events

Subscribe to lifecycle events and raw Rig stream items:

```rust
let mut events = handle.subscribe();

while let Ok(event) = events.recv().await {
    match event {
        rigloop::AgentLoopEvent::Rig(item) => {
            // Full Rig stream granularity, including text deltas and tool deltas.
        }
        rigloop::AgentLoopEvent::AssistantMessageFinished { messages, .. } => {
            // A partial assistant snapshot is available for persistence.
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
such as a failed turn hook or invalid Rig message history shape/order.

## History and Resume

Seed history with `with_history(...)`:

```rust
let loop_from_history = AgentLoop::new(agent)
    .with_history(existing_messages)
    .resume()?;
```

`resume()` validates that the committed Rig message history is usable before
starting another model call. Invalid tool-call ordering is rejected before it
can be sent to a provider.

Histories loaded from partial assistant-message persistence are repaired by
default when an assistant tool call is missing a matching tool result. Rigloop
inserts a failed tool result that says `tool crashed before a result returned`;
other invalid message ordering still fails validation.

## End Reasons

`AgentLoopResult` includes `end_reason`, `history`, and `last_response`.

Known end reasons include:

| Reason | Meaning |
| --- | --- |
| `Idle` | The prompt and all queued work completed. |
| `Aborted` | The caller aborted the loop. |
| `AbortedByHook` | A turn hook stopped the loop. |
| `ContentFilter` | The provider refused or filtered the request/response. |
| `ContextFull` | The provider rejected the request because the context was too large. |
| `Length` | The provider stopped due to an output limit. |
| `MaxTurns` | Rig hit the configured multi-turn tool-call limit. |
| `TurnTimedOut` | The active turn exceeded the configured wall-clock timeout. |
| `LoopTimedOut` | The loop exceeded the configured wall-clock timeout. |
| `ToolError` | Tool execution failed. |
| `ApiError` | The provider/client returned an API error. |

## Inspiration

The API shape is inspired by
[`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent):
stateful agent control, event streaming, steering, follow-ups, and hooks. This
crate keeps that idea intentionally small and Rig-native.
