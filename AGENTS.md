# AGENTS.md

Build this crate with a "Less, but better" bias. Prefer the smallest API and implementation that makes the agent loop correct, understandable, and pleasant to use.

Correctness comes first. Preserve provider message ordering, especially assistant tool-call messages and their matching user tool-result messages. Interrupt, abort, persistence, and compaction paths must never leave in-memory or persisted history in a shape that a provider would reject.

Keep state transitions explicit. Durability hooks should receive only appendable messages, context transforms should be boundary operations, and recovery should stay caller-owned unless the crate grows a clear need for more.

Optimize for DX without hiding behavior. Use predictable names, clear errors, builder methods that compose naturally, and examples that compile. Avoid clever abstractions until repetition or real complexity proves they are needed.

Tests should cover behavior, ordering, and failure paths. Add coverage when touching queue semantics, interruption, persistence, context transforms, stream forwarding, or end reasons.
