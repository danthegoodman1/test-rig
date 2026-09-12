//! A durable, single-owner session around Rig's native agent runner.
//!
//! ```no_run
//! # async fn example(agent: rig::agent::Agent) -> Result<(), rigloop::Error> {
//! use rigloop::{MemoryStore, Session};
//! let session = Session::new(agent, MemoryStore::default()).start()?;
//! session.follow_up("Hello").await?;
//! session.wait_for_idle().await?;
//! session.pause().await?;
//! # Ok(())
//! # }
//! ```
pub mod context;
mod journal;
mod session;
pub mod store;
pub use journal::validate_history;
pub use session::{EndReason, Error, Event, Outcome, Session, SessionHandle};
pub use store::{
    Commit, InboxEntry, InboxStatus, MemoryStore, SignalKind, Snapshot, Store, StoreError,
    StoreFuture, ToolKey,
};
#[cfg(test)]
mod tests;
