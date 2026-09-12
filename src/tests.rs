use super::*;
use rig::{
    agent::{
        Agent, AgentBuilder, AgentHook,
        hook::{
            HookContext, ModelTurnAction, ModelTurnFinished, ToolCall, ToolCallAction,
            ToolResultAction, ToolResultEvent,
        },
    },
    message::{AssistantContent, Message, ToolResult, ToolResultContent, UserContent},
    test_utils::{MockAddTool, MockCompletionModel, MockStreamEvent as E},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::Notify;

fn model(turns: usize) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns((0..turns).map(|i| {
        vec![
            E::text(format!("answer {i}")),
            E::final_response_with_default_usage(),
        ]
    }))
}
fn agent(model: &MockCompletionModel) -> Agent {
    AgentBuilder::new(model.clone()).build()
}
fn tool_turn() -> Vec<E> {
    vec![
        E::tool_call("fc_a", "add", serde_json::json!({"x":1,"y":2})).with_call_id("call_a"),
        E::tool_call("fc_b", "add", serde_json::json!({"x":3,"y":4})).with_call_id("call_b"),
        E::final_response_with_default_usage(),
    ]
}
fn results(history: &[Message]) -> Vec<&ToolResult> {
    history
        .iter()
        .flat_map(|m| match m {
            Message::User { content } => content.as_slice(),
            _ => &[],
        })
        .filter_map(|c| match c {
            UserContent::ToolResult(r) => Some(r),
            _ => None,
        })
        .collect()
}
fn text(message: &Message) -> String {
    match message {
        Message::User { content } => content
            .iter()
            .filter_map(|c| match c {
                UserContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect(),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect(),
        Message::System { content } => content.clone(),
    }
}
async fn idle(session: &SessionHandle) -> EndReason {
    tokio::time::timeout(Duration::from_secs(3), session.wait_for_idle())
        .await
        .expect("session hung")
        .unwrap()
}

#[tokio::test]
async fn canonical_chunks_metadata_and_no_duplicate_commits() {
    let model = MockCompletionModel::from_stream_turns([vec![
        E::message_id("msg-1"),
        E::text("hello "),
        E::text("world"),
        E::final_response_with_default_usage(),
    ]]);
    let store = MemoryStore::default();
    let session = Session::new(agent(&model), store.clone()).start().unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    let snapshot = store.snapshot();
    assert_eq!(snapshot.history.len(), 2);
    let Message::Assistant { id, content } = &snapshot.history[1] else {
        panic!()
    };
    assert_eq!(id.as_deref(), Some("msg-1"));
    assert_eq!(content.len(), 1);
    assert_eq!(text(&snapshot.history[1]), "hello world");
    assert_eq!(snapshot.acked_ids.len(), 1);
    assert!(snapshot.pending().is_empty());
    assert_eq!(session.history().await.unwrap(), snapshot.history);
}

#[tokio::test]
async fn grouped_calls_results_and_provider_identity_round_trip() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("done"), E::final_response_with_default_usage()],
    ]);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone()).tool(MockAddTool).build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    let snapshot = store.snapshot();
    assert_eq!(snapshot.history.len(), 4);
    assert_eq!(results(&snapshot.history).len(), 2);
    assert_eq!(snapshot.tool_results.len(), 2);
    for (call, result) in crate::journal::calls(&snapshot.history[1])
        .iter()
        .zip(results(&snapshot.history))
    {
        assert_eq!(call.id, result.call);
        assert_eq!(call.provider, result.provider);
        assert_eq!(result.name, "add");
    }
    assert_eq!(model.requests()[1].chat_history, snapshot.history[..3]);
    validate_history(&snapshot.history).unwrap();
}

#[derive(Clone)]
struct Gate {
    reached: Arc<Notify>,
    release: Arc<Notify>,
    stop: bool,
}
impl AgentHook for Gate {
    async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        if event.tool_call_id == Some("call_b") {
            self.reached.notify_one();
            if self.stop {
                return ToolCallAction::Stop("blocked B".into());
            }
            self.release.notified().await;
        }
        ToolCallAction::Run
    }
}
fn gate() -> Gate {
    Gate {
        reached: Arc::new(Notify::new()),
        release: Arc::new(Notify::new()),
        stop: false,
    }
}

#[tokio::test]
async fn completed_result_survives_sibling_hook_stop() {
    let model = MockCompletionModel::from_stream_turns([tool_turn()]);
    let store = MemoryStore::default();
    let mut hook = gate();
    hook.stop = true;
    let session = Session::new(
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(hook)
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(
        idle(&session).await,
        EndReason::HookStopped("blocked B".into())
    );
    let snapshot = store.snapshot();
    let results = results(&snapshot.history);
    assert_eq!(results.len(), 2);
    assert!(!format!("{:?}", results[0].content).contains("Recovery"));
    assert!(format!("{:?}", results[1].content).contains("Recovery"));
    assert_eq!(snapshot.tool_results.len(), 1);
    validate_history(&snapshot.history).unwrap();
}

#[tokio::test]
async fn interrupt_preserves_first_result_and_runs_interrupt_before_follow_up() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("interrupt"), E::final_response_with_default_usage()],
        vec![E::text("follow"), E::final_response_with_default_usage()],
    ]);
    let hook = gate();
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    hook.reached.notified().await;
    assert_eq!(store.snapshot().tool_results.len(), 1);
    session.follow_up("later").await.unwrap();
    session.interrupt("now").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    let history = store.snapshot().history;
    validate_history(&history).unwrap();
    assert_eq!(results(&history).len(), 2);
    assert_eq!(text(&history[3]), "now");
    assert_eq!(text(&history[5]), "later");
    assert_eq!(model.request_count(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accepting_work_after_idle_never_closes_the_session() {
    let model = model(60);
    let store = MemoryStore::default();
    let session = Session::new(agent(&model), store.clone()).start().unwrap();
    for i in 0..60 {
        session.follow_up(format!("work {i}")).await.unwrap();
        let (a, b) = tokio::join!(session.wait_for_idle(), session.wait_for_idle());
        assert_eq!(a.unwrap(), EndReason::Idle);
        assert_eq!(b.unwrap(), EndReason::Idle);
    }
    assert_eq!(store.snapshot().history.len(), 120);
    assert_eq!(store.snapshot().acked_ids.len(), 60);
}

#[tokio::test]
async fn provider_failure_keeps_prior_completed_round_trip() {
    let model =
        MockCompletionModel::from_stream_turns([tool_turn(), vec![E::error("provider failed")]]);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model).tool(MockAddTool).build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert!(matches!(idle(&session).await, EndReason::ProviderError(_)));
    let snapshot = store.snapshot();
    assert_eq!(results(&snapshot.history).len(), 2);
    assert!(snapshot.commits.values().any(|c| {
        c.outcome
            .as_ref()
            .is_some_and(|o| matches!(o.reason, EndReason::ProviderError(_)))
    }));
    validate_history(&snapshot.history).unwrap();
}

#[tokio::test]
async fn drop_and_restart_restores_receipt_and_pending_identity() {
    let model = MockCompletionModel::from_stream_turns([tool_turn()]);
    let hook = gate();
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    let old_id = session.follow_up("start").await.unwrap();
    hook.reached.notified().await;
    let pending_id = session.follow_up("queued").await.unwrap();
    drop(session);
    tokio::task::yield_now().await;
    let saved = store.snapshot();
    assert_eq!(saved.pending()[0].id, pending_id);
    assert_eq!(saved.history.len(), 2);
    let resumed_model = model_for_resume();
    let resumed = Session::new(agent(&resumed_model), store.clone())
        .pending(saved.pending())
        .history(saved.history)
        .start()
        .unwrap();
    assert_eq!(idle(&resumed).await, EndReason::Idle);
    let saved = store.snapshot();
    assert_eq!(saved.acked_ids, vec![old_id, pending_id]);
    assert_eq!(results(&saved.history).len(), 2);
    assert!(!format!("{:?}", results(&saved.history)[0]).contains("Recovery"));
    validate_history(&saved.history).unwrap();
    let new_id = resumed.follow_up("fresh ID").await.unwrap();
    assert!(!saved.acked_ids.contains(&new_id));
    assert_eq!(idle(&resumed).await, EndReason::Idle);
}
fn model_for_resume() -> MockCompletionModel {
    model(2)
}

#[tokio::test]
async fn incomplete_or_corrupt_stored_result_batch_is_rejected_without_model_io() {
    let call_a = rig::message::ToolCall::from_wire(
        "a",
        rig::message::ToolFunction {
            name: "add".into(),
            arguments: serde_json::json!({}),
        },
    );
    let call_b = rig::message::ToolCall::from_wire(
        "b",
        rig::message::ToolFunction {
            name: "add".into(),
            arguments: serde_json::json!({}),
        },
    );
    let assistant = Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::ToolCall(call_a),
            AssistantContent::ToolCall(call_b),
        ],
    };
    let partial = Message::User {
        content: vec![UserContent::tool_result_from_wire(
            "a",
            "add",
            vec![ToolResultContent::text("ok")],
        )],
    };
    let model = model(1);
    assert!(
        Session::new(agent(&model), MemoryStore::default())
            .history(vec![assistant, partial])
            .start()
            .is_err()
    );
    assert_eq!(model.request_count(), 0);
}

#[tokio::test]
async fn submissions_are_idempotent_and_conflicting_ids_fail() {
    let model = model(1);
    let store = MemoryStore::default();
    let session = Session::new(agent(&model), store.clone()).start().unwrap();
    let entry = InboxEntry::new(SignalKind::FollowUp, "same");
    session.submit(entry.clone()).await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert_eq!(
        session.submit(entry.clone()).await.unwrap(),
        InboxStatus::Acked
    );
    let mut conflicting = entry;
    conflicting.message = Message::user("different");
    assert!(session.submit(conflicting).await.is_err());
    assert_eq!(model.request_count(), 1);
    assert_eq!(idle(&session).await, EndReason::Idle);
}

#[derive(Clone)]
struct HangingHook(Arc<Notify>);
impl AgentHook for HangingHook {
    async fn on_model_turn_finished(
        &self,
        _: &HookContext,
        _: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.0.notify_one();
        std::future::pending().await
    }
}
#[tokio::test(start_paused = true)]
async fn turn_deadline_bounds_a_hung_user_hook() {
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model(1))
            .add_hook(HangingHook(Arc::new(Notify::new())))
            .build(),
        store.clone(),
    )
    .turn_timeout(Duration::from_millis(20))
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::TurnTimedOut);
    assert_eq!(store.snapshot().history.len(), 1);
}

#[tokio::test]
async fn context_policy_runs_inside_tool_loop_without_rewriting_journal() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("done"), E::final_response_with_default_usage()],
    ]);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .add_hook(context::ContextPolicy(
                rig_memory::SlidingWindowMemory::last_messages(2),
            ))
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("old prompt").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert_eq!(model.requests()[1].chat_history.len(), 2);
    assert_eq!(store.snapshot().history.len(), 4);
    validate_history(&store.snapshot().history).unwrap();
}

struct BadPolicy;
impl rig_memory::MemoryPolicy for BadPolicy {
    fn apply(&self, _: Vec<Message>) -> Result<Vec<Message>, rig_memory::MemoryError> {
        Err(rig_memory::MemoryError::Internal("bad policy".into()))
    }
}
#[tokio::test]
async fn policy_error_stops_before_provider_request() {
    let model = model(1);
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .add_hook(context::ContextPolicy(BadPolicy))
            .build(),
        MemoryStore::default(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert!(matches!(idle(&session).await, EndReason::HookStopped(_)));
    assert_eq!(model.request_count(), 0);
}

#[tokio::test]
async fn missing_usage_stays_unknown() {
    let model = MockCompletionModel::from_stream_turns([vec![
        E::text("ok"),
        E::final_response(rig::completion::Usage::new()),
    ]]);
    let store = MemoryStore::default();
    let session = Session::new(agent(&model), store.clone()).start().unwrap();
    session.follow_up("start").await.unwrap();
    idle(&session).await;
    let snapshot = store.snapshot();
    let outcome = snapshot
        .commits
        .values()
        .find_map(|c| c.outcome.as_ref())
        .unwrap();
    assert_eq!(outcome.usage, None);
}

#[derive(Clone)]
struct FailStore {
    inner: MemoryStore,
    calls: Arc<AtomicUsize>,
    mode: usize,
}
impl Store for FailStore {
    fn submit(&self, entry: InboxEntry) -> StoreFuture<InboxStatus> {
        self.inner.submit(entry)
    }
    fn commit(&self, commit: Commit) -> StoreFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            // Fail the assistant checkpoint, after the prompt's ACK succeeded.
            if store.calls.fetch_add(1, Ordering::SeqCst) == 1 {
                if store.mode == 1 {
                    store.inner.commit(commit).await?;
                }
                if store.mode == 2 {
                    return std::future::pending().await;
                }
                return Err("lost acknowledgement".into());
            }
            store.inner.commit(commit).await
        })
    }
    fn save_tool_result(&self, key: ToolKey, result: ToolResult) -> StoreFuture<()> {
        self.inner.save_tool_result(key, result)
    }
    fn load_tool_result(&self, key: ToolKey) -> StoreFuture<Option<ToolResult>> {
        self.inner.load_tool_result(key)
    }
}
#[tokio::test(start_paused = true)]
async fn uncertain_commit_is_retryable_before_or_after_write() {
    for mode in [0, 1, 2] {
        let store = FailStore {
            inner: MemoryStore::default(),
            calls: Arc::new(AtomicUsize::new(0)),
            mode,
        };
        let session = Session::new(agent(&model(1)), store.clone())
            .io_timeout(Duration::from_millis(20))
            .start()
            .unwrap();
        session.follow_up("start").await.unwrap();
        let error = tokio::time::timeout(Duration::from_secs(2), session.wait_for_idle())
            .await
            .unwrap()
            .unwrap_err();
        let Error::CommitUncertain { commit, .. } = error else {
            panic!("{error:?}")
        };
        store.inner.commit((*commit).clone()).await.unwrap();
        store.inner.commit((*commit).clone()).await.unwrap();
        assert_eq!(store.inner.snapshot().history.len(), 2);
        assert_eq!(store.inner.snapshot().acked_ids.len(), 1);
        let mut conflict = (*commit).clone();
        conflict.messages = vec![Message::assistant("wrong")];
        assert!(store.inner.commit(conflict).await.is_err());
    }
}

#[tokio::test]
async fn observer_lag_is_explicit_and_never_stalls_commits() {
    let mut chunks: Vec<_> = (0..2000).map(|_| E::text("x")).collect();
    chunks.push(E::final_response_with_default_usage());
    let model = MockCompletionModel::from_stream_turns([chunks]);
    let session = Session::new(agent(&model), MemoryStore::default())
        .start()
        .unwrap();
    let mut events = session.subscribe();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert!(matches!(
        events.recv().await,
        Err(tokio::sync::broadcast::error::RecvError::Lagged(_))
    ));
}

#[tokio::test]
async fn pause_latches_at_boundary_and_leaves_pending_inbox() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("done"), E::final_response_with_default_usage()],
    ]);
    let hook = gate();
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    hook.reached.notified().await;
    let next = session.follow_up("pending").await.unwrap();
    session.pause().await.unwrap();
    hook.release.notify_one();
    assert_eq!(idle(&session).await, EndReason::Paused);
    assert_eq!(store.snapshot().pending()[0].id, next);
    validate_history(&store.snapshot().history).unwrap();
}

struct RewriteResults;
impl AgentHook for RewriteResults {
    async fn on_tool_result(&self, _: &HookContext, _: ToolResultEvent<'_>) -> ToolResultAction {
        ToolResultAction::rewrite("redacted")
    }
}
#[tokio::test]
async fn durable_receipts_use_accepted_rewritten_presentation() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("done"), E::final_response_with_default_usage()],
    ]);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .add_hook(RewriteResults)
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    let snapshot = store.snapshot();
    for result in results(&snapshot.history) {
        assert_eq!(result.content, vec![ToolResultContent::text("redacted")]);
    }
    assert_eq!(model.requests()[1].chat_history, snapshot.history[..3]);
}
struct RetryOnce;
impl AgentHook for RetryOnce {
    async fn on_model_turn_finished(
        &self,
        _: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event.turn == 1 {
            ModelTurnAction::repeat()
        } else {
            ModelTurnAction::Continue
        }
    }
}
#[tokio::test]
async fn rejected_model_attempt_is_not_persisted() {
    let model = model(2);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone()).add_hook(RetryOnce).build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert_eq!(
        store.snapshot().history,
        vec![Message::user("start"), Message::assistant("answer 1")]
    );
    assert_eq!(model.request_count(), 2);
}

#[tokio::test]
async fn queue_capacity_rejects_work_before_persistence_and_steers_batch() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("done"), E::final_response_with_default_usage()],
        vec![E::text("steered"), E::final_response_with_default_usage()],
    ]);
    let hook = gate();
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build(),
        store.clone(),
    )
    .queue_capacity(2)
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    hook.reached.notified().await;
    let first = InboxEntry::new(SignalKind::Steer, "first steer");
    session.submit(first.clone()).await.unwrap();
    session.submit(first).await.unwrap();
    session.steer("second steer").await.unwrap();
    assert!(matches!(
        session.follow_up("overflow").await,
        Err(Error::QueueFull)
    ));
    assert_eq!(store.snapshot().inbox.len(), 3);
    hook.release.notify_one();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert_eq!(model.request_count(), 3);
    let history = store.snapshot().history;
    assert_eq!(text(&history[4]), "first steer");
    assert_eq!(text(&history[5]), "second steer");
}

#[tokio::test]
async fn abort_preserves_completed_results_and_leaves_unstarted_inbox() {
    let model = MockCompletionModel::from_stream_turns([tool_turn()]);
    let hook = gate();
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    hook.reached.notified().await;
    session.follow_up("pending").await.unwrap();
    session.abort().await.unwrap();
    assert_eq!(idle(&session).await, EndReason::Aborted);
    assert_eq!(results(&store.snapshot().history).len(), 2);
    assert_eq!(store.snapshot().pending().len(), 1);
    validate_history(&store.snapshot().history).unwrap();
}

#[tokio::test]
async fn resume_does_not_duplicate_the_last_message_and_invalid_resume_is_local() {
    let store = MemoryStore::default();
    let session = Session::new(agent(&model(1)), store.clone())
        .start()
        .unwrap();
    assert!(matches!(
        session.resume().await,
        Err(Error::InvalidHistory(_))
    ));
    session.follow_up("hello").await.unwrap();
    idle(&session).await;
    session.pause().await.unwrap();
    session.stopped().await.unwrap();
    let history = store.snapshot().history;
    let model = model(1);
    let resumed = Session::new(agent(&model), store.clone())
        .history(history.clone())
        .start()
        .unwrap();
    resumed.resume().await.unwrap();
    assert_eq!(idle(&resumed).await, EndReason::Idle);
    assert_eq!(store.snapshot().history.len(), history.len() + 1);
    assert_eq!(model.requests()[0].chat_history, history);
}

#[tokio::test(start_paused = true)]
async fn loop_deadline_includes_idle_time() {
    let session = Session::new(agent(&model(1)), MemoryStore::default())
        .loop_timeout(Duration::from_millis(10))
        .start()
        .unwrap();
    assert_eq!(session.stopped().await.unwrap(), EndReason::LoopTimedOut);
}

#[derive(Clone)]
struct ReceiptFailure {
    inner: MemoryStore,
    hang: bool,
}
impl Store for ReceiptFailure {
    fn submit(&self, entry: InboxEntry) -> StoreFuture<InboxStatus> {
        self.inner.submit(entry)
    }
    fn commit(&self, commit: Commit) -> StoreFuture<()> {
        self.inner.commit(commit)
    }
    fn save_tool_result(&self, key: ToolKey, result: ToolResult) -> StoreFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            store.inner.save_tool_result(key, result).await?;
            if store.hang {
                std::future::pending().await
            } else {
                Err("lost receipt acknowledgement".into())
            }
        })
    }
    fn load_tool_result(&self, key: ToolKey) -> StoreFuture<Option<ToolResult>> {
        self.inner.load_tool_result(key)
    }
}
#[tokio::test(start_paused = true)]
async fn uncertain_tool_write_restarts_without_losing_the_real_result() {
    for hang in [false, true] {
        let store = ReceiptFailure {
            inner: MemoryStore::default(),
            hang,
        };
        let model = MockCompletionModel::from_stream_turns([tool_turn()]);
        let session = Session::new(
            AgentBuilder::new(model).tool(MockAddTool).build(),
            store.clone(),
        )
        .io_timeout(Duration::from_millis(10))
        .start()
        .unwrap();
        session.follow_up("start").await.unwrap();
        let error = session.wait_for_idle().await.unwrap_err();
        let Error::ToolWriteUncertain { key, result, .. } = error else {
            panic!("{error:?}")
        };
        store
            .inner
            .save_tool_result(key, (*result).clone())
            .await
            .unwrap();
        let history = store.inner.snapshot().history;
        let resumed = Session::new(agent(&model_for_resume()), store.inner.clone())
            .history(history)
            .start()
            .unwrap();
        assert_eq!(idle(&resumed).await, EndReason::Idle);
        assert_eq!(results(&store.inner.snapshot().history)[0], &*result);
        validate_history(&store.inner.snapshot().history).unwrap();
    }
}

#[derive(Clone)]
struct TestCompactor {
    inputs: Arc<std::sync::Mutex<Vec<Vec<Message>>>>,
}
impl rig_memory::Compactor for TestCompactor {
    type Artifact = Message;
    fn compact<'a>(
        &'a self,
        _: &'a str,
        evicted: &'a [Message],
        carry: Option<&'a Message>,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, Result<Message, rig_memory::MemoryError>> {
        Box::pin(async move {
            self.inputs.lock().unwrap().push(evicted.to_vec());
            let mut summary = carry.map(text).unwrap_or_default();
            for message in evicted {
                summary.push_str(&text(message));
            }
            Ok(Message::system(format!(
                "Remember project decisions: {summary}"
            )))
        })
    }
}
#[tokio::test]
async fn custom_compactor_runs_each_boundary_and_rebuilds_after_restart() {
    let store = MemoryStore::default();
    let inputs = Arc::new(std::sync::Mutex::new(Vec::new()));
    let make_agent = |model: MockCompletionModel| {
        AgentBuilder::new(model)
            .add_hook(context::CompactingContext::new(
                "conversation",
                rig_memory::SlidingWindowMemory::last_messages(1),
                TestCompactor {
                    inputs: inputs.clone(),
                },
            ))
            .build()
    };
    let first_model = model(3);
    let session = Session::new(make_agent(first_model.clone()), store.clone())
        .start()
        .unwrap();
    for prompt in ["decision A", "decision B", "decision C"] {
        session.follow_up(prompt).await.unwrap();
        idle(&session).await;
    }
    assert_eq!(
        inputs.lock().unwrap().iter().map(Vec::len).sum::<usize>(),
        4
    );
    assert_eq!(first_model.requests()[2].chat_history.len(), 2);
    assert!(text(&first_model.requests()[2].chat_history[0]).contains("decision A"));
    session.pause().await.unwrap();
    session.stopped().await.unwrap();
    let before = store.snapshot().history;
    let new_model = model(1);
    let resumed = Session::new(make_agent(new_model.clone()), store.clone())
        .history(before.clone())
        .start()
        .unwrap();
    resumed.follow_up("continue").await.unwrap();
    idle(&resumed).await;
    assert!(text(&new_model.requests()[0].chat_history[0]).contains("decision A"));
    assert_eq!(store.snapshot().history[..before.len()], before);
}

#[derive(Clone, Debug, Default)]
struct CacheHttp {
    requests: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    stream: rig::test_utils::SequencedStreamingHttpClient,
}
impl rig::http_client::HttpClientExt for CacheHttp {
    fn send<T, U>(
        &self,
        _: rig::http_client::Request<T>,
    ) -> impl std::future::Future<
        Output = rig::http_client::Result<
            rig::http_client::Response<rig::http_client::LazyBody<U>>,
        >,
    > + Send
    + 'static
    where
        T: Into<bytes::Bytes> + Send,
        U: From<bytes::Bytes> + Send + 'static,
    {
        std::future::ready(Err(rig::http_client::Error::Instance(Box::new(
            std::io::Error::other("stream-only fixture"),
        ))))
    }
    fn send_multipart<U>(
        &self,
        _: rig::http_client::Request<rig::http_client::MultipartForm>,
    ) -> impl std::future::Future<
        Output = rig::http_client::Result<
            rig::http_client::Response<rig::http_client::LazyBody<U>>,
        >,
    > + Send
    + 'static
    where
        U: From<bytes::Bytes> + Send + 'static,
    {
        std::future::ready(Err(rig::http_client::Error::Instance(Box::new(
            std::io::Error::other("stream-only fixture"),
        ))))
    }
    async fn send_streaming<T>(
        &self,
        request: rig::http_client::Request<T>,
    ) -> rig::http_client::Result<rig::http_client::StreamingResponse>
    where
        T: Into<bytes::Bytes> + Send,
    {
        let (parts, body) = request.into_parts();
        let body = body.into();
        self.requests
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&body).unwrap());
        self.stream
            .send_streaming(rig::http_client::Request::from_parts(parts, body))
            .await
    }
}
#[tokio::test]
async fn anthropic_cache_controls_reach_the_session_wire_request() {
    use rig::{
        client::CompletionClient,
        providers::anthropic::{self, completion::CacheTtl},
    };
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    for automatic in [false, true] {
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let http = CacheHttp {
            requests: requests.clone(),
            stream: rig::test_utils::SequencedStreamingHttpClient::new(vec![Ok(
                bytes::Bytes::from_static(sse.as_bytes()),
            )]),
        };
        let client = anthropic::Client::builder()
            .api_key("test-key")
            .http_client(http)
            .build()
            .unwrap();
        let model = client
            .completion_model("claude-sonnet-4-6")
            .with_static_prefix_cache_ttl(CacheTtl::OneHour);
        let model = if automatic {
            model.with_automatic_caching()
        } else {
            model.with_prompt_caching()
        };
        let session = Session::new(
            AgentBuilder::new(model)
                .preamble("Static instructions")
                .tool(MockAddTool)
                .build(),
            MemoryStore::default(),
        )
        .start()
        .unwrap();
        session.follow_up("hello").await.unwrap();
        assert_eq!(idle(&session).await, EndReason::Idle);
        let requests = requests.lock().unwrap();
        let body = &requests[0];
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
        if automatic {
            assert_eq!(body["cache_control"]["type"], "ephemeral");
            assert!(body["messages"][0]["content"][0]["cache_control"].is_null());
        } else {
            assert_eq!(
                body["messages"][0]["content"][0]["cache_control"]["type"],
                "ephemeral"
            );
            assert!(body["cache_control"].is_null());
        }
    }
}

#[derive(Clone)]
struct RecoveredToolObserver(Arc<AtomicUsize>);
impl AgentHook for RecoveredToolObserver {
    async fn on_invalid_tool_call(
        &self,
        _: &HookContext,
        _: &rig::agent::hook::InvalidToolCallContext,
    ) -> Option<rig::agent::hook::InvalidToolCallAction> {
        Some(rig::agent::hook::InvalidToolCallAction::repair("add"))
    }
    async fn on_tool_result(&self, _: &HookContext, _: ToolResultEvent<'_>) -> ToolResultAction {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResultAction::Keep
    }
}
#[tokio::test]
async fn unsupported_invalid_tool_repair_fails_before_execution() {
    let counter = Arc::new(AtomicUsize::new(0));
    let model = MockCompletionModel::from_stream_turns([vec![
        E::tool_call("a", "missing", serde_json::json!({"x":1,"y":2})),
        E::final_response_with_default_usage(),
    ]]);
    let session = Session::new(
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(RecoveredToolObserver(counter.clone()))
            .build(),
        MemoryStore::default(),
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert!(matches!(
        session.wait_for_idle().await,
        Err(Error::InvalidHistory(_))
    ));
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rejected_assistant_checkpoint_prevents_tool_execution() {
    let counter = Arc::new(AtomicUsize::new(0));
    let store = FailStore {
        inner: MemoryStore::default(),
        calls: Arc::new(AtomicUsize::new(0)),
        mode: 0,
    };
    let model = MockCompletionModel::from_stream_turns([tool_turn()]);
    let session = Session::new(
        AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(RecoveredToolObserver(counter.clone()))
            .build(),
        store,
    )
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert!(matches!(
        session.wait_for_idle().await,
        Err(Error::CommitUncertain { .. })
    ));
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn max_turns_keeps_the_completed_tool_batch() {
    let model = MockCompletionModel::from_stream_turns([tool_turn()]);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model).tool(MockAddTool).build(),
        store.clone(),
    )
    .max_turns(1)
    .start()
    .unwrap();
    session.follow_up("start").await.unwrap();
    assert_eq!(idle(&session).await, EndReason::MaxTurns);
    assert_eq!(results(&store.snapshot().history).len(), 2);
    validate_history(&store.snapshot().history).unwrap();
}

#[tokio::test]
async fn repeated_provider_call_ids_have_distinct_receipt_positions() {
    let model = MockCompletionModel::from_stream_turns([
        tool_turn(),
        vec![E::text("one"), E::final_response_with_default_usage()],
        tool_turn(),
        vec![E::text("two"), E::final_response_with_default_usage()],
    ]);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model).tool(MockAddTool).build(),
        store.clone(),
    )
    .start()
    .unwrap();
    for _ in 0..2 {
        session.follow_up("start").await.unwrap();
        assert_eq!(idle(&session).await, EndReason::Idle);
    }
    assert_eq!(store.snapshot().tool_results.len(), 4);
    assert_eq!(results(&store.snapshot().history).len(), 4);
}

#[tokio::test]
async fn store_rejects_double_acks_and_stale_or_conflicting_transactions() {
    let store = MemoryStore::default();
    let entry = InboxEntry::new(SignalKind::FollowUp, "one");
    store.submit(entry.clone()).await.unwrap();
    let mut commit = Commit {
        id: uuid::Uuid::new_v4(),
        expected_messages: 0,
        messages: vec![entry.message.clone()],
        ack_inbox_ids: vec![entry.id.clone(), entry.id.clone()],
        outcome: None,
    };
    assert!(store.commit(commit.clone()).await.is_err());
    commit.ack_inbox_ids.pop();
    store.commit(commit.clone()).await.unwrap();
    store.commit(commit.clone()).await.unwrap();
    commit.id = uuid::Uuid::new_v4();
    assert!(store.commit(commit).await.is_err());
    assert_eq!(store.snapshot().acked_ids.len(), 1);
    assert_eq!(store.snapshot().history.len(), 1);
}

#[test]
fn validator_rejects_orphans_duplicate_results_and_provider_metadata_conflicts() {
    let call = rig::message::ToolCall::from_wire(
        "a",
        rig::message::ToolFunction {
            name: "add".into(),
            arguments: serde_json::json!({}),
        },
    );
    let assistant = Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(call.clone())],
    };
    let result = ToolResult {
        call: call.id,
        provider: call.provider,
        name: "add".into(),
        content: vec![ToolResultContent::text("ok")],
    };
    let user = |result: ToolResult| Message::User {
        content: vec![UserContent::ToolResult(result)],
    };
    assert!(validate_history(&[user(result.clone())]).is_err());
    assert!(
        validate_history(&[
            assistant.clone(),
            Message::User {
                content: vec![
                    UserContent::ToolResult(result.clone()),
                    UserContent::ToolResult(result.clone())
                ]
            }
        ])
        .is_err()
    );
    let mut wrong = result.clone();
    wrong.provider = None;
    assert!(validate_history(&[assistant.clone(), user(wrong)]).is_err());
    validate_history(&[assistant, user(result)]).unwrap();
}

#[tokio::test]
async fn compaction_preserves_configured_system_preamble() {
    let model = model(2);
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .preamble("Persistent application instructions")
            .add_hook(context::ContextPolicy(
                rig_memory::SlidingWindowMemory::last_messages(1),
            ))
            .build(),
        MemoryStore::default(),
    )
    .start()
    .unwrap();
    for _ in 0..2 {
        session.follow_up("start").await.unwrap();
        idle(&session).await;
    }
    for request in model.requests() {
        assert_eq!(
            request.chat_history.first(),
            Some(&Message::system("Persistent application instructions"))
        );
    }
}

#[tokio::test]
async fn reasoning_deltas_use_rigs_canonical_assembly() {
    let model = MockCompletionModel::from_stream_turns([vec![
        E::reasoning_delta("first "),
        E::reasoning_delta("second"),
        E::text("answer"),
        E::final_response_with_default_usage(),
    ]]);
    let store = MemoryStore::default();
    let session = Session::new(agent(&model), store.clone()).start().unwrap();
    session.follow_up("start").await.unwrap();
    idle(&session).await;
    let snapshot = store.snapshot();
    let Message::Assistant { content, .. } = &snapshot.history[1] else {
        panic!()
    };
    assert!(format!("{content:?}").contains("first second"));
    assert_eq!(snapshot.history.len(), 2);
}

#[tokio::test]
#[ignore = "release-mode bookkeeping measurement, not a provider-latency benchmark"]
async fn append_bookkeeping_scaling() {
    use std::{hint::black_box, time::Instant};
    println!("history_messages,body_bytes,append_us,prior_prefix_copy_pattern_us");
    for size in [64, 4096] {
        for n in [100, 1000, 10000] {
            let history: Vec<_> = (0..n).map(|_| Message::user("x".repeat(size))).collect();
            let persisted = history.clone();
            let store = MemoryStore::default();
            store
                .commit(Commit {
                    id: uuid::Uuid::new_v4(),
                    expected_messages: 0,
                    messages: history.clone(),
                    ack_inbox_ids: vec![],
                    outcome: None,
                })
                .await
                .unwrap();
            let rounds = 25;
            let start = Instant::now();
            for i in 0..rounds {
                store
                    .commit(Commit {
                        id: uuid::Uuid::new_v4(),
                        expected_messages: n + i,
                        messages: vec![Message::assistant("delta")],
                        ack_inbox_ids: vec![],
                        outcome: None,
                    })
                    .await
                    .unwrap();
            }
            let append = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
            let start = Instant::now();
            for _ in 0..rounds {
                // Reference to the removed checkpointer's successful-prefix
                // path. This is not an end-to-end old-runtime benchmark.
                assert!(black_box(&history).starts_with(black_box(&persisted)));
                black_box(history.clone());
            }
            let prior = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
            println!("{n},{size},{append:.2},{prior:.2}");
        }
    }
}

#[derive(Clone)]
struct BlockedSubmission {
    inner: MemoryStore,
    written: Arc<Notify>,
    release: Arc<Notify>,
}
impl Store for BlockedSubmission {
    fn submit(&self, entry: InboxEntry) -> StoreFuture<InboxStatus> {
        let store = self.clone();
        Box::pin(async move {
            let status = store.inner.submit(entry).await?;
            store.written.notify_one();
            store.release.notified().await;
            Ok(status)
        })
    }
    fn commit(&self, commit: Commit) -> StoreFuture<()> {
        self.inner.commit(commit)
    }
    fn save_tool_result(&self, key: ToolKey, result: ToolResult) -> StoreFuture<()> {
        self.inner.save_tool_result(key, result)
    }
    fn load_tool_result(&self, key: ToolKey) -> StoreFuture<Option<ToolResult>> {
        self.inner.load_tool_result(key)
    }
}
#[tokio::test]
async fn cancelling_submit_caller_does_not_lose_durable_work() {
    let store = BlockedSubmission {
        inner: MemoryStore::default(),
        written: Arc::new(Notify::new()),
        release: Arc::new(Notify::new()),
    };
    let model = model(1);
    let session = Session::new(agent(&model), store.clone()).start().unwrap();
    let entry = InboxEntry::new(SignalKind::FollowUp, "start");
    let caller = tokio::spawn({
        let session = session.clone();
        let entry = entry.clone();
        async move { session.submit(entry).await }
    });
    store.written.notified().await;
    caller.abort();
    store.release.notify_one();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert_eq!(store.inner.snapshot().acked_ids, vec![entry.id]);
    assert_eq!(model.request_count(), 1);
}

#[derive(Clone)]
struct BlockedRecovery(MemoryStore);
impl Store for BlockedRecovery {
    fn submit(&self, entry: InboxEntry) -> StoreFuture<InboxStatus> {
        self.0.submit(entry)
    }
    fn commit(&self, commit: Commit) -> StoreFuture<()> {
        self.0.commit(commit)
    }
    fn save_tool_result(&self, key: ToolKey, result: ToolResult) -> StoreFuture<()> {
        self.0.save_tool_result(key, result)
    }
    fn load_tool_result(&self, _: ToolKey) -> StoreFuture<Option<ToolResult>> {
        Box::pin(std::future::pending())
    }
}
#[tokio::test(start_paused = true)]
async fn recovery_reads_are_bounded_before_any_model_request() {
    let call = rig::message::ToolCall::from_wire(
        "a",
        rig::message::ToolFunction {
            name: "add".into(),
            arguments: serde_json::json!({}),
        },
    );
    let history = vec![Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(call)],
    }];
    for loop_limit in [false, true] {
        let store = MemoryStore::default();
        store
            .commit(Commit {
                id: uuid::Uuid::new_v4(),
                expected_messages: 0,
                messages: history.clone(),
                ack_inbox_ids: vec![],
                outcome: None,
            })
            .await
            .unwrap();
        let model = model(1);
        let builder = Session::new(agent(&model), BlockedRecovery(store)).history(history.clone());
        let builder = if loop_limit {
            builder.loop_timeout(Duration::from_millis(10))
        } else {
            builder.io_timeout(Duration::from_millis(10))
        };
        let session = builder.start().unwrap();
        assert!(matches!(
            session.stopped().await,
            Err(Error::Store {
                operation: "load tool receipt",
                ..
            })
        ));
        assert_eq!(model.request_count(), 0);
    }
}

struct BlockingCompactor {
    inner: TestCompactor,
    started: Arc<Notify>,
    release: Arc<Notify>,
    calls: AtomicUsize,
}
impl rig_memory::Compactor for BlockingCompactor {
    type Artifact = Message;
    fn compact<'a>(
        &'a self,
        id: &'a str,
        evicted: &'a [Message],
        carry: Option<&'a Message>,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, Result<Message, rig_memory::MemoryError>> {
        Box::pin(async move {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.started.notify_one();
                self.release.notified().await;
            }
            self.inner.compact(id, evicted, carry).await
        })
    }
}
#[tokio::test]
async fn follow_up_is_accepted_while_compaction_is_running() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let compactor = BlockingCompactor {
        inner: TestCompactor {
            inputs: Arc::default(),
        },
        started: started.clone(),
        release: release.clone(),
        calls: AtomicUsize::new(0),
    };
    let model = model(3);
    let store = MemoryStore::default();
    let session = Session::new(
        AgentBuilder::new(model.clone())
            .add_hook(context::CompactingContext::new(
                "one",
                rig_memory::SlidingWindowMemory::last_messages(1),
                compactor,
            ))
            .build(),
        store.clone(),
    )
    .start()
    .unwrap();
    session.follow_up("first").await.unwrap();
    idle(&session).await;
    session.follow_up("second").await.unwrap();
    started.notified().await;
    session.follow_up("queued during compaction").await.unwrap();
    release.notify_one();
    assert_eq!(idle(&session).await, EndReason::Idle);
    assert_eq!(store.snapshot().history.len(), 6);
    assert!(store.snapshot().pending().is_empty());
    assert_eq!(model.request_count(), 3);
}
