use super::*;
use rig::{
    OneOrMany,
    agent::AgentBuilder,
    agent::MultiTurnStreamItem,
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse, ToolDefinition,
        Usage,
    },
    message::{AssistantContent, Message, ToolCall, ToolResult, ToolResultContent, UserContent},
    streaming::{
        RawStreamingChoice, StreamedAssistantContent, StreamedUserContent,
        StreamingCompletionResponse, ToolCallDeltaContent,
    },
    test_utils::{MockAddTool, MockCompletionModel, MockResponse, MockStreamEvent},
    tool::Tool,
};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::{
    sync::{Notify, broadcast},
    time::{Duration, sleep, timeout},
};

#[tokio::test(flavor = "current_thread")]
async fn follow_up_keeps_loop_running_after_first_response() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("follow"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent);

    let handle = agent_loop.prompt(Message::user("start"));
    handle.follow_up(Message::user("follow-up")).unwrap();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(result.last_response.as_deref(), Some("follow"));
    assert_eq!(model.request_count(), 2);
    assert_eq!(user_prompts(&model), vec!["start", "follow-up"]);
}

#[tokio::test(flavor = "current_thread")]
async fn with_history_seeds_active_state_and_first_request() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent).with_history([Message::user("previous")]);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    let requests = model.requests();
    assert_eq!(
        user_texts(requests[0].chat_history.iter()),
        vec!["previous", "start"]
    );
    assert_eq!(user_texts(result.history.iter()), vec!["previous", "start"]);
}

#[tokio::test(flavor = "current_thread")]
async fn agent_loop_resume_starts_from_history_without_appending_a_user_message() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("resumed"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop =
        AgentLoop::new(agent).with_history([Message::user("start"), Message::assistant("first")]);

    let result = agent_loop.resume().unwrap().wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 1);
    assert_eq!(user_texts(result.history.iter()), vec!["start"]);
    assert_eq!(
        assistant_texts(result.history.iter()),
        vec!["first", "resumed"]
    );

    let requests = model.requests();
    assert_eq!(user_texts(requests[0].chat_history.iter()), vec!["start"]);
    assert_eq!(
        assistant_texts(requests[0].chat_history.iter()),
        vec!["first"]
    );
}

#[test]
fn agent_loop_resume_errors_when_seeded_history_is_invalid() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent).with_history([Message::system("summary")]);

    let err = match agent_loop.resume() {
        Ok(_) => panic!("invalid message history should not start a resumed loop"),
        Err(err) => err,
    };

    assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
}

#[tokio::test(flavor = "current_thread")]
async fn unanswered_tool_call_history_is_repaired_before_prompt() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent).with_history([assistant_tool_call_message("call_1")]);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 1);
    let requests = model.requests();
    assert_eq!(
        tool_result_texts(requests[0].chat_history.iter()),
        vec![DEFAULT_TOOL_REPAIR_MESSAGE]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unanswered_tool_call_repair_message_can_be_overridden() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent)
        .with_history([assistant_tool_call_message("call_1")])
        .with_unanswered_tool_call_repair_message("custom recovery message");

    agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    let requests = model.requests();
    assert_eq!(
        tool_result_texts(requests[0].chat_history.iter()),
        vec!["custom recovery message"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn incremental_tool_result_persistence_records_completed_results() {
    let model = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::tool_call(
            "call_1",
            "add",
            serde_json::json!({"x": 1, "y": 2}),
        )],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let persistence = MemoryToolResultPersistence::default();
    let agent = AgentBuilder::new(model.clone()).tool(MockAddTool).build();
    let agent_loop =
        AgentLoop::new(agent).with_incremental_tool_result_persistence(persistence.clone());

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 2);
    let persisted = persistence
        .result(ToolResultKey::new("call_1", None))
        .expect("tool result should be persisted as soon as it completes");
    assert_eq!(persisted.id, "call_1");
    assert_eq!(tool_result_text(&persisted).as_deref(), Some("3"));
}

#[tokio::test(flavor = "current_thread")]
async fn unanswered_tool_call_repair_prefers_persisted_tool_results() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let persistence = MemoryToolResultPersistence::default();
    persistence.insert(
        ToolResultKey::new("call_1", None),
        test_tool_result_with_text("call_1", "persisted result"),
    );
    let agent = AgentBuilder::new(model.clone()).build();
    let history = vec![Message::Assistant {
        id: None,
        content: OneOrMany::many(vec![
            AssistantContent::ToolCall(test_tool_call("call_1")),
            AssistantContent::ToolCall(test_tool_call("call_2")),
        ])
        .unwrap(),
    }];
    let agent_loop = AgentLoop::new(agent)
        .with_history(history)
        .with_incremental_tool_result_persistence(persistence);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    validate_message_history(&result.history).unwrap();
    assert_eq!(model.request_count(), 1);
    let requests = model.requests();
    assert_eq!(
        tool_result_texts(requests[0].chat_history.iter()),
        vec!["persisted result", DEFAULT_TOOL_REPAIR_MESSAGE]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unanswered_tool_call_repair_uses_provider_call_id_key() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let persistence = MemoryToolResultPersistence::default();
    persistence.insert(
        ToolResultKey::new("call_1", Some("provider_call_1".to_string())),
        test_tool_result_with_text("call_1", "persisted with call id"),
    );
    let agent = AgentBuilder::new(model.clone()).build();
    let history = vec![Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::ToolCall(
            test_tool_call("call_1").with_call_id("provider_call_1".to_string()),
        )),
    }];
    let agent_loop = AgentLoop::new(agent)
        .with_history(history)
        .with_incremental_tool_result_persistence(persistence);

    agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert_eq!(model.request_count(), 1);
    let requests = model.requests();
    let repaired_results = tool_results(requests[0].chat_history.iter());
    assert_eq!(repaired_results.len(), 1);
    assert_eq!(repaired_results[0].id, "call_1");
    assert_eq!(
        repaired_results[0].call_id.as_deref(),
        Some("provider_call_1")
    );
    assert_eq!(
        tool_result_text(&repaired_results[0]).as_deref(),
        Some("persisted with call id")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unanswered_tool_call_repair_merges_persisted_results_into_partial_tool_batch() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let persistence = MemoryToolResultPersistence::default();
    persistence.insert(
        ToolResultKey::new("call_1", None),
        test_tool_result_with_text("call_1", "persisted first"),
    );
    let agent = AgentBuilder::new(model.clone()).build();
    let history = vec![
        Message::Assistant {
            id: None,
            content: OneOrMany::many(vec![
                AssistantContent::ToolCall(test_tool_call("call_1")),
                AssistantContent::ToolCall(test_tool_call("call_2")),
            ])
            .unwrap(),
        },
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_2"))),
        },
    ];
    let agent_loop = AgentLoop::new(agent)
        .with_history(history)
        .with_incremental_tool_result_persistence(persistence);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    validate_message_history(&result.history).unwrap();
    assert_eq!(model.request_count(), 1);
    let requests = model.requests();
    assert_eq!(
        tool_result_texts(requests[0].chat_history.iter()),
        vec!["persisted first", "echoed"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn tool_result_persistence_load_error_stops_before_request() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let persistence = MemoryToolResultPersistence::default();
    persistence.fail_load();
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent)
        .with_history([assistant_tool_call_message("call_1")])
        .with_incremental_tool_result_persistence(persistence);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    let err = handle
        .wait()
        .await
        .expect_err("load failures should fail startup repair");

    assert!(matches!(err, AgentLoopError::ToolResultPersistence(_)));
    assert_eq!(model.request_count(), 0);
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::LoopFailed { error }
                if error.message.contains("tool result persistence failed: load failed")
        )
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn tool_result_persistence_persist_error_stops_loop() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::tool_call(
        "call_1",
        "add",
        serde_json::json!({"x": 1, "y": 2}),
    )]]);
    let persistence = MemoryToolResultPersistence::default();
    persistence.fail_persist();
    let agent = AgentBuilder::new(model.clone()).tool(MockAddTool).build();
    let agent_loop =
        AgentLoop::new(agent).with_incremental_tool_result_persistence(persistence.clone());
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    let err = handle
        .wait()
        .await
        .expect_err("persist failures should fail the active turn");

    assert!(matches!(err, AgentLoopError::ToolResultPersistence(_)));
    assert_eq!(model.request_count(), 1);
    assert!(
        persistence
            .result(ToolResultKey::new("call_1", None))
            .is_none()
    );
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::LoopFailed { error }
                if error.message.contains("tool result persistence failed: persist failed")
        )
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_message_history_with_orphan_tool_result_still_fails_before_request() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent).with_history([Message::User {
        content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_1"))),
    }]);

    let err = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .expect_err("unrepairable message history should fail before a request is sent");

    assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
    assert_eq!(model.request_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn turn_hook_replaces_active_history_between_turns() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("follow"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let compacted = Arc::new(Mutex::new(false));
    let should_compact = compacted.clone();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
        let should_compact = should_compact.clone();
        async move {
            let replace = {
                let mut compacted = should_compact.lock().unwrap();
                let replace = !*compacted && turn.history.len() >= 2;
                if replace {
                    *compacted = true;
                }
                replace
            };

            if replace {
                Ok::<_, std::io::Error>(TurnHookAction::ReplaceHistory(vec![Message::user(
                    "summary",
                )]))
            } else {
                Ok(TurnHookAction::Continue)
            }
        }
    });

    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();
    handle.follow_up(Message::user("follow-up")).unwrap();

    let result = handle.wait().await.unwrap();

    let requests = model.requests();
    assert_eq!(
        user_texts(requests[1].chat_history.iter()),
        vec!["summary", "follow-up"]
    );
    assert_eq!(
        user_texts(result.history.iter()),
        vec!["summary", "follow-up"]
    );

    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::HistoryReplaced { messages }
                if user_texts(messages.iter()) == vec!["summary"]
        )
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn turn_hook_error_stops_the_loop() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent)
        .with_turn_hook(|_, _turn| async { Err(std::io::Error::other("hook failed")) });
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    let err = handle
        .wait()
        .await
        .expect_err("turn hook failure should fail the run");

    assert!(matches!(err, AgentLoopError::TurnHook(_)));
    assert_eq!(model.request_count(), 1);

    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::LoopFailed { error }
                if error.message.contains("turn hook failed: hook failed")
        )
    }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentLoopEvent::LoopEnded { .. }))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn turn_hook_invalid_replacement_fails_the_commit() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(|_, _turn| async {
        Ok::<_, std::io::Error>(TurnHookAction::ReplaceHistory(vec![
            assistant_tool_call_message("call_1"),
        ]))
    });

    let err = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .expect_err("invalid replacement history should fail the commit");

    assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
    assert_eq!(model.request_count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn steering_runs_before_follow_ups() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("steered"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("followed"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent);

    let handle = agent_loop.prompt(Message::user("start"));
    handle.follow_up(Message::user("follow-up")).unwrap();
    handle.steer(Message::user("steer")).unwrap();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 3);
    assert_eq!(user_prompts(&model), vec!["start", "steer", "follow-up"]);
}

#[tokio::test(flavor = "current_thread")]
async fn steering_messages_are_applied_together_by_default() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("steered"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("followed"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent);

    let handle = agent_loop.prompt(Message::user("start"));
    handle.follow_up(Message::user("follow-up")).unwrap();
    handle.steer(Message::user("steer-one")).unwrap();
    handle.steer(Message::user("steer-two")).unwrap();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 3);
    assert_eq!(
        user_prompts(&model),
        vec!["start", "steer-two", "follow-up"]
    );

    let requests = model.requests();
    assert_eq!(
        user_texts(requests[1].chat_history.iter()),
        vec!["start", "steer-one", "steer-two"]
    );
    assert_eq!(
        user_texts(result.history.iter()),
        vec!["start", "steer-one", "steer-two", "follow-up"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn steer_ready_at_idle_boundary_prevents_loop_from_ending() {
    let first_commit_entered = Arc::new(Notify::new());
    let release_first_commit = Arc::new(Notify::new());
    let first_commit_seen = Arc::new(AtomicBool::new(false));
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("steered"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let hook_first_commit_entered = first_commit_entered.clone();
    let hook_release_first_commit = release_first_commit.clone();
    let hook_first_commit_seen = first_commit_seen.clone();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
        let hook_first_commit_entered = hook_first_commit_entered.clone();
        let hook_release_first_commit = hook_release_first_commit.clone();
        let hook_first_commit_seen = hook_first_commit_seen.clone();
        async move {
            if !hook_first_commit_seen.swap(true, Ordering::SeqCst) {
                assert_eq!(turn.end_reason, Some(EndReason::Idle));
                hook_first_commit_entered.notify_one();
                hook_release_first_commit.notified().await;
            }
            Ok::<_, std::io::Error>(TurnHookAction::Continue)
        }
    });

    let handle = agent_loop.prompt(Message::user("start"));
    handle.finish_when_idle().unwrap();
    timeout(Duration::from_secs(1), first_commit_entered.notified())
        .await
        .expect("first commit should reach idle boundary");

    handle.steer(Message::user("late steer")).unwrap();
    release_first_commit.notify_one();
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(result.last_response.as_deref(), Some("steered"));
    assert_eq!(model.request_count(), 2);
    assert_eq!(user_prompts(&model), vec!["start", "late steer"]);
    assert_eq!(
        user_texts(result.history.iter()),
        vec!["start", "late steer"]
    );
    assert_eq!(
        assistant_texts(result.history.iter()),
        vec!["first", "steered"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn interrupt_ready_at_idle_boundary_prevents_loop_from_ending() {
    let first_commit_entered = Arc::new(Notify::new());
    let release_first_commit = Arc::new(Notify::new());
    let first_commit_seen = Arc::new(AtomicBool::new(false));
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("interrupted"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let hook_first_commit_entered = first_commit_entered.clone();
    let hook_release_first_commit = release_first_commit.clone();
    let hook_first_commit_seen = first_commit_seen.clone();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
        let hook_first_commit_entered = hook_first_commit_entered.clone();
        let hook_release_first_commit = hook_release_first_commit.clone();
        let hook_first_commit_seen = hook_first_commit_seen.clone();
        async move {
            if !hook_first_commit_seen.swap(true, Ordering::SeqCst) {
                assert_eq!(turn.end_reason, Some(EndReason::Idle));
                hook_first_commit_entered.notify_one();
                hook_release_first_commit.notified().await;
            }
            Ok::<_, std::io::Error>(TurnHookAction::Continue)
        }
    });

    let handle = agent_loop.prompt(Message::user("start"));
    handle.finish_when_idle().unwrap();
    timeout(Duration::from_secs(1), first_commit_entered.notified())
        .await
        .expect("first commit should reach idle boundary");

    handle.interrupt(Message::user("late interrupt")).unwrap();
    release_first_commit.notify_one();
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(result.last_response.as_deref(), Some("interrupted"));
    assert_eq!(model.request_count(), 2);
    assert_eq!(user_prompts(&model), vec!["start", "late interrupt"]);
    assert_eq!(
        user_texts(result.history.iter()),
        vec!["start", "late interrupt"]
    );
    assert_eq!(
        assistant_texts(result.history.iter()),
        vec!["first", "interrupted"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn follow_ups_are_applied_one_at_a_time() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("follow-one"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("follow-two"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent);

    let handle = agent_loop.prompt(Message::user("start"));
    handle.follow_up(Message::user("follow-up-one")).unwrap();
    handle.follow_up(Message::user("follow-up-two")).unwrap();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 3);
    assert_eq!(
        user_prompts(&model),
        vec!["start", "follow-up-one", "follow-up-two"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn resume_runs_again_without_appending_a_user_message() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("resumed"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    loop {
        match events.recv().await.unwrap() {
            AgentLoopEvent::TurnCommitted { messages }
                if first_user_text(&messages[0]).is_some() =>
            {
                handle.resume().unwrap();
                break;
            }
            AgentLoopEvent::LoopEnded { .. } => {
                panic!("loop ended before resume could be queued")
            }
            _ => {}
        }
    }

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(result.last_response.as_deref(), Some("resumed"));
    assert_eq!(model.request_count(), 2);
    assert_eq!(user_texts(result.history.iter()), vec!["start"]);
    assert_eq!(
        assistant_texts(result.history.iter()),
        vec!["first", "resumed"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn resume_is_noop_while_turn_is_running() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone())
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));

    entered.notified().await;
    handle.resume().unwrap();
    release.notify_one();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    assert_eq!(result.last_response.as_deref(), Some("done"));
    assert_eq!(model.request_count(), 2);
}

#[test]
fn resume_history_requires_valid_committed_context() {
    assert!(validate_resume_history(&[]).is_err());
    assert!(validate_resume_history(&[Message::system("summary")]).is_err());
}

#[test]
fn repair_unanswered_tool_calls_adds_missing_results_to_next_user_message() {
    let history = vec![
        Message::Assistant {
            id: None,
            content: OneOrMany::many(vec![
                AssistantContent::ToolCall(test_tool_call("call_1")),
                AssistantContent::ToolCall(test_tool_call("call_2")),
            ])
            .unwrap(),
        },
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_1"))),
        },
    ];

    let repaired = repair_unanswered_tool_calls(history, "tool repaired");

    validate_message_history(&repaired).unwrap();
    assert_eq!(
        tool_result_texts(repaired.iter()),
        vec!["echoed", "tool repaired"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn repair_unanswered_tool_calls_with_persistence_prefers_recorded_results() {
    let persistence = MemoryToolResultPersistence::default();
    persistence.insert(
        ToolResultKey::new("call_1", None),
        test_tool_result_with_text("call_1", "persisted first"),
    );
    let history = vec![
        Message::Assistant {
            id: None,
            content: OneOrMany::many(vec![
                AssistantContent::ToolCall(test_tool_call("call_1")),
                AssistantContent::ToolCall(test_tool_call("call_2")),
                AssistantContent::ToolCall(test_tool_call("call_3")),
            ])
            .unwrap(),
        },
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_2"))),
        },
    ];

    let repaired =
        repair_unanswered_tool_calls_with_persistence(history, "tool repaired", Some(&persistence))
            .await
            .unwrap();

    validate_message_history(&repaired).unwrap();
    assert_eq!(
        tool_result_texts(repaired.iter()),
        vec!["persisted first", "echoed", "tool repaired"]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn abort_returns_aborted_end_reason() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);

    let handle = agent_loop.prompt(Message::user("start"));
    handle.abort().unwrap();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Aborted);
}

#[tokio::test(flavor = "current_thread")]
async fn pause_while_idle_returns_paused_end_reason() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    loop {
        match events.recv().await.unwrap() {
            AgentLoopEvent::TurnCommitted { .. } => break,
            AgentLoopEvent::LoopEnded { end_reason } => {
                panic!("loop ended before pause could be queued: {end_reason:?}")
            }
            _ => {}
        }
    }

    handle.pause().unwrap();
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("pause while idle should finish promptly")
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Paused);
    assert_eq!(model.request_count(), 1);
    assert_eq!(user_texts(result.history.iter()), vec!["start"]);
}

#[tokio::test(flavor = "current_thread")]
async fn pause_mid_turn_leaves_follow_up_unprocessed() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("queued follow-up"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model.clone())
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let seen_end_reasons = Arc::new(Mutex::new(Vec::new()));
    let hook_seen_end_reasons = seen_end_reasons.clone();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
        let hook_seen_end_reasons = hook_seen_end_reasons.clone();
        async move {
            hook_seen_end_reasons.lock().unwrap().push(turn.end_reason);
            Ok::<_, std::io::Error>(TurnHookAction::Continue)
        }
    });
    let handle = agent_loop.prompt(Message::user("start"));

    timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("tool should block the active turn");
    handle.follow_up(Message::user("follow-up")).unwrap();
    handle.pause().unwrap();
    release.notify_one();

    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("pause should finish after the active turn")
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Paused);
    assert_eq!(result.last_response.as_deref(), Some("done"));
    assert_eq!(model.request_count(), 2);
    assert_eq!(user_texts(result.history.iter()), vec!["start"]);
    assert_eq!(
        &seen_end_reasons.lock().unwrap()[..],
        &[Some(EndReason::Paused)]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_handle_aborts_running_loop() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let dropped = Arc::new(Notify::new());
    let agent = AgentBuilder::new(BlockingStreamModel {
        entered: entered.clone(),
        release,
        dropped: dropped.clone(),
    })
    .build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    entered.notified().await;
    drop(handle);

    timeout(Duration::from_secs(1), dropped.notified())
        .await
        .expect("dropping the handle should abort the in-flight stream");
    let events = drain_events(&mut events);
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentLoopEvent::LoopEnded { .. } | AgentLoopEvent::LoopFailed { .. }
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn cancelling_wait_aborts_running_loop() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let dropped = Arc::new(Notify::new());
    let agent = AgentBuilder::new(BlockingStreamModel {
        entered: entered.clone(),
        release,
        dropped: dropped.clone(),
    })
    .build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));

    entered.notified().await;
    let mut wait = Box::pin(handle.wait());
    assert!(futures::poll!(wait.as_mut()).is_pending());
    drop(wait);

    timeout(Duration::from_secs(1), dropped.notified())
        .await
        .expect("cancelling wait should abort the in-flight stream");
}

#[tokio::test(flavor = "current_thread")]
async fn turn_hook_receives_candidate_history_and_new_messages() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize, Option<String>)>::new()));
    let hook_seen = seen.clone();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
        let hook_seen = hook_seen.clone();
        let first_user = turn.new_messages.first().and_then(first_user_text);
        async move {
            hook_seen.lock().unwrap().push((
                turn.history.len(),
                turn.new_messages.len(),
                first_user,
            ));
            Ok::<_, std::io::Error>(TurnHookAction::Continue)
        }
    });

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    let seen = seen.lock().unwrap();
    assert_eq!(&seen[..], &[(2, 2, Some("start".to_string()))]);
}

#[tokio::test(flavor = "current_thread")]
async fn assistant_message_hook_runs_while_tool_execution_is_in_flight() {
    let hook_entered = Arc::new(Notify::new());
    let hook_release = Arc::new(Notify::new());
    let tool_entered = Arc::new(Notify::new());
    let tool_release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("I will call the tool."),
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model)
        .tool(BlockingTool {
            entered: tool_entered.clone(),
            release: tool_release.clone(),
        })
        .build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize, usize)>::new()));
    let hook_seen = seen.clone();
    let hook_entered_for_hook = hook_entered.clone();
    let hook_release_for_hook = hook_release.clone();
    let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(move |_, context| {
        let hook_seen = hook_seen.clone();
        let hook_entered = hook_entered_for_hook.clone();
        let hook_release = hook_release_for_hook.clone();
        async move {
            let tool_call_count = match &context.message {
                Message::Assistant { content, .. } => assistant_tool_call_ids(content).len(),
                _ => 0,
            };
            hook_seen.lock().unwrap().push((
                context.history.len(),
                context.new_messages.len(),
                tool_call_count,
            ));
            if tool_call_count > 0 {
                hook_entered.notify_one();
                hook_release.notified().await;
            }
            Ok::<_, std::io::Error>(())
        }
    });

    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    timeout(Duration::from_secs(1), hook_entered.notified())
        .await
        .expect("assistant message hook should start");
    timeout(Duration::from_secs(1), tool_entered.notified())
        .await
        .expect("tool execution should start while hook is still pending");

    hook_release.notify_one();
    tool_release.notify_one();
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("loop should finish after hook and tool are released")
        .unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    let seen = seen.lock().unwrap();
    assert!(seen.contains(&(2, 2, 1)));
    assert!(seen.contains(&(4, 4, 0)));
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::AssistantMessageFinished { message, messages }
                if matches!(message, Message::Assistant { .. }) && messages.len() == 2
        )
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn assistant_message_snapshot_can_be_repaired_after_restart() {
    let persisted_messages = Arc::new(Mutex::new(None::<Vec<Message>>));
    let persisted = Arc::new(Notify::new());
    let tool_entered = Arc::new(Notify::new());
    let tool_release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("I will call the tool."),
        MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model)
        .tool(BlockingTool {
            entered: tool_entered.clone(),
            release: tool_release.clone(),
        })
        .build();
    let messages_for_hook = persisted_messages.clone();
    let persisted_for_hook = persisted.clone();
    let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(move |_, context| {
        let messages_for_hook = messages_for_hook.clone();
        let persisted = persisted_for_hook.clone();
        async move {
            let has_tool_call = matches!(
                &context.message,
                Message::Assistant { content, .. } if !assistant_tool_call_ids(content).is_empty()
            );
            if has_tool_call {
                *messages_for_hook.lock().unwrap() = Some(context.new_messages);
                persisted.notify_one();
            }
            Ok::<_, std::io::Error>(())
        }
    });

    let handle = agent_loop.prompt(Message::user("start"));

    timeout(Duration::from_secs(1), persisted.notified())
        .await
        .expect("assistant snapshot should be persisted before tool result");
    timeout(Duration::from_secs(1), tool_entered.notified())
        .await
        .expect("tool should be running when the process crashes");

    drop(handle);
    tool_release.notify_waiters();

    let partial_history = persisted_messages
        .lock()
        .unwrap()
        .clone()
        .expect("assistant snapshot should be captured");
    assert!(validate_message_history(&partial_history).is_err());

    let recovery_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let recovery_agent = AgentBuilder::new(recovery_model.clone()).build();
    let result = AgentLoop::new(recovery_agent)
        .with_history(partial_history)
        .resume()
        .unwrap()
        .wait()
        .await
        .unwrap();

    validate_message_history(&result.history).unwrap();
    assert_eq!(
        tool_result_texts(result.history.iter()),
        vec![DEFAULT_TOOL_REPAIR_MESSAGE]
    );
    assert_eq!(
        assistant_texts(result.history.iter()),
        vec!["I will call the tool.", "recovered"]
    );
    assert_eq!(recovery_model.request_count(), 1);

    let requests = recovery_model.requests();
    assert_eq!(
        tool_result_texts(requests[0].chat_history.iter()),
        vec![DEFAULT_TOOL_REPAIR_MESSAGE]
    );
    let tool_call_ids = requests[0]
        .chat_history
        .iter()
        .flat_map(|message| match message {
            Message::Assistant { content, .. } => assistant_tool_call_ids(content),
            Message::User { .. } | Message::System { .. } => Vec::new(),
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_call_ids, vec!["call_1"]);
}

#[tokio::test(flavor = "current_thread")]
async fn assistant_message_hook_error_stops_the_loop() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model.clone()).build();
    let agent_loop = AgentLoop::new(agent).on_assistant_message_finished(|_, _context| async {
        Err(std::io::Error::other("assistant hook failed"))
    });
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    let err = handle
        .wait()
        .await
        .expect_err("assistant message hook failure should fail the run");

    assert!(matches!(err, AgentLoopError::AssistantMessageHook(_)));
    assert_eq!(model.request_count(), 1);

    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::LoopFailed { error }
                if error.message.contains("assistant message hook failed: assistant hook failed")
        )
    }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentLoopEvent::LoopEnded { .. }))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn turn_hook_abort_stops_after_committing_the_turn() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("unused"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(|_, _turn| async {
        Ok::<_, std::io::Error>(TurnHookAction::Abort {
            reason: "waiting for human".to_string(),
        })
    });

    let handle = agent_loop.prompt(Message::user("start"));
    handle.follow_up(Message::user("follow-up")).unwrap();

    let result = handle.wait().await.unwrap();

    assert!(matches!(
        result.end_reason,
        EndReason::AbortedByHook { reason } if reason == "waiting for human"
    ));
    assert_eq!(user_texts(result.history.iter()), vec!["start"]);
}

#[tokio::test(flavor = "current_thread")]
async fn turn_hook_runs_for_abort_with_empty_append_batch() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model)
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
    let hook_seen = seen.clone();
    let agent_loop = AgentLoop::new(agent).with_turn_hook(move |_, turn| {
        let hook_seen = hook_seen.clone();
        async move {
            hook_seen
                .lock()
                .unwrap()
                .push((turn.history.len(), turn.new_messages.len()));
            Ok::<_, std::io::Error>(TurnHookAction::Continue)
        }
    });

    let handle = agent_loop.prompt(Message::user("start"));

    entered.notified().await;
    handle.abort().unwrap();
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("abort should finish without waiting for an unanswered tool")
        .unwrap();
    release.notify_one();

    assert_eq!(result.end_reason, EndReason::Aborted);
    assert_eq!(&seen.lock().unwrap()[..], &[(0, 0)]);
}

#[tokio::test(flavor = "current_thread")]
async fn turn_timeout_covers_stream_setup_and_runs_hook() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let turn_timeout = Duration::from_millis(50);
    let agent = AgentBuilder::new(BlockingStreamModel {
        entered: entered.clone(),
        release: release.clone(),
        dropped: Arc::new(Notify::new()),
    })
    .build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
    let hook_seen = seen.clone();
    let agent_loop = AgentLoop::new(agent)
        .turn_timeout(turn_timeout)
        .with_turn_hook(move |_, turn| {
            let hook_seen = hook_seen.clone();
            async move {
                hook_seen
                    .lock()
                    .unwrap()
                    .push((turn.history.len(), turn.new_messages.len()));
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        });

    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    entered.notified().await;
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("turn timeout should finish while stream setup is blocked")
        .unwrap();
    release.notify_one();

    assert_eq!(
        result.end_reason,
        EndReason::TurnTimedOut {
            timeout: turn_timeout
        }
    );
    assert!(result.history.is_empty());
    assert_eq!(&seen.lock().unwrap()[..], &[(0, 0)]);
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(event, AgentLoopEvent::TurnTimedOut { messages } if messages.is_empty())
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn turn_timeout_runs_hook_with_safe_partial_append() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let turn_timeout = Duration::from_millis(50);
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model)
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
    let hook_seen = seen.clone();
    let agent_loop = AgentLoop::new(agent)
        .turn_timeout(turn_timeout)
        .with_turn_hook(move |_, turn| {
            let hook_seen = hook_seen.clone();
            async move {
                hook_seen
                    .lock()
                    .unwrap()
                    .push((turn.history.len(), turn.new_messages.len()));
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        });

    let handle = agent_loop.prompt(Message::user("start"));

    entered.notified().await;
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("turn timeout should finish without waiting for tool completion")
        .unwrap();
    release.notify_one();

    assert_eq!(
        result.end_reason,
        EndReason::TurnTimedOut {
            timeout: turn_timeout
        }
    );
    assert!(result.history.is_empty());
    assert_eq!(&seen.lock().unwrap()[..], &[(0, 0)]);
}

#[tokio::test(flavor = "current_thread")]
async fn recovered_history_from_max_turns_commits_through_hook() {
    let model = MockCompletionModel::from_stream_turns([
        [MockStreamEvent::tool_call(
            "call_1",
            "add",
            serde_json::json!({"x": 1, "y": 2}),
        )],
        [MockStreamEvent::tool_call(
            "call_2",
            "add",
            serde_json::json!({"x": 3, "y": 4}),
        )],
        [MockStreamEvent::tool_call(
            "call_3",
            "add",
            serde_json::json!({"x": 5, "y": 6}),
        )],
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
    let hook_seen = seen.clone();
    let agent_loop = AgentLoop::new(agent)
        .max_turns(1)
        .with_turn_hook(move |_, turn| {
            let hook_seen = hook_seen.clone();
            async move {
                hook_seen
                    .lock()
                    .unwrap()
                    .push((turn.history.len(), turn.new_messages.len()));
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        });
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::MaxTurns { max_turns: 1 });
    let committed_len = result.history.len();
    assert!(committed_len >= 3);
    assert_eq!(&seen.lock().unwrap()[..], &[(committed_len, committed_len)]);
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
            matches!(event, AgentLoopEvent::TurnCommitted { messages } if messages.len() == committed_len)
        }));
}

#[tokio::test(flavor = "current_thread")]
async fn recovered_history_must_extend_committed_base() {
    let base = vec![Message::user("start"), Message::assistant("base")];
    let state = Arc::new(Mutex::new(base.clone()));
    let (events_tx, mut events) = broadcast::channel(EVENT_BUFFER_SIZE);
    let mut runner = test_runner(state.clone(), events_tx);

    for recovered_history in [
        vec![Message::user("different")],
        vec![Message::user("start"), Message::assistant("different")],
    ] {
        let err = runner
            .commit_recovered_history(
                &base,
                recovered_history,
                EndReason::Other {
                    message: "recovered".to_string(),
                },
            )
            .await
            .expect_err("non-appendable recovered history should fail");
        assert!(matches!(err, AgentLoopError::InvalidMessageHistory(_)));
        assert_eq!(*lock_messages(&state), base);
    }

    let events = drain_events(&mut events);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentLoopEvent::TurnCommitted { .. }))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn loop_timeout_wins_over_turn_timeout_during_active_turn() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let loop_timeout = Duration::from_millis(50);
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model)
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let agent_loop = AgentLoop::new(agent)
        .turn_timeout(Duration::from_secs(5))
        .loop_timeout(loop_timeout);

    let handle = agent_loop.prompt(Message::user("start"));

    entered.notified().await;
    let result = timeout(Duration::from_secs(1), handle.wait())
        .await
        .expect("loop timeout should finish while turn is blocked")
        .unwrap();
    release.notify_one();

    assert_eq!(
        result.end_reason,
        EndReason::LoopTimedOut {
            timeout: loop_timeout
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn loop_timeout_while_idle_does_not_run_turn_hook_again() {
    let loop_timeout = Duration::from_millis(50);
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
    let hook_seen = seen.clone();
    let agent_loop = AgentLoop::new(agent)
        .loop_timeout(loop_timeout)
        .with_turn_hook(move |_, turn| {
            let hook_seen = hook_seen.clone();
            async move {
                hook_seen
                    .lock()
                    .unwrap()
                    .push((turn.history.len(), turn.new_messages.len()));
                Ok::<_, std::io::Error>(TurnHookAction::Continue)
            }
        });
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    loop {
        match events.recv().await.unwrap() {
            AgentLoopEvent::TurnCommitted { .. } => break,
            AgentLoopEvent::LoopEnded { end_reason } => {
                panic!("loop ended before commit: {end_reason:?}")
            }
            _ => {}
        }
    }

    sleep(loop_timeout + Duration::from_millis(50)).await;
    let result = handle.wait().await.unwrap();

    assert_eq!(
        result.end_reason,
        EndReason::LoopTimedOut {
            timeout: loop_timeout
        }
    );
    assert_eq!(&seen.lock().unwrap()[..], &[(2, 2)]);
}

#[tokio::test(flavor = "current_thread")]
async fn provider_content_filter_error_returns_end_reason() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
        "content_filter: request blocked by policy",
    )]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert!(matches!(
        result.end_reason,
        EndReason::ContentFilter { error }
            if error.kind == ApiErrorKind::Provider
                && error.message == "content_filter: request blocked by policy"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn provider_context_full_error_returns_end_reason() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
        "context_length_exceeded: maximum context length exceeded",
    )]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert!(matches!(
        result.end_reason,
        EndReason::ContextFull { error }
            if error.kind == ApiErrorKind::Provider
                && error.message == "context_length_exceeded: maximum context length exceeded"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn provider_length_error_returns_end_reason() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
        "OpenAI response stream was incomplete: max_output_tokens",
    )]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert!(matches!(
        result.end_reason,
        EndReason::Length { error }
            if error.kind == ApiErrorKind::Provider
                && error.message == "OpenAI response stream was incomplete: max_output_tokens"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn provider_api_error_returns_end_reason() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error(
        "server_error: response stream failed",
    )]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);

    let result = agent_loop
        .prompt(Message::user("start"))
        .wait()
        .await
        .unwrap();

    assert!(matches!(
        result.end_reason,
        EndReason::ApiError { error }
            if error.kind == ApiErrorKind::Provider
                && error.message == "server_error: response stream failed"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn subscribe_forwards_rig_stream_items_with_tool_deltas() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call_name_delta("call_1", "internal_1", "add"),
        MockStreamEvent::tool_call_arguments_delta("call_1", "internal_1", r#"{"x":1,"y":2}"#),
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::Rig(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCallDelta {
                    content: ToolCallDeltaContent::Name(name),
                    ..
                }
            )) if name == "add"
        )
    }));
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::Rig(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCallDelta {
                    content: ToolCallDeltaContent::Delta(arguments),
                    ..
                }
            )) if arguments == r#"{"x":1,"y":2}"#
        )
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn subscribe_emits_queue_and_lifecycle_events() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("follow"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    handle.follow_up(Message::user("follow-up")).unwrap();
    let result = handle.wait().await.unwrap();

    assert_eq!(result.end_reason, EndReason::Idle);
    let events = drain_events(&mut events);
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::Queued {
                kind: QueueKind::FollowUp,
                ..
            }
        )
    }));
    assert!(events.iter().any(|event| {
        matches!(event, AgentLoopEvent::TurnCommitted { messages } if !messages.is_empty())
    }));
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentLoopEvent::LoopEnded {
                end_reason: EndReason::Idle
            }
        )
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn handle_state_returns_committed_messages_while_running() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("first"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let agent_loop = AgentLoop::new(agent);
    let handle = agent_loop.prompt(Message::user("start"));
    let mut events = handle.subscribe();

    loop {
        match events.recv().await.unwrap() {
            AgentLoopEvent::TurnCommitted { messages } => {
                assert_eq!(messages.len(), 2);
                let state = handle.state();
                assert_eq!(state.len(), 2);
                assert_eq!(first_user_text(&state[0]).as_deref(), Some("start"));
                break;
            }
            AgentLoopEvent::LoopEnded { .. } => panic!("loop ended before commit event"),
            _ => {}
        }
    }

    let result = handle.wait().await.unwrap();
    assert_eq!(result.history.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_drains_signals_submitted_before_start() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("first"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("steered"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let start = harness.follow_up("start").await.unwrap();
    let steer = harness.steer("steer").await.unwrap();
    harness.start().unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 2);
    let state = store.snapshot();
    assert_eq!(
        state
            .inbox
            .iter()
            .map(|entry| &entry.id)
            .collect::<Vec<_>>(),
        vec![&start.id, &steer.id]
    );
    assert_eq!(state.acked_inbox_ids, vec![start.id, steer.id]);
    assert_eq!(user_texts(state.persisted.iter()), vec!["start", "steer"]);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_wait_for_idle_reports_no_run_when_nothing_ran() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store);

    harness.start().unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(end_reason, EndReason::NoRun);
    assert_eq!(model.request_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_pause_while_idle_resolves_waiters_as_paused() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store);

    harness.start().unwrap();
    harness.pause().unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(end_reason, EndReason::Paused);
    assert_eq!(model.request_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_pause_mid_turn_leaves_queued_follow_up_submitted() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("queued follow-up"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("later follow-up"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone())
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let start = harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("tool should block the active durable turn");

    let queued = harness.follow_up("queued").await.unwrap();
    harness.pause().unwrap();
    release.notify_one();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(end_reason, EndReason::Paused);
    assert_eq!(model.request_count(), 2);

    let later = harness.follow_up("later").await.unwrap();
    let later_end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(later_end_reason, EndReason::Paused);
    assert_eq!(model.request_count(), 2);

    let state = store.snapshot();
    assert_eq!(
        state
            .inbox
            .iter()
            .map(|entry| &entry.id)
            .collect::<Vec<_>>(),
        vec![&start.id, &queued.id, &later.id]
    );
    assert!(state.acked_inbox_ids.contains(&start.id));
    assert!(!state.acked_inbox_ids.contains(&queued.id));
    assert!(!state.acked_inbox_ids.contains(&later.id));
    assert_eq!(user_texts(state.persisted.iter()), vec!["start"]);
    assert_eq!(
        state.turn_outcomes,
        vec![DurableTurnOutcome {
            kind: TurnOutcomeKind::Completed,
            end_reason: Some(EndReason::Paused),
            usage: Some(Usage::new()),
        }]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn in_memory_durable_store_tracks_pending_acks_tool_results_and_turn_outcomes() {
    let store = InMemoryDurableAgentStore::default();
    let message = tag_message_with_inbox_id(Message::user("start"), "inbox-1").unwrap();
    let entry = DurableInboxEntry {
        id: "inbox-1".to_string(),
        kind: DurableInboxKind::FollowUp,
        message: message.clone(),
        status: DurableInboxStatus::Submitted,
        submitted_at: tokio::time::Instant::now(),
    };
    let tool_result = test_tool_result("call_1");
    let tool_result_message = Message::User {
        content: OneOrMany::one(UserContent::ToolResult(tool_result.clone())),
    };

    store.submit_inbox_entry(entry).await.unwrap();
    assert_eq!(store.snapshot().pending_inbox_entries.len(), 1);

    store
        .persist_messages_and_ack(PersistMessagesArgs {
            messages: vec![message, tool_result_message],
            ack_inbox_entry_ids: vec!["inbox-1".to_string()],
            turn_outcome: Some(DurableTurnOutcome {
                kind: TurnOutcomeKind::Completed,
                end_reason: Some(EndReason::Idle),
                usage: Some(Usage::new()),
            }),
        })
        .await
        .unwrap();

    let snapshot = store.snapshot();
    assert!(snapshot.pending_inbox_entries.is_empty());
    assert_eq!(snapshot.acked_inbox_ids, vec!["inbox-1"]);
    assert_eq!(
        snapshot.acked_inbox_entries[0].status,
        DurableInboxStatus::Acked
    );
    assert_eq!(snapshot.persisted_messages.len(), 2);
    assert_eq!(
        snapshot.turn_outcomes,
        vec![DurableTurnOutcome {
            kind: TurnOutcomeKind::Completed,
            end_reason: Some(EndReason::Idle),
            usage: Some(Usage::new()),
        }]
    );
    assert_eq!(
        snapshot
            .tool_results
            .get(&ToolResultKey::new("call_1", None)),
        Some(&tool_result)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_started_manager_wakes_for_later_signal() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("late"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    harness.start().unwrap();
    assert_eq!(
        timeout(Duration::from_secs(1), harness.wait_for_idle())
            .await
            .unwrap()
            .unwrap(),
        EndReason::NoRun
    );

    let late = harness.follow_up("late").await.unwrap();
    assert_eq!(
        timeout(Duration::from_secs(1), harness.wait_for_idle())
            .await
            .unwrap()
            .unwrap(),
        EndReason::Idle
    );
    assert_eq!(
        timeout(Duration::from_secs(1), harness.wait_for_idle())
            .await
            .unwrap()
            .unwrap(),
        EndReason::NoRun
    );

    assert_eq!(model.request_count(), 1);
    let state = store.snapshot();
    assert_eq!(state.acked_inbox_ids, vec![late.id]);
    assert_eq!(user_texts(state.persisted.iter()), vec!["late"]);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_acks_multiple_interrupts_and_continues_to_follow_up() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("handled second interrupt"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("handled first interrupt"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("handled follow-up"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone())
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let start = harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("first turn should enter the blocking tool before interrupts");

    let first_interrupt = harness.interrupt("first interrupt").await.unwrap();
    let second_interrupt = harness.interrupt("second interrupt").await.unwrap();
    let after = harness.follow_up("after interrupts").await.unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    release.notify_waiters();

    assert_eq!(end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 4);

    let state = store.snapshot();
    assert_eq!(
        state
            .inbox
            .iter()
            .map(|entry| &entry.id)
            .collect::<Vec<_>>(),
        vec![
            &start.id,
            &first_interrupt.id,
            &second_interrupt.id,
            &after.id
        ]
    );

    for id in [
        &start.id,
        &first_interrupt.id,
        &second_interrupt.id,
        &after.id,
    ] {
        assert!(
            state.acked_inbox_ids.contains(id),
            "expected inbox entry {id} to be acked"
        );
        assert!(
            state.transactions.iter().any(|transaction| {
                transaction.ack_inbox_entry_ids.contains(id)
                    && transaction
                        .messages
                        .iter()
                        .any(|message| inbox_id_from_message(message).as_ref() == Some(id))
            }),
            "expected inbox entry {id} to be acked in the same transaction as its message"
        );
    }

    validate_message_history(&state.persisted).unwrap();
    assert_eq!(
        tool_result_texts(state.persisted.iter()),
        vec![DEFAULT_TOOL_REPAIR_MESSAGE]
    );
    assert_eq!(
        user_texts(state.persisted.iter()),
        vec![
            "start",
            "second interrupt",
            "first interrupt",
            "after interrupts"
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_interrupt_repair_prefers_persisted_tool_result() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("handled interrupt"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = MemoryDurableAgentStore::default();
    store.insert_tool_result(
        ToolResultKey::new("call_1", None),
        test_tool_result_with_text("call_1", "persisted tool result"),
    );
    let agent = AgentBuilder::new(model.clone())
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("first turn should enter the blocking tool before interrupt");

    let interrupt = harness.interrupt("interrupt").await.unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    release.notify_waiters();

    assert_eq!(end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 2);
    let state = store.snapshot();
    validate_message_history(&state.persisted).unwrap();
    assert_eq!(
        tool_result_texts(state.persisted.iter()),
        vec!["persisted tool result"]
    );
    assert!(state.acked_inbox_ids.contains(&interrupt.id));
    assert_eq!(
        state.turn_outcomes,
        vec![
            DurableTurnOutcome {
                kind: TurnOutcomeKind::Interrupted,
                end_reason: None,
                usage: None,
            },
            DurableTurnOutcome {
                kind: TurnOutcomeKind::Completed,
                end_reason: Some(EndReason::Idle),
                usage: Some(Usage::new()),
            },
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_checkpoint_repair_failure_is_sticky() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::tool_call("call_1", BlockingTool::NAME, serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        [
            MockStreamEvent::text("unused"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone())
        .tool(BlockingTool {
            entered: entered.clone(),
            release: release.clone(),
        })
        .build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), entered.notified())
        .await
        .expect("first turn should enter the blocking tool before interrupt");
    store.fail_load_tool_result();

    let interrupt = harness.interrupt("interrupt").await.unwrap();
    let err = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .expect_err("tool-result load failure during repair should fail the run");
    assert!(err.to_string().contains("tool result load failed"));

    let later_err = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .expect_err("repair failure should be remembered for later waiters");
    release.notify_waiters();

    assert!(later_err.to_string().contains("tool result load failed"));
    let state = store.snapshot();
    assert!(!state.acked_inbox_ids.contains(&interrupt.id));
    assert_eq!(user_texts(state.persisted.iter()), vec!["start"]);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_multiple_waiters_receive_same_idle_result() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let entry = harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    let (first, second) = tokio::join!(
        timeout(Duration::from_secs(1), harness.wait_for_idle()),
        timeout(Duration::from_secs(1), harness.wait_for_idle())
    );

    assert_eq!(first.unwrap().unwrap(), EndReason::Idle);
    assert_eq!(second.unwrap().unwrap(), EndReason::Idle);
    assert_eq!(model.request_count(), 1);
    assert_eq!(store.snapshot().acked_inbox_ids, vec![entry.id]);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_rejects_invalid_signal_before_submit() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let assistant_err = harness
        .follow_up(Message::assistant("not a user signal"))
        .await
        .expect_err("assistant messages should not be durable signals");
    assert!(assistant_err.to_string().contains("must be user messages"));

    let tool_only_err = harness
        .follow_up(Message::User {
            content: OneOrMany::one(UserContent::ToolResult(test_tool_result("call_1"))),
        })
        .await
        .expect_err("user messages without text cannot be tagged");
    assert!(
        tool_only_err
            .to_string()
            .contains("must contain text content")
    );

    assert!(store.snapshot().inbox.is_empty());
    harness.start().unwrap();
    assert_eq!(
        timeout(Duration::from_secs(1), harness.wait_for_idle())
            .await
            .unwrap()
            .unwrap(),
        EndReason::NoRun
    );
    assert_eq!(model.request_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_message_persist_failure_does_not_ack_and_is_sticky() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store.clone());

    let entry = harness.follow_up("start").await.unwrap();
    store.fail_persist_messages();
    harness.start().unwrap();
    let err = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .expect_err("message persistence failure should fail the run");
    assert!(err.to_string().contains("message persist failed"));

    let later_err = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .expect_err("message persistence failure should be remembered");
    assert!(later_err.to_string().contains("message persist failed"));

    let state = store.snapshot();
    assert_eq!(
        state
            .inbox
            .iter()
            .map(|entry| &entry.id)
            .collect::<Vec<_>>(),
        vec![&entry.id]
    );
    assert!(state.persisted.is_empty());
    assert!(state.acked_inbox_ids.is_empty());
    assert!(state.transactions.is_empty());
    assert_eq!(model.request_count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_restarts_from_partial_assistant_snapshot_with_kv_repair() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("recovered"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    store.insert_tool_result(
        ToolResultKey::new("call_1", None),
        test_tool_result_with_text("call_1", "persisted restart result"),
    );
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store.clone())
        .with_history([assistant_tool_call_message("call_1")]);

    harness.start().unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(end_reason, EndReason::Idle);
    assert_eq!(model.request_count(), 1);
    let state = store.snapshot();
    validate_message_history(
        &[
            vec![assistant_tool_call_message("call_1")],
            state.persisted.clone(),
        ]
        .concat(),
    )
    .unwrap();
    assert_eq!(
        tool_result_texts(state.persisted.iter()),
        vec!["persisted restart result"]
    );
    assert_eq!(assistant_texts(state.persisted.iter()), vec!["recovered"]);
    assert!(state.acked_inbox_ids.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_runs_checkpoint_handler_after_persistence() {
    let usage = Usage {
        input_tokens: 123,
        output_tokens: 45,
        total_tokens: 168,
        ..Usage::new()
    };
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response(usage),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let seen_persisted_lengths = Arc::new(Mutex::new(Vec::new()));
    let seen_end_reasons = Arc::new(Mutex::new(Vec::new()));
    let seen_usage = Arc::new(Mutex::new(Vec::new()));
    let handler_lengths = seen_persisted_lengths.clone();
    let handler_end_reasons = seen_end_reasons.clone();
    let handler_usage = seen_usage.clone();
    let handler_store = store.clone();
    let agent = AgentBuilder::new(model).build();
    let harness =
        DurableAgentHarness::new(agent, store.clone()).with_checkpoint_handler(move |checkpoint| {
            let handler_store = handler_store.clone();
            let handler_lengths = handler_lengths.clone();
            let handler_end_reasons = handler_end_reasons.clone();
            let handler_usage = handler_usage.clone();
            async move {
                match checkpoint {
                    DurableCheckpoint::AssistantMessageFinished(context) => {
                        handler_end_reasons.lock().unwrap().push(context.end_reason);
                        handler_usage.lock().unwrap().push(context.usage);
                    }
                    DurableCheckpoint::TurnBoundary(turn) => {
                        handler_lengths
                            .lock()
                            .unwrap()
                            .push(handler_store.snapshot().persisted.len());
                        handler_end_reasons.lock().unwrap().push(turn.end_reason);
                        handler_usage.lock().unwrap().push(turn.usage);
                    }
                }
                Ok::<_, std::convert::Infallible>(DurableCheckpointAction::Continue)
            }
        });

    harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(&seen_persisted_lengths.lock().unwrap()[..], &[2]);
    assert_eq!(
        &seen_end_reasons.lock().unwrap()[..],
        &[Some(EndReason::Idle), Some(EndReason::Idle)]
    );
    assert_eq!(&seen_usage.lock().unwrap()[..], &[Some(usage), Some(usage)]);

    let expected_outcome = DurableTurnOutcome {
        kind: TurnOutcomeKind::Completed,
        end_reason: Some(EndReason::Idle),
        usage: Some(usage),
    };
    let state = store.snapshot();
    assert_eq!(state.turn_outcomes, vec![expected_outcome.clone()]);
    assert!(state.transactions.iter().any(|transaction| {
        transaction.messages.is_empty()
            && transaction.ack_inbox_entry_ids.is_empty()
            && transaction.turn_outcome.as_ref() == Some(&expected_outcome)
    }));
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_checkpoint_handler_can_abort_after_assistant_snapshot() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("summary-worthy answer"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let compaction_required = Arc::new(AtomicBool::new(false));
    let agent = AgentBuilder::new(model).build();
    let harness = DurableAgentHarness::new(agent, store.clone());
    let handler_flag = compaction_required.clone();
    let harness = harness.with_checkpoint_handler(move |checkpoint| {
        let handler_flag = handler_flag.clone();
        async move {
            if matches!(checkpoint, DurableCheckpoint::AssistantMessageFinished(_)) {
                handler_flag.store(true, Ordering::SeqCst);
                return Ok::<_, DurableAgentError>(DurableCheckpointAction::Abort {
                    reason: "compaction requested".to_string(),
                });
            }
            Ok(DurableCheckpointAction::Continue)
        }
    });

    let start = harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    let end_reason = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(
        end_reason,
        EndReason::AbortedByHook { reason } if reason == "compaction requested"
    ));
    assert!(compaction_required.load(Ordering::SeqCst));
    let state = store.snapshot();
    assert_eq!(
        assistant_texts(state.persisted.iter()),
        vec!["summary-worthy answer"]
    );
    assert_eq!(state.acked_inbox_ids, vec![start.id]);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_submit_failure_does_not_signal_agent() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("unused"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    store.fail_submit();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness = DurableAgentHarness::new(agent, store);

    let err = harness
        .follow_up("start")
        .await
        .expect_err("submit failure should reject the signal");

    assert!(err.to_string().contains("submit failed"));
    harness.start().unwrap();
    timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(model.request_count(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn durable_harness_remembers_checkpoint_handler_failure_for_later_wait() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let store = MemoryDurableAgentStore::default();
    let handler_started = Arc::new(Notify::new());
    let handler_notify = handler_started.clone();
    let agent = AgentBuilder::new(model.clone()).build();
    let harness =
        DurableAgentHarness::new(agent, store).with_checkpoint_handler(move |checkpoint| {
            let handler_notify = handler_notify.clone();
            async move {
                if matches!(checkpoint, DurableCheckpoint::TurnBoundary(_)) {
                    handler_notify.notify_one();
                    return Err(std::io::Error::other("handler failed"));
                }
                Ok(DurableCheckpointAction::Continue)
            }
        });

    harness.follow_up("start").await.unwrap();
    harness.start().unwrap();
    timeout(Duration::from_secs(1), handler_started.notified())
        .await
        .unwrap();

    let err = timeout(Duration::from_secs(1), harness.wait_for_idle())
        .await
        .unwrap()
        .expect_err("checkpoint handler failure should be remembered for later waiters");

    assert!(err.to_string().contains("handler failed"));
    assert_eq!(model.request_count(), 1);
}

#[test]
fn partial_turn_commits_completed_tool_results() {
    let mut partial = PartialTurn::default();
    let tool_call = test_tool_call("call_1");
    let tool_result = test_tool_result("call_1");

    partial.note_assistant_item(
        StreamedAssistantContent::<rig::test_utils::MockResponse>::ToolCall {
            internal_call_id: "internal_1".to_string(),
            tool_call,
        },
    );
    partial.note_user_item(StreamedUserContent::ToolResult {
        internal_call_id: "internal_1".to_string(),
        tool_result,
    });

    let messages = partial
        .append_messages(&Message::user("start"))
        .expect("completed tool results should produce history");

    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[1], Message::Assistant { .. }));
    assert!(
        matches!(&messages[2], Message::User { content } if matches!(content.first(), UserContent::ToolResult(_)))
    );
}

#[test]
fn partial_turn_preserves_text_before_completed_tool_result() {
    let mut partial = PartialTurn::default();

    partial.note_assistant_item(
        StreamedAssistantContent::<rig::test_utils::MockResponse>::Text(rig::message::Text::new(
            "I will call the tool.",
        )),
    );
    partial.note_assistant_item(
        StreamedAssistantContent::<rig::test_utils::MockResponse>::ToolCall {
            internal_call_id: "internal_1".to_string(),
            tool_call: test_tool_call("call_1"),
        },
    );
    partial.note_user_item(StreamedUserContent::ToolResult {
        internal_call_id: "internal_1".to_string(),
        tool_result: test_tool_result("call_1"),
    });

    let messages = partial
        .append_messages(&Message::user("start"))
        .expect("completed tool results should produce history");
    let Message::Assistant { content, .. } = &messages[1] else {
        panic!("expected assistant message");
    };
    let items = content.iter().collect::<Vec<_>>();

    assert!(
        matches!(items[0], AssistantContent::Text(text) if text.text == "I will call the tool.")
    );
    assert!(matches!(items[1], AssistantContent::ToolCall(tool_call) if tool_call.id == "call_1"));
}

#[test]
fn partial_turn_ignores_unanswered_tool_call() {
    let mut partial = PartialTurn::default();

    partial.note_assistant_item(
        StreamedAssistantContent::<rig::test_utils::MockResponse>::ToolCall {
            internal_call_id: "internal_1".to_string(),
            tool_call: test_tool_call("call_1"),
        },
    );

    assert!(partial.append_messages(&Message::user("start")).is_none());
}

fn user_prompts(model: &MockCompletionModel) -> Vec<String> {
    request_user_prompts(&model.requests())
}

fn request_user_prompts(requests: &[CompletionRequest]) -> Vec<String> {
    requests
        .iter()
        .map(|request| {
            request
                .chat_history
                .iter()
                .filter_map(first_user_text)
                .last()
                .unwrap_or_default()
        })
        .collect()
}

fn user_texts<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<String> {
    messages.into_iter().filter_map(first_user_text).collect()
}

fn assistant_texts<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<String> {
    messages
        .into_iter()
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

fn tool_result_texts<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<String> {
    messages
        .into_iter()
        .flat_map(|message| {
            let Message::User { content } = message else {
                return Vec::new();
            };

            content
                .iter()
                .filter_map(|item| {
                    let UserContent::ToolResult(tool_result) = item else {
                        return None;
                    };

                    tool_result
                        .content
                        .iter()
                        .find_map(|content| match content {
                            ToolResultContent::Text(text) => Some(text.text.clone()),
                            ToolResultContent::Image(_) => None,
                        })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn tool_results<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<ToolResult> {
    messages
        .into_iter()
        .flat_map(|message| {
            let Message::User { content } = message else {
                return Vec::new();
            };

            content
                .iter()
                .filter_map(|item| match item {
                    UserContent::ToolResult(tool_result) => Some(tool_result.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn tool_result_text(tool_result: &ToolResult) -> Option<String> {
    tool_result
        .content
        .iter()
        .find_map(|content| match content {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            ToolResultContent::Image(_) => None,
        })
}

fn first_user_text(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };

    content.iter().find_map(|item| match item {
        UserContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

fn assistant_tool_call_message(id: &str) -> Message {
    Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::ToolCall(test_tool_call(id))),
    }
}

fn test_tool_call(id: &str) -> ToolCall {
    ToolCall::new(
        id.to_string(),
        rig::message::ToolFunction::new("echo".to_string(), serde_json::json!({"text": "hi"})),
    )
}

fn test_tool_result(id: &str) -> ToolResult {
    test_tool_result_with_text(id, "echoed")
}

fn test_tool_result_with_text(id: &str, text: &str) -> ToolResult {
    ToolResult {
        id: id.to_string(),
        call_id: None,
        content: rig::message::ToolResultContent::from_tool_output(text.to_string()),
    }
}

#[derive(Clone, Debug, Default)]
struct MemoryDurableAgentStore {
    inner: Arc<Mutex<MemoryDurableAgentStoreState>>,
    fail_submit: Arc<AtomicBool>,
    fail_persist_messages: Arc<AtomicBool>,
    fail_persist_tool_result: Arc<AtomicBool>,
    fail_load_tool_result: Arc<AtomicBool>,
}

#[derive(Clone, Debug, Default)]
struct MemoryDurableAgentStoreState {
    inbox: Vec<DurableInboxEntry>,
    persisted: Vec<Message>,
    acked_inbox_ids: Vec<String>,
    transactions: Vec<PersistMessagesArgs>,
    turn_outcomes: Vec<DurableTurnOutcome>,
    tool_results: BTreeMap<ToolResultKey, ToolResult>,
}

impl MemoryDurableAgentStore {
    fn snapshot(&self) -> MemoryDurableAgentStoreState {
        self.inner.lock().unwrap().clone()
    }

    fn insert_tool_result(&self, key: ToolResultKey, result: ToolResult) {
        self.inner.lock().unwrap().tool_results.insert(key, result);
    }

    fn fail_submit(&self) {
        self.fail_submit.store(true, Ordering::SeqCst);
    }

    fn fail_persist_messages(&self) {
        self.fail_persist_messages.store(true, Ordering::SeqCst);
    }

    fn fail_load_tool_result(&self) {
        self.fail_load_tool_result.store(true, Ordering::SeqCst);
    }
}

impl IncrementalToolResultPersistence for MemoryDurableAgentStore {
    fn persist_tool_result(
        &self,
        key: ToolResultKey,
        result: ToolResult,
    ) -> ToolResultPersistenceFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            if store.fail_persist_tool_result.load(Ordering::SeqCst) {
                return Err(
                    Box::new(std::io::Error::other("tool result persist failed"))
                        as ToolResultPersistenceError,
                );
            }
            store.inner.lock().unwrap().tool_results.insert(key, result);
            Ok(())
        })
    }

    fn load_tool_result(
        &self,
        key: ToolResultKey,
    ) -> ToolResultPersistenceFuture<Option<ToolResult>> {
        let store = self.clone();
        Box::pin(async move {
            if store.fail_load_tool_result.load(Ordering::SeqCst) {
                return Err(Box::new(std::io::Error::other("tool result load failed"))
                    as ToolResultPersistenceError);
            }
            Ok(store.inner.lock().unwrap().tool_results.get(&key).cloned())
        })
    }
}

impl DurableAgentStore for MemoryDurableAgentStore {
    fn submit_inbox_entry(&self, entry: DurableInboxEntry) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            if store.fail_submit.load(Ordering::SeqCst) {
                return Err(
                    Box::new(std::io::Error::other("submit failed")) as DurableAgentStoreError
                );
            }
            store.inner.lock().unwrap().inbox.push(entry);
            Ok(())
        })
    }

    fn persist_messages_and_ack(&self, args: PersistMessagesArgs) -> DurableAgentFuture<()> {
        let store = self.clone();
        Box::pin(async move {
            if store.fail_persist_messages.load(Ordering::SeqCst) {
                return Err(Box::new(std::io::Error::other("message persist failed"))
                    as DurableAgentStoreError);
            }
            let mut state = store.inner.lock().unwrap();
            state.transactions.push(args.clone());
            if let Some(turn_outcome) = &args.turn_outcome {
                state.turn_outcomes.push(turn_outcome.clone());
            }
            state.persisted.extend(args.messages);
            state.acked_inbox_ids.extend(args.ack_inbox_entry_ids);
            Ok(())
        })
    }
}

#[derive(Clone, Default)]
struct MemoryToolResultPersistence {
    results: Arc<Mutex<BTreeMap<ToolResultKey, ToolResult>>>,
    fail_persist: Arc<AtomicBool>,
    fail_load: Arc<AtomicBool>,
}

impl MemoryToolResultPersistence {
    fn insert(&self, key: ToolResultKey, result: ToolResult) {
        self.results.lock().unwrap().insert(key, result);
    }

    fn result(&self, key: ToolResultKey) -> Option<ToolResult> {
        self.results.lock().unwrap().get(&key).cloned()
    }

    fn fail_persist(&self) {
        self.fail_persist
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn fail_load(&self) {
        self.fail_load
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl IncrementalToolResultPersistence for MemoryToolResultPersistence {
    fn persist_tool_result(
        &self,
        key: ToolResultKey,
        result: ToolResult,
    ) -> ToolResultPersistenceFuture<()> {
        let results = self.results.clone();
        let fail_persist = self.fail_persist.clone();
        Box::pin(async move {
            if fail_persist.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(
                    Box::new(std::io::Error::other("persist failed")) as ToolResultPersistenceError
                );
            }

            results.lock().unwrap().insert(key, result);
            Ok(())
        })
    }

    fn load_tool_result(
        &self,
        key: ToolResultKey,
    ) -> ToolResultPersistenceFuture<Option<ToolResult>> {
        let results = self.results.clone();
        let fail_load = self.fail_load.clone();
        Box::pin(async move {
            if fail_load.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(
                    Box::new(std::io::Error::other("load failed")) as ToolResultPersistenceError
                );
            }

            Ok(results.lock().unwrap().get(&key).cloned())
        })
    }
}

#[derive(Clone)]
struct BlockingTool {
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

#[derive(Debug)]
struct BlockingToolError;

impl fmt::Display for BlockingToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("blocking tool failed")
    }
}

impl Error for BlockingToolError {}

impl Tool for BlockingTool {
    const NAME: &'static str = "blocking_tool";

    type Error = BlockingToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Blocks until the test releases it".to_string(),
            parameters: serde_json::json!({ "type": "object" }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok("released".to_string())
    }
}

#[derive(Clone)]
struct BlockingStreamModel {
    entered: Arc<Notify>,
    release: Arc<Notify>,
    dropped: Arc<Notify>,
}

impl CompletionModel for BlockingStreamModel {
    type Response = MockResponse;
    type StreamingResponse = MockResponse;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self {
            entered: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
            dropped: Arc::new(Notify::new()),
        }
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        Err(CompletionError::ProviderError(
            "blocking stream model does not support non-streaming completion".to_string(),
        ))
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let _drop = NotifyOnDrop(self.dropped.clone());
        self.entered.notify_one();
        self.release.notified().await;
        let stream: rig::streaming::StreamingResult<Self::StreamingResponse> =
            Box::pin(futures::stream::empty::<
                Result<RawStreamingChoice<MockResponse>, CompletionError>,
            >());
        Ok(StreamingCompletionResponse::stream(stream))
    }
}

struct NotifyOnDrop(Arc<Notify>);

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

fn test_runner(
    state: SharedMessages,
    events_tx: broadcast::Sender<AgentLoopEvent<MockResponse>>,
) -> Runner<MockCompletionModel, (), ()> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = Arc::new(AgentBuilder::new(model).build());
    Runner::new(RunnerInit {
        agent,
        app_state: Arc::new(()),
        max_turns: DEFAULT_MAX_TURNS,
        turn_timeout: None,
        loop_timeout: None,
        turn_hook: None,
        assistant_message_hook: None,
        unanswered_tool_call_repair: Some(DEFAULT_TOOL_REPAIR_MESSAGE.to_string()),
        tool_result_persistence: None,
        events_tx,
        state,
        turn_running: Arc::new(AtomicBool::new(false)),
        initial_turn: PendingTurn::Resume,
    })
}

fn drain_events<R: Clone>(
    events: &mut broadcast::Receiver<AgentLoopEvent<R>>,
) -> Vec<AgentLoopEvent<R>> {
    let mut drained = Vec::new();
    while let Ok(event) = events.try_recv() {
        drained.push(event);
    }
    drained
}
