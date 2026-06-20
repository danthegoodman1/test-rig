use std::{
    collections::VecDeque,
    future::IntoFuture,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use futures::StreamExt;
use rig::{
    agent::{Agent, MultiTurnStreamItem, PromptHook, StreamingError},
    completion::{CompletionError, CompletionModel, GetTokenUsage, PromptError, Usage},
    message::{Message, ToolResult},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt},
    wasm_compat::WasmCompatSend,
};
use tokio::{
    sync::{
        broadcast,
        mpsc::{self, error::TryRecvError},
    },
    task::JoinHandle,
    time::{Instant, sleep_until},
};

use crate::{
    api::{
        AgentLoopError, AgentLoopErrorInfo, AgentLoopEvent, AgentLoopResult, ApiErrorInfo,
        ApiErrorKind, AssistantMessageContext, AssistantMessageHook, AssistantMessageHookFuture,
        Command, EndReason, IncrementalToolResultPersistence, InvalidMessageHistoryError,
        SharedMessages, SharedTurnRunning, ToolResultKey, TurnHook, TurnHookAction,
        TurnHookContext, lock_messages,
    },
    history::{
        PartialTurn, repair_unanswered_tool_calls_with_persistence, validate_message_history,
        validate_resume_history,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimeoutKind {
    Turn,
    Loop,
}

#[derive(Clone, Copy, Debug)]
struct TimeoutDeadline {
    kind: TimeoutKind,
    at: Instant,
    timeout: Duration,
}

impl TimeoutDeadline {
    fn new(kind: TimeoutKind, timeout: Duration) -> Self {
        Self {
            kind,
            at: Instant::now() + timeout,
            timeout,
        }
    }

    fn end_reason(self) -> EndReason {
        match self.kind {
            TimeoutKind::Turn => EndReason::TurnTimedOut {
                timeout: self.timeout,
            },
            TimeoutKind::Loop => EndReason::LoopTimedOut {
                timeout: self.timeout,
            },
        }
    }
}

fn next_timeout(
    turn: Option<TimeoutDeadline>,
    loop_deadline: Option<TimeoutDeadline>,
) -> Option<TimeoutDeadline> {
    match (turn, loop_deadline) {
        (Some(turn), Some(loop_deadline)) if loop_deadline.at <= turn.at => Some(loop_deadline),
        (Some(turn), Some(_)) => Some(turn),
        (Some(turn), None) => Some(turn),
        (None, Some(loop_deadline)) => Some(loop_deadline),
        (None, None) => None,
    }
}

fn elapsed_timeout(deadline: Option<TimeoutDeadline>) -> Option<TimeoutDeadline> {
    deadline.filter(|deadline| Instant::now() >= deadline.at)
}

async fn wait_for_timeout(deadline: Option<TimeoutDeadline>) {
    if let Some(deadline) = deadline {
        sleep_until(deadline.at).await;
    } else {
        std::future::pending::<()>().await;
    }
}

async fn poll_assistant_message_hook(
    hook: &mut Option<AssistantMessageHookTask>,
) -> Result<(), AgentLoopError> {
    match hook {
        Some(hook) => match (&mut hook.handle).await {
            Ok(result) => result,
            Err(err) => Err(AgentLoopError::TaskJoin(err)),
        },
        None => std::future::pending().await,
    }
}

async fn wait_for_assistant_message_hook(
    hook: &mut Option<AssistantMessageHookTask>,
) -> Result<(), AgentLoopError> {
    if let Some(mut hook) = hook.take() {
        match (&mut hook.handle).await {
            Ok(result) => result?,
            Err(err) => return Err(AgentLoopError::TaskJoin(err)),
        }
    }

    Ok(())
}

struct AssistantMessageHookTask {
    handle: JoinHandle<Result<(), AgentLoopError>>,
}

impl AssistantMessageHookTask {
    fn spawn(hook: AssistantMessageHookFuture) -> Self {
        Self {
            handle: tokio::spawn(async move {
                hook.await.map_err(AgentLoopError::AssistantMessageHook)
            }),
        }
    }
}

impl Drop for AssistantMessageHookTask {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

pub(crate) struct Runner<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    agent: Arc<Agent<M, P>>,
    app_state: Arc<S>,
    max_turns: usize,
    turn_timeout: Option<Duration>,
    loop_timeout: Option<Duration>,
    turn_hook: Option<TurnHook<S>>,
    assistant_message_hook: Option<AssistantMessageHook<S>>,
    unanswered_tool_call_repair: Option<String>,
    tool_result_persistence: Option<Arc<dyn IncrementalToolResultPersistence>>,
    events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
    state: SharedMessages,
    turn_running: SharedTurnRunning,
    immediate: VecDeque<Message>,
    steering: VecDeque<Message>,
    resumes: usize,
    follow_ups: VecDeque<Message>,
    finish_when_idle: bool,
    last_response: Option<String>,
}

pub(crate) struct RunnerInit<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    pub(crate) agent: Arc<Agent<M, P>>,
    pub(crate) app_state: Arc<S>,
    pub(crate) max_turns: usize,
    pub(crate) turn_timeout: Option<Duration>,
    pub(crate) loop_timeout: Option<Duration>,
    pub(crate) turn_hook: Option<TurnHook<S>>,
    pub(crate) assistant_message_hook: Option<AssistantMessageHook<S>>,
    pub(crate) unanswered_tool_call_repair: Option<String>,
    pub(crate) tool_result_persistence: Option<Arc<dyn IncrementalToolResultPersistence>>,
    pub(crate) events_tx: broadcast::Sender<AgentLoopEvent<M::StreamingResponse>>,
    pub(crate) state: SharedMessages,
    pub(crate) turn_running: SharedTurnRunning,
    pub(crate) initial_turn: PendingTurn,
}

impl<M, P, S> Runner<M, P, S>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    pub(crate) fn new(init: RunnerInit<M, P, S>) -> Self {
        let (immediate, resumes) = match init.initial_turn {
            PendingTurn::Prompt { prelude, prompt } => {
                debug_assert!(prelude.is_empty());
                (VecDeque::from([*prompt]), 0)
            }
            PendingTurn::Resume => (VecDeque::new(), 1),
        };

        Self {
            agent: init.agent,
            app_state: init.app_state,
            max_turns: init.max_turns,
            turn_timeout: init.turn_timeout,
            loop_timeout: init.loop_timeout,
            turn_hook: init.turn_hook,
            assistant_message_hook: init.assistant_message_hook,
            unanswered_tool_call_repair: init.unanswered_tool_call_repair,
            tool_result_persistence: init.tool_result_persistence,
            events_tx: init.events_tx,
            state: init.state,
            turn_running: init.turn_running,
            immediate,
            steering: VecDeque::new(),
            resumes,
            follow_ups: VecDeque::new(),
            finish_when_idle: false,
            last_response: None,
        }
    }

    fn next_turn(&mut self) -> Option<PendingTurn> {
        if let Some(message) = self.immediate.pop_front() {
            return Some(PendingTurn::single(message));
        }

        if !self.steering.is_empty() {
            return PendingTurn::new(self.steering.drain(..).collect());
        }

        if self.resumes > 0 {
            self.resumes -= 1;
            return Some(PendingTurn::Resume);
        }

        self.follow_ups.pop_front().map(PendingTurn::single)
    }

    fn has_pending_turn(&self) -> bool {
        !self.immediate.is_empty()
            || !self.steering.is_empty()
            || self.resumes > 0
            || !self.follow_ups.is_empty()
    }

    fn handle_idle_command(&mut self, command: Command) -> CommandAction {
        match command {
            Command::Steer(message) => self.steering.push_back(message),
            Command::FollowUp(message) => self.follow_ups.push_back(message),
            Command::Resume => self.resumes += 1,
            Command::Interrupt(message) => self.immediate.push_front(message),
            Command::Abort => return CommandAction::Abort,
            Command::FinishWhenIdle => self.finish_when_idle = true,
        }

        CommandAction::Continue
    }

    fn drain_ready_commands(
        &mut self,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
    ) -> CommandAction {
        loop {
            match commands_rx.try_recv() {
                Ok(command) => {
                    if self.handle_idle_command(command) == CommandAction::Abort {
                        return CommandAction::Abort;
                    }
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                    return CommandAction::Continue;
                }
            }
        }
    }
}

pub(crate) enum PendingTurn {
    Prompt {
        prelude: Vec<Message>,
        prompt: Box<Message>,
    },
    Resume,
}

impl PendingTurn {
    fn new(mut messages: Vec<Message>) -> Option<Self> {
        let prompt = messages.pop()?;
        Some(Self::Prompt {
            prelude: messages,
            prompt: Box::new(prompt),
        })
    }

    pub(crate) fn single(prompt: Message) -> Self {
        Self::Prompt {
            prelude: Vec::new(),
            prompt: Box::new(prompt),
        }
    }
}

struct PreparedTurn {
    committed_base_history: Vec<Message>,
    request_history: Vec<Message>,
    prelude: Vec<Message>,
    prompt: Message,
    commit_prompt: bool,
}

impl PreparedTurn {
    fn from_pending(
        turn: PendingTurn,
        committed_base_history: Vec<Message>,
    ) -> Result<Self, AgentLoopError> {
        match turn {
            PendingTurn::Prompt { prelude, prompt } => {
                let prompt = *prompt;
                let mut request_history = committed_base_history.clone();
                request_history.extend(prelude.clone());
                let mut full_request_history = request_history.clone();
                full_request_history.push(prompt.clone());
                validate_message_history(&full_request_history)
                    .map_err(AgentLoopError::InvalidMessageHistory)?;

                Ok(Self {
                    committed_base_history,
                    request_history,
                    prelude,
                    prompt,
                    commit_prompt: true,
                })
            }
            PendingTurn::Resume => {
                validate_resume_history(&committed_base_history)
                    .map_err(AgentLoopError::InvalidMessageHistory)?;
                let Some((prompt, history)) = committed_base_history.split_last() else {
                    unreachable!("validated resume history is non-empty");
                };

                Ok(Self {
                    committed_base_history: committed_base_history.clone(),
                    request_history: history.to_vec(),
                    prelude: Vec::new(),
                    prompt: prompt.clone(),
                    commit_prompt: false,
                })
            }
        }
    }

    fn append_messages(&self, mut messages: Vec<Message>) -> Vec<Message> {
        if !self.commit_prompt && !messages.is_empty() {
            messages.remove(0);
        }

        if self.prelude.is_empty() {
            return messages;
        }

        let mut append = self.prelude.clone();
        append.append(&mut messages);
        append
    }

    fn partial_messages(&self, partial_turn: &PartialTurn) -> Vec<Message> {
        partial_turn
            .append_messages(&self.prompt)
            .map(|messages| self.append_messages(messages))
            .unwrap_or_default()
    }

    fn assistant_messages(&self, partial_turn: &PartialTurn, message: Message) -> Vec<Message> {
        let mut messages = Vec::new();
        if !self.prelude.is_empty() {
            messages.extend(self.prelude.clone());
        }
        if self.commit_prompt {
            messages.push(self.prompt.clone());
        }
        messages.extend(partial_turn.completed_messages.clone());
        messages.push(message);
        messages
    }
}

impl<M, P, S> Runner<M, P, S>
where
    M: CompletionModel + Send + Sync + 'static,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
    P: PromptHook<M> + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    pub(crate) async fn run(
        mut self,
        mut commands_rx: mpsc::UnboundedReceiver<Command>,
    ) -> Result<AgentLoopResult, AgentLoopError> {
        let result = self.run_inner(&mut commands_rx).await;
        if let Err(err) = &result {
            self.set_turn_running(false);
            self.emit(AgentLoopEvent::LoopFailed {
                error: AgentLoopErrorInfo::from_error(err),
            });
        }

        result
    }

    async fn run_inner(
        &mut self,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
    ) -> Result<AgentLoopResult, AgentLoopError> {
        self.emit(AgentLoopEvent::LoopStarted);
        self.repair_initial_history().await?;
        let loop_deadline = self
            .loop_timeout
            .map(|timeout| TimeoutDeadline::new(TimeoutKind::Loop, timeout));

        loop {
            if let Some(deadline) = elapsed_timeout(loop_deadline) {
                return Ok(self.finish(deadline.end_reason()));
            }

            if self.drain_ready_commands(commands_rx) == CommandAction::Abort {
                return Ok(self.finish(EndReason::Aborted));
            }

            if self.finish_when_idle && !self.has_pending_turn() {
                return Ok(self.finish(EndReason::Idle));
            }

            if !self.has_pending_turn() {
                let command = tokio::select! {
                    command = commands_rx.recv() => command,
                    _ = wait_for_timeout(loop_deadline), if loop_deadline.is_some() => {
                        let deadline = loop_deadline.expect("loop timeout branch requires a deadline");
                        return Ok(self.finish(deadline.end_reason()));
                    }
                };

                let Some(command) = command else {
                    return Ok(self.finish(EndReason::Idle));
                };

                if self.handle_idle_command(command) == CommandAction::Abort {
                    return Ok(self.finish(EndReason::Aborted));
                }
                continue;
            }

            if self.drain_ready_commands(commands_rx) == CommandAction::Abort {
                return Ok(self.finish(EndReason::Aborted));
            }

            let Some(turn) = self.next_turn() else {
                continue;
            };

            match self.run_turn(turn, commands_rx, loop_deadline).await? {
                PromptAction::Continue => {}
                PromptAction::Abort => return Ok(self.finish(EndReason::Aborted)),
                PromptAction::Finish(end_reason) => return Ok(self.finish(end_reason)),
            }
        }
    }

    async fn run_turn(
        &mut self,
        turn: PendingTurn,
        commands_rx: &mut mpsc::UnboundedReceiver<Command>,
        loop_deadline: Option<TimeoutDeadline>,
    ) -> Result<PromptAction, AgentLoopError> {
        let committed_base_history = self.history_snapshot();
        let turn = PreparedTurn::from_pending(turn, committed_base_history)?;
        self.set_turn_running(true);

        self.emit(AgentLoopEvent::TurnStarted {
            prompt: turn.prompt.clone(),
        });

        let mut partial_turn = PartialTurn::default();
        let mut assistant_message_hook = None;
        let mut commands_closed = false;
        let turn_deadline = self
            .turn_timeout
            .map(|timeout| TimeoutDeadline::new(TimeoutKind::Turn, timeout));
        let stream_future = self
            .agent
            .stream_prompt(turn.prompt.clone())
            .with_history(turn.request_history.clone())
            .multi_turn(self.max_turns)
            .into_future();
        tokio::pin!(stream_future);

        let mut stream = loop {
            let deadline = next_timeout(turn_deadline, loop_deadline);
            if let Some(deadline) = elapsed_timeout(deadline) {
                wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                return self.timeout_turn(&turn, &partial_turn, deadline).await;
            }

            tokio::select! {
                command = commands_rx.recv(), if !commands_closed => {
                    let Some(command) = command else {
                        commands_closed = true;
                        continue;
                    };

                    if let Some(action) = self
                        .handle_turn_command(command, &turn, &partial_turn)
                        .await?
                    {
                        return Ok(action);
                    }
                }
                _ = wait_for_timeout(deadline), if deadline.is_some() => {
                    let deadline = deadline.expect("timeout branch requires a deadline");
                    wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                    return self.timeout_turn(&turn, &partial_turn, deadline).await;
                }
                stream = &mut stream_future => break stream,
            }
        };

        loop {
            let deadline = next_timeout(turn_deadline, loop_deadline);
            if let Some(deadline) = elapsed_timeout(deadline) {
                wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                return self.timeout_turn(&turn, &partial_turn, deadline).await;
            }

            tokio::select! {
                command = commands_rx.recv(), if !commands_closed => {
                    let Some(command) = command else {
                        commands_closed = true;
                        continue;
                    };

                    if matches!(command, Command::Interrupt(_) | Command::Abort) {
                        wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                    }
                    if let Some(action) = self
                        .handle_turn_command(command, &turn, &partial_turn)
                        .await?
                    {
                        return Ok(action);
                    }
                }
                _ = wait_for_timeout(deadline), if deadline.is_some() => {
                    let deadline = deadline.expect("timeout branch requires a deadline");
                    wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                    return self.timeout_turn(&turn, &partial_turn, deadline).await;
                }
                hook = poll_assistant_message_hook(&mut assistant_message_hook), if assistant_message_hook.is_some() => {
                    assistant_message_hook = None;
                    hook?;
                }
                item = stream.next() => {
                    let Some(item) = item else {
                        wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                        self.set_turn_running(false);
                        return Ok(PromptAction::Continue);
                    };

                    match item {
                        Ok(item) => {
                            self.emit(AgentLoopEvent::Rig(item.clone()));
                            if let Some(end_reason) = self
                                .handle_stream_item(
                                    item,
                                    &turn,
                                    &mut partial_turn,
                                    &mut assistant_message_hook,
                                    commands_closed,
                                )
                                .await?
                            {
                                return Ok(PromptAction::Finish(end_reason));
                            }
                        }
                        Err(err) => {
                            wait_for_assistant_message_hook(&mut assistant_message_hook).await?;
                            let end_reason = end_reason_from_streaming_error(&err);
                            if let Some(history) = history_from_error(&err) {
                                if let CommitOutcome::Abort(end_reason) = self
                                    .commit_recovered_history(
                                        &turn.committed_base_history,
                                        history,
                                        end_reason.clone(),
                                    )
                                    .await?
                                {
                                    return Ok(PromptAction::Finish(end_reason));
                                }
                            } else {
                                self.set_turn_running(false);
                            }
                            return Ok(PromptAction::Finish(end_reason));
                        }
                    }
                }
            }
        }
    }

    async fn handle_turn_command(
        &mut self,
        command: Command,
        turn: &PreparedTurn,
        partial_turn: &PartialTurn,
    ) -> Result<Option<PromptAction>, AgentLoopError> {
        match command {
            Command::Steer(message) => self.steering.push_back(message),
            Command::FollowUp(message) => self.follow_ups.push_back(message),
            Command::Resume => self.resumes += 1,
            Command::FinishWhenIdle => self.finish_when_idle = true,
            Command::Interrupt(message) => {
                let messages = turn.partial_messages(partial_turn);
                if let CommitOutcome::Abort(end_reason) = self
                    .commit_append(messages, CommitEvent::Interrupted)
                    .await?
                {
                    return Ok(Some(PromptAction::Finish(end_reason)));
                }
                self.immediate.push_front(message);
                return Ok(Some(PromptAction::Continue));
            }
            Command::Abort => {
                let messages = turn.partial_messages(partial_turn);
                self.commit_append(messages, CommitEvent::Aborted).await?;
                return Ok(Some(PromptAction::Abort));
            }
        }

        Ok(None)
    }

    async fn timeout_turn(
        &mut self,
        turn: &PreparedTurn,
        partial_turn: &PartialTurn,
        deadline: TimeoutDeadline,
    ) -> Result<PromptAction, AgentLoopError> {
        let messages = turn.partial_messages(partial_turn);
        let end_reason = deadline.end_reason();
        if let CommitOutcome::Abort(end_reason) = self
            .commit_append(
                messages,
                CommitEvent::TimedOut {
                    end_reason: end_reason.clone(),
                },
            )
            .await?
        {
            return Ok(PromptAction::Finish(end_reason));
        }
        Ok(PromptAction::Finish(end_reason))
    }

    async fn handle_stream_item(
        &mut self,
        item: MultiTurnStreamItem<M::StreamingResponse>,
        turn: &PreparedTurn,
        partial_turn: &mut PartialTurn,
        assistant_message_hook: &mut Option<AssistantMessageHookTask>,
        commands_closed: bool,
    ) -> Result<Option<EndReason>, AgentLoopError> {
        match item {
            MultiTurnStreamItem::StreamAssistantItem(item) => {
                let is_message_boundary = matches!(item, StreamedAssistantContent::ToolCall { .. });
                partial_turn.note_assistant_item(item);
                if is_message_boundary {
                    self.start_assistant_message_hook(
                        turn,
                        partial_turn,
                        assistant_message_hook,
                        None,
                        None,
                    )
                    .await?;
                }
            }
            MultiTurnStreamItem::StreamUserItem(item) => {
                let StreamedUserContent::ToolResult { tool_result, .. } = &item;
                self.persist_incremental_tool_result(tool_result.clone())
                    .await?;
                partial_turn.note_user_item(item);
            }
            MultiTurnStreamItem::FinalResponse(final_response) => {
                let end_reason = self.projected_final_response_end_reason(commands_closed);
                let usage = final_response.usage();
                self.start_assistant_message_hook(
                    turn,
                    partial_turn,
                    assistant_message_hook,
                    end_reason.clone(),
                    Some(usage),
                )
                .await?;
                wait_for_assistant_message_hook(assistant_message_hook).await?;
                self.last_response = Some(final_response.response().to_string());
                if let Some(messages) = final_response.history()
                    && let CommitOutcome::Abort(end_reason) = self
                        .commit_append(
                            turn.append_messages(messages.to_vec()),
                            CommitEvent::Committed {
                                end_reason,
                                usage: Some(usage),
                            },
                        )
                        .await?
                {
                    return Ok(Some(end_reason));
                }
            }
            MultiTurnStreamItem::CompletionCall(_) => {}
            _ => {}
        }

        Ok(None)
    }

    fn projected_final_response_end_reason(&self, commands_closed: bool) -> Option<EndReason> {
        if self.has_pending_turn() || !(self.finish_when_idle || commands_closed) {
            return None;
        }

        Some(EndReason::Idle)
    }

    async fn repair_initial_history(&mut self) -> Result<(), AgentLoopError> {
        let Some(repair_message) = &self.unanswered_tool_call_repair else {
            return Ok(());
        };

        let repaired = repair_unanswered_tool_calls_with_persistence(
            self.history_snapshot(),
            repair_message,
            self.tool_result_persistence.as_deref(),
        )
        .await?;
        *lock_messages(&self.state) = repaired;
        Ok(())
    }

    async fn persist_incremental_tool_result(
        &self,
        tool_result: ToolResult,
    ) -> Result<(), AgentLoopError> {
        let Some(persistence) = &self.tool_result_persistence else {
            return Ok(());
        };

        let key = ToolResultKey::from_tool_result(&tool_result);
        persistence
            .persist_tool_result(key, tool_result)
            .await
            .map_err(AgentLoopError::ToolResultPersistence)
    }

    async fn start_assistant_message_hook(
        &self,
        turn: &PreparedTurn,
        partial_turn: &mut PartialTurn,
        assistant_message_hook: &mut Option<AssistantMessageHookTask>,
        end_reason: Option<EndReason>,
        usage: Option<Usage>,
    ) -> Result<(), AgentLoopError> {
        let Some(message) = partial_turn.assistant_message() else {
            return Ok(());
        };

        wait_for_assistant_message_hook(assistant_message_hook).await?;

        let messages = turn.assistant_messages(partial_turn, message.clone());
        let mut history = self.history_snapshot();
        history.extend(messages.clone());
        partial_turn.mark_assistant_message_finished(message.clone());
        self.emit(AgentLoopEvent::AssistantMessageFinished {
            message: message.clone(),
            messages: messages.clone(),
        });

        let Some(hook) = &self.assistant_message_hook else {
            return Ok(());
        };

        let context = AssistantMessageContext {
            message,
            history,
            new_messages: messages,
            end_reason,
            usage,
        };
        *assistant_message_hook = Some(AssistantMessageHookTask::spawn(hook(
            self.app_state.clone(),
            context,
        )));
        Ok(())
    }

    async fn commit_append(
        &mut self,
        messages: Vec<Message>,
        event: CommitEvent,
    ) -> Result<CommitOutcome, AgentLoopError> {
        let before = self.history_snapshot();
        let mut next_history = before;
        next_history.extend(messages.clone());
        validate_message_history(&next_history).map_err(AgentLoopError::InvalidMessageHistory)?;

        let mut replaced_history = None;
        let mut abort_reason = None;

        if let Some(hook) = &self.turn_hook {
            let turn = TurnHookContext {
                history: next_history.clone(),
                new_messages: messages.clone(),
                end_reason: event.end_reason(),
                usage: event.usage(),
            };

            match hook(self.app_state.clone(), turn)
                .await
                .map_err(AgentLoopError::TurnHook)?
            {
                TurnHookAction::Continue => {}
                TurnHookAction::ReplaceHistory(history) => {
                    validate_message_history(&history)
                        .map_err(AgentLoopError::InvalidMessageHistory)?;
                    next_history = history.clone();
                    replaced_history = Some(history);
                }
                TurnHookAction::Abort { reason } => {
                    abort_reason = Some(reason);
                }
                TurnHookAction::ReplaceHistoryAndAbort { history, reason } => {
                    validate_message_history(&history)
                        .map_err(AgentLoopError::InvalidMessageHistory)?;
                    next_history = history.clone();
                    replaced_history = Some(history);
                    abort_reason = Some(reason);
                }
            }
        }

        *lock_messages(&self.state) = next_history;
        self.set_turn_running(false);
        self.emit(event.into_agent_event(messages));

        if let Some(messages) = replaced_history {
            self.emit(AgentLoopEvent::HistoryReplaced { messages });
        }

        if let Some(reason) = abort_reason {
            return Ok(CommitOutcome::Abort(EndReason::AbortedByHook { reason }));
        }

        Ok(CommitOutcome::Continue)
    }

    pub(crate) async fn commit_recovered_history(
        &mut self,
        base_history: &[Message],
        recovered_history: Vec<Message>,
        end_reason: EndReason,
    ) -> Result<CommitOutcome, AgentLoopError> {
        validate_message_history(&recovered_history)
            .map_err(AgentLoopError::InvalidMessageHistory)?;

        if !recovered_history.starts_with(base_history) {
            return Err(AgentLoopError::InvalidMessageHistory(
                InvalidMessageHistoryError::new(
                    "recovered history does not extend the committed base history",
                ),
            ));
        }

        let append = recovered_history[base_history.len()..].to_vec();
        self.commit_append(
            append,
            CommitEvent::Committed {
                end_reason: Some(end_reason),
                usage: None,
            },
        )
        .await
    }

    fn finish(&mut self, end_reason: EndReason) -> AgentLoopResult {
        self.emit(AgentLoopEvent::LoopEnded {
            end_reason: end_reason.clone(),
        });

        AgentLoopResult {
            end_reason,
            history: self.history_snapshot(),
            last_response: self.last_response.take(),
        }
    }

    fn emit(&self, event: AgentLoopEvent<M::StreamingResponse>) {
        let _ = self.events_tx.send(event);
    }

    fn history_snapshot(&self) -> Vec<Message> {
        lock_messages(&self.state).clone()
    }

    fn set_turn_running(&self, running: bool) {
        self.turn_running.store(running, Ordering::SeqCst);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CommitEvent {
    Committed {
        end_reason: Option<EndReason>,
        usage: Option<Usage>,
    },
    Interrupted,
    Aborted,
    TimedOut {
        end_reason: EndReason,
    },
}

impl CommitEvent {
    fn end_reason(&self) -> Option<EndReason> {
        match self {
            Self::Committed { end_reason, .. } => end_reason.clone(),
            Self::Interrupted => None,
            Self::Aborted => Some(EndReason::Aborted),
            Self::TimedOut { end_reason } => Some(end_reason.clone()),
        }
    }

    fn usage(&self) -> Option<Usage> {
        match self {
            Self::Committed { usage, .. } => *usage,
            Self::Interrupted | Self::Aborted | Self::TimedOut { .. } => None,
        }
    }

    fn into_agent_event<R>(self, messages: Vec<Message>) -> AgentLoopEvent<R> {
        match self {
            Self::Committed { .. } => AgentLoopEvent::TurnCommitted { messages },
            Self::Interrupted => AgentLoopEvent::TurnInterrupted { messages },
            Self::Aborted => AgentLoopEvent::TurnAborted { messages },
            Self::TimedOut { .. } => AgentLoopEvent::TurnTimedOut { messages },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandAction {
    Continue,
    Abort,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CommitOutcome {
    Continue,
    Abort(EndReason),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PromptAction {
    Continue,
    Abort,
    Finish(EndReason),
}

fn history_from_error(err: &StreamingError) -> Option<Vec<Message>> {
    let StreamingError::Prompt(err) = err else {
        return None;
    };

    match err.as_ref() {
        PromptError::MaxTurnsError { chat_history, .. } => Some(chat_history.as_ref().clone()),
        PromptError::PromptCancelled { chat_history, .. } => Some(chat_history.clone()),
        PromptError::UnknownToolCall { chat_history, .. } => Some(chat_history.as_ref().clone()),
        PromptError::CompletionError(_)
        | PromptError::ToolError(_)
        | PromptError::ToolServerError(_) => None,
    }
}

fn end_reason_from_streaming_error(err: &StreamingError) -> EndReason {
    match err {
        StreamingError::Completion(err) => end_reason_from_completion_error(err),
        StreamingError::Prompt(err) => end_reason_from_prompt_error(err),
        StreamingError::Tool(err) => EndReason::ToolError {
            message: err.to_string(),
        },
    }
}

fn end_reason_from_prompt_error(err: &PromptError) -> EndReason {
    match err {
        PromptError::CompletionError(err) => end_reason_from_completion_error(err),
        PromptError::ToolError(err) => EndReason::ToolError {
            message: err.to_string(),
        },
        PromptError::ToolServerError(err) => EndReason::ToolError {
            message: err.to_string(),
        },
        PromptError::MaxTurnsError { max_turns, .. } => EndReason::MaxTurns {
            max_turns: *max_turns,
        },
        PromptError::PromptCancelled { reason, .. } => EndReason::Other {
            message: reason.clone(),
        },
        PromptError::UnknownToolCall { tool_name, .. } => EndReason::ToolError {
            message: format!("unknown tool call: {tool_name}"),
        },
    }
}

fn end_reason_from_completion_error(err: &CompletionError) -> EndReason {
    let error = ApiErrorInfo::from_completion_error(err);

    if is_content_filter_error(&error.message) {
        EndReason::ContentFilter { error }
    } else if is_context_full_error(&error.message) {
        EndReason::ContextFull { error }
    } else if is_length_error(&error.message) {
        EndReason::Length { error }
    } else {
        EndReason::ApiError { error }
    }
}

impl ApiErrorInfo {
    fn from_completion_error(err: &CompletionError) -> Self {
        let (kind, message) = match err {
            CompletionError::HttpError(err) => (ApiErrorKind::Http, err.to_string()),
            CompletionError::JsonError(err) => (ApiErrorKind::Json, err.to_string()),
            CompletionError::UrlError(err) => (ApiErrorKind::Url, err.to_string()),
            CompletionError::RequestError(err) => (ApiErrorKind::Request, err.to_string()),
            CompletionError::ResponseError(message) => (ApiErrorKind::Response, message.clone()),
            CompletionError::ProviderError(message) => (ApiErrorKind::Provider, message.clone()),
        };

        Self { kind, message }
    }
}

fn is_content_filter_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    contains_any(
        &message,
        &[
            "content_filter",
            "content filter",
            "safety filter",
            "policy violation",
            "responsible_ai_policy",
            "unsafe prompt",
            "blocked by policy",
        ],
    )
}

fn is_context_full_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    contains_any(
        &message,
        &[
            "context_length_exceeded",
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "too many input tokens",
        ],
    )
}

fn is_length_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    contains_any(
        &message,
        &[
            "max_output_tokens",
            "max output tokens",
            "output token limit",
            "finish_reason: length",
            "finish reason length",
        ],
    )
}

fn contains_any(message: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| message.contains(needle))
}
