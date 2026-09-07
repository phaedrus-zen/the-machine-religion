//! Bounded App Registry lease for Machine Spirit 3.
//!
//! Replaces the detached one-shot `POST /apps/register` with:
//! GET/list exact-name rows first, then at most one create, singleton
//! reuse + heartbeat, or fail-closed zero mutation on duplicates.

use serde_json::{json, Value};
use std::time::Duration;

pub const APP_NAME: &str = "machine_spirit_3";
pub const DEFAULT_HEARTBEAT_TIMEOUT_SECS: u64 = 120;
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(3);
pub const CLEANUP_BOUND: Duration = Duration::from_secs(8);
pub const CLEANUP_TIMEOUT_MESSAGE: &str = "App Registry lease cleanup timed out";

const MAX_APP_ID_LEN: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppRow {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldLease {
    pub app_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupOutcome {
    Deleted,
    NoDestructiveAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatOutcome {
    Ok,
    NotFound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseError {
    DuplicateExactName { count: usize },
    EmptyAppId,
    NotFound,
    HttpNon2xx { status: u16, body: String },
    Malformed(String),
    Timeout,
    Transport(String),
}

impl std::fmt::Display for LeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateExactName { count } => write!(
                f,
                "duplicate exact-name App Registry rows ({count}); fail-closed with zero mutations"
            ),
            Self::EmptyAppId => write!(f, "register response missing nonempty app_id"),
            Self::NotFound => write!(f, "App Registry row not found"),
            Self::HttpNon2xx { status, body } => {
                write!(f, "App Registry HTTP {status}: {body}")
            }
            Self::Malformed(msg) => write!(f, "malformed App Registry response: {msg}"),
            Self::Timeout => write!(f, "App Registry request timed out"),
            Self::Transport(msg) => write!(f, "App Registry transport error: {msg}"),
        }
    }
}

impl std::error::Error for LeaseError {}

pub trait RegistryApi: Send + Sync {
    async fn list_apps(&self) -> Result<Vec<AppRow>, LeaseError>;
    async fn register(&self, manifest: &Value) -> Result<String, LeaseError>;
    async fn update(&self, app_id: &str, manifest: &Value) -> Result<(), LeaseError>;
    async fn heartbeat(&self, app_id: &str) -> Result<HeartbeatOutcome, LeaseError>;
    async fn delete(&self, app_id: &str) -> Result<(), LeaseError>;
}

pub fn heartbeat_interval_secs(timeout_secs: u64) -> u64 {
    if timeout_secs <= 1 {
        return 1;
    }
    (timeout_secs / 3).clamp(1, timeout_secs - 1)
}

pub fn ms3_lease_manifest(port: u16, version: &str, heartbeat_timeout_secs: u64) -> Value {
    json!({
        "name": APP_NAME,
        "version": version,
        "kind": "consciousness_framework",
        "priority": "normal",
        "health_url": format!("http://localhost:{port}/health"),
        "needs": ["chat"],
        "lifecycle": {
            "heartbeat_required": true,
            "heartbeat_timeout_secs": heartbeat_timeout_secs
        },
        "models": {
            "max_q": { "capabilities": ["reasoning", "tool_calling"] },
            "balanced": null,
            "max_p": null
        }
    })
}

pub fn validate_app_id(id: &str) -> Result<&str, LeaseError> {
    let id = id.trim();
    if id.is_empty() {
        return Err(LeaseError::EmptyAppId);
    }
    if id.len() > MAX_APP_ID_LEN {
        return Err(LeaseError::Malformed("app_id exceeds length limit".into()));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(LeaseError::Malformed(
            "app_id has invalid characters".into(),
        ));
    }
    Ok(id)
}

pub fn parse_app_row(value: &Value) -> Result<AppRow, LeaseError> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| value.get("app_id").and_then(Value::as_str))
        .unwrap_or("")
        .trim();
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return Err(LeaseError::Malformed("app row missing exact name".into()));
    }
    let id = validate_app_id(id)?.to_string();
    Ok(AppRow {
        id,
        name: name.to_string(),
    })
}

pub fn parse_list_body(value: &Value) -> Result<Vec<AppRow>, LeaseError> {
    let apps = value
        .get("apps")
        .and_then(Value::as_array)
        .ok_or_else(|| LeaseError::Malformed("list body missing apps array".into()))?;
    apps.iter().map(parse_app_row).collect()
}

pub fn parse_register_app_id(value: &Value) -> Result<String, LeaseError> {
    if !value.is_object() {
        return Err(LeaseError::Malformed(
            "register body is not an object".into(),
        ));
    }
    let id = value
        .get("app_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    validate_app_id(id).map(|id| id.to_string())
}

pub fn exact_name_rows(rows: &[AppRow], name: &str) -> Vec<AppRow> {
    rows.iter()
        .filter(|row| row.name == name)
        .cloned()
        .collect()
}

async fn list_exact<T: RegistryApi>(client: &T, name: &str) -> Result<Vec<AppRow>, LeaseError> {
    let rows = client.list_apps().await?;
    Ok(exact_name_rows(&rows, name))
}

async fn reuse_then_heartbeat<T: RegistryApi>(
    client: &T,
    app_id: &str,
    manifest: &Value,
) -> Result<HeldLease, LeaseError> {
    let app_id = validate_app_id(app_id)?.to_string();
    client.update(&app_id, manifest).await?;
    match client.heartbeat(&app_id).await? {
        HeartbeatOutcome::Ok => Ok(HeldLease { app_id }),
        HeartbeatOutcome::NotFound => Err(LeaseError::NotFound),
    }
}

async fn recover_after_lost_register<T: RegistryApi>(
    client: &T,
    name: &str,
    manifest: &Value,
    original: LeaseError,
) -> Result<HeldLease, LeaseError> {
    match list_exact(client, name).await {
        Ok(rows) if rows.len() == 1 => reuse_then_heartbeat(client, &rows[0].id, manifest).await,
        Ok(rows) if rows.len() > 1 => Err(LeaseError::DuplicateExactName { count: rows.len() }),
        Ok(_) => Err(original),
        Err(e) => Err(e),
    }
}

async fn post_once_or_recover<T: RegistryApi>(
    client: &T,
    name: &str,
    manifest: &Value,
) -> Result<HeldLease, LeaseError> {
    match client.register(manifest).await {
        Ok(id) => match validate_app_id(&id) {
            Ok(id) => Ok(HeldLease {
                app_id: id.to_string(),
            }),
            Err(e) => recover_after_lost_register(client, name, manifest, e).await,
        },
        Err(e) => recover_after_lost_register(client, name, manifest, e).await,
    }
}

pub async fn acquire_lease<T: RegistryApi>(
    client: &T,
    name: &str,
    manifest: &Value,
) -> Result<HeldLease, LeaseError> {
    let rows = list_exact(client, name).await?;
    match rows.len() {
        0 => post_once_or_recover(client, name, manifest).await,
        1 => reuse_then_heartbeat(client, &rows[0].id, manifest).await,
        count => Err(LeaseError::DuplicateExactName { count }),
    }
}

pub async fn beat_or_reacquire<T: RegistryApi>(
    client: &T,
    name: &str,
    manifest: &Value,
    lease: HeldLease,
) -> Result<HeldLease, LeaseError> {
    match client.heartbeat(&lease.app_id).await {
        Ok(HeartbeatOutcome::Ok) => Ok(lease),
        Ok(HeartbeatOutcome::NotFound) | Err(LeaseError::NotFound) => {
            acquire_lease(client, name, manifest).await
        }
        Err(e) => Err(e),
    }
}

pub async fn shutdown_cleanup<T: RegistryApi>(
    client: &T,
    name: &str,
    held_id: &str,
) -> Result<CleanupOutcome, LeaseError> {
    let held_id = validate_app_id(held_id)?;
    let rows = list_exact(client, name).await?;
    if rows.len() == 1 && rows[0].id == held_id {
        client.delete(held_id).await?;
        return Ok(CleanupOutcome::Deleted);
    }
    Ok(CleanupOutcome::NoDestructiveAction)
}

/// Hold the App Registry lease until `shutdown`, then return the exact
/// `shutdown_cleanup` result so a retained `JoinHandle` can propagate it.
pub async fn run_lease_until_shutdown<T: RegistryApi>(
    client: &T,
    name: &str,
    manifest: &Value,
    heartbeat_timeout_secs: u64,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> Result<(), ShutdownStepError> {
    let interval = Duration::from_secs(heartbeat_interval_secs(heartbeat_timeout_secs));
    let mut held: Option<HeldLease> = None;
    match acquire_lease(client, name, manifest).await {
        Ok(lease) => {
            tracing::info!("║ App Registry lease acquired: app_id={}", lease.app_id);
            held = Some(lease);
        }
        Err(e) => {
            tracing::warn!("App Registry lease acquire fail-closed: {e}");
        }
    }

    while held.is_some() && !*shutdown.borrow() {
        tokio::select! {
            biased;
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            _ = tokio::time::sleep(interval) => {
                if *shutdown.borrow() {
                    break;
                }
                if let Some(lease) = held.clone() {
                    match beat_or_reacquire(client, name, manifest, lease).await {
                        Ok(next) => held = Some(next),
                        Err(e) => {
                            tracing::warn!("App Registry heartbeat fail-closed: {e}");
                        }
                    }
                }
            }
        }
    }

    if let Some(lease) = held {
        return match shutdown_cleanup(client, name, &lease.app_id).await {
            Ok(CleanupOutcome::Deleted) => {
                tracing::info!("║ App Registry lease released: app_id={}", lease.app_id);
                Ok(())
            }
            Ok(CleanupOutcome::NoDestructiveAction) => {
                tracing::info!(
                    "App Registry cleanup: no destructive action for app_id={}",
                    lease.app_id
                );
                Ok(())
            }
            Err(e) => {
                tracing::warn!("App Registry cleanup fail-closed: {e}");
                Err(ShutdownStepError(e.to_string()))
            }
        };
    }
    Ok(())
}

/// Fail-closed error from one shutdown step. The coordinator records it
/// and continues; it never skips a later step with `?`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutdownStepError(pub String);

impl std::fmt::Display for ShutdownStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ShutdownStepError {}

/// Recorded result of one ordered shutdown step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownStepResult {
    Completed,
    Failed(ShutdownStepError),
}

/// Outcome of the exclusive Ctrl-C shutdown coordinator used by `main`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutdownCoordinatorReport {
    pub cleanup: ShutdownStepResult,
    pub save: ShutdownStepResult,
    pub stop: ShutdownStepResult,
    pub server_return: ShutdownStepResult,
}

fn record_shutdown_step(result: Result<(), ShutdownStepError>) -> ShutdownStepResult {
    match result {
        Ok(()) => ShutdownStepResult::Completed,
        Err(error) => ShutdownStepResult::Failed(error),
    }
}

/// Sole Ctrl-C owner used by production `main`.
///
/// Exact order after `signal` (fail-closed: a failed step is recorded and
/// every later step still runs):
/// 1. `cleanup` — release the App Registry lease
/// 2. `save` — persist mind / manager state
/// 3. `stop` — `ServerHandle.stop(true)` graceful drain
/// 4. await `server` — `HttpServer::run()` completes so destructors run
///    and `main` returns normally
///
/// `server` is polled concurrently with `signal` so the HTTP server stays
/// live until shutdown. This function never calls `std::process::exit`.
/// Pair with `HttpServer::disable_signals()` so Actix default signal
/// handling cannot return and drop the runtime first.
///
/// `cleanup` must retain any spawned Registry task `JoinHandle`. On
/// timeout it must abort that handle and join termination before this
/// function proceeds to `save`. Use [`await_cleanup_task_bound`].
pub async fn run_exclusive_ctrl_c_shutdown<Sig, Clean, Save, Stop, Server>(
    signal: Sig,
    cleanup: Clean,
    save: Save,
    stop: Stop,
    server: Server,
) -> ShutdownCoordinatorReport
where
    Sig: std::future::Future<Output = ()>,
    Clean: std::future::Future<Output = Result<(), ShutdownStepError>>,
    Save: std::future::Future<Output = Result<(), ShutdownStepError>>,
    Stop: std::future::Future<Output = Result<(), ShutdownStepError>>,
    Server: std::future::Future<Output = Result<(), ShutdownStepError>>,
{
    tokio::pin!(signal);
    tokio::pin!(server);

    let server_ended_before_signal = tokio::select! {
        biased;
        _ = &mut signal => None,
        result = &mut server => Some(result),
    };

    let cleanup = record_shutdown_step(cleanup.await);
    let save = record_shutdown_step(save.await);
    let stop = record_shutdown_step(stop.await);
    let server_return = match server_ended_before_signal {
        Some(result) => record_shutdown_step(result),
        None => record_shutdown_step(server.await),
    };

    ShutdownCoordinatorReport {
        cleanup,
        save,
        stop,
        server_return,
    }
}

fn map_cleanup_join(
    joined: Result<Result<(), ShutdownStepError>, tokio::task::JoinError>,
) -> Result<(), ShutdownStepError> {
    match joined {
        Ok(result) => result,
        Err(join_err) if join_err.is_cancelled() => Err(ShutdownStepError(
            "App Registry lease cleanup cancelled".into(),
        )),
        Err(join_err) => Err(ShutdownStepError(format!(
            "App Registry lease task panicked: {join_err}"
        ))),
    }
}

/// Await a retained cleanup `JoinHandle` for `bound`.
///
/// Passing the handle by value into `tokio::time::timeout` would drop it on
/// expiry and detach the task. This waits on `&mut task` so the handle stays
/// owned. On timeout the task is aborted and joined before return, so a late
/// Registry call cannot run concurrently with save / stop / server return.
pub async fn await_cleanup_task_bound(
    mut task: tokio::task::JoinHandle<Result<(), ShutdownStepError>>,
    bound: Duration,
) -> Result<(), ShutdownStepError> {
    match tokio::time::timeout(bound, &mut task).await {
        Ok(joined) => map_cleanup_join(joined),
        Err(_elapsed) => {
            task.abort();
            match task.await {
                Ok(result) => result,
                Err(join_err) if join_err.is_cancelled() => {
                    Err(ShutdownStepError(CLEANUP_TIMEOUT_MESSAGE.into()))
                }
                Err(join_err) => Err(ShutdownStepError(format!(
                    "App Registry lease task panicked: {join_err}"
                ))),
            }
        }
    }
}

/// Production waiter used by `main` when lease-client construction may fail
/// closed with no spawned task.
pub async fn await_optional_cleanup_task_bound(
    task: Option<tokio::task::JoinHandle<Result<(), ShutdownStepError>>>,
    bound: Duration,
) -> Result<(), ShutdownStepError> {
    match task {
        Some(task) => await_cleanup_task_bound(task, bound).await,
        None => Ok(()),
    }
}

pub struct HttpRegistry {
    client: reqwest::Client,
    base: String,
}

impl HttpRegistry {
    pub fn new(base: &str, timeout: Duration) -> Result<Self, LeaseError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| LeaseError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            base: base.trim_end_matches('/').to_string(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn map_reqwest(err: reqwest::Error) -> LeaseError {
        if err.is_timeout() {
            LeaseError::Timeout
        } else {
            LeaseError::Transport(err.to_string())
        }
    }

    async fn send_json(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<(reqwest::StatusCode, Value), LeaseError> {
        let response = builder.send().await.map_err(Self::map_reqwest)?;
        let status = response.status();
        if status.as_u16() == 404 {
            return Err(LeaseError::NotFound);
        }
        let body = response
            .text()
            .await
            .map_err(|e| LeaseError::Malformed(e.to_string()))?;
        if !status.is_success() {
            return Err(LeaseError::HttpNon2xx {
                status: status.as_u16(),
                body,
            });
        }
        if body.trim().is_empty() {
            return Ok((status, json!({})));
        }
        let value =
            serde_json::from_str(&body).map_err(|e| LeaseError::Malformed(e.to_string()))?;
        Ok((status, value))
    }
}

impl RegistryApi for HttpRegistry {
    async fn list_apps(&self) -> Result<Vec<AppRow>, LeaseError> {
        match self.send_json(self.client.get(self.url("/apps"))).await {
            Ok((_, body)) => parse_list_body(&body),
            Err(LeaseError::NotFound) => Err(LeaseError::HttpNon2xx {
                status: 404,
                body: "list endpoint returned 404".into(),
            }),
            Err(e) => Err(e),
        }
    }

    async fn register(&self, manifest: &Value) -> Result<String, LeaseError> {
        match self
            .send_json(self.client.post(self.url("/apps/register")).json(manifest))
            .await
        {
            Ok((_, body)) => parse_register_app_id(&body),
            Err(LeaseError::NotFound) => Err(LeaseError::HttpNon2xx {
                status: 404,
                body: "register endpoint returned 404".into(),
            }),
            Err(e) => Err(e),
        }
    }

    async fn update(&self, app_id: &str, manifest: &Value) -> Result<(), LeaseError> {
        let app_id = validate_app_id(app_id)?;
        self.send_json(
            self.client
                .put(self.url(&format!("/apps/{app_id}")))
                .json(manifest),
        )
        .await
        .map(|_| ())
    }

    async fn heartbeat(&self, app_id: &str) -> Result<HeartbeatOutcome, LeaseError> {
        let app_id = validate_app_id(app_id)?;
        match self
            .send_json(
                self.client
                    .post(self.url(&format!("/apps/{app_id}/heartbeat")))
                    .json(&json!({ "status": "healthy" })),
            )
            .await
        {
            Ok(_) => Ok(HeartbeatOutcome::Ok),
            Err(LeaseError::NotFound) => Ok(HeartbeatOutcome::NotFound),
            Err(e) => Err(e),
        }
    }

    async fn delete(&self, app_id: &str) -> Result<(), LeaseError> {
        let app_id = validate_app_id(app_id)?;
        self.send_json(self.client.delete(self.url(&format!("/apps/{app_id}"))))
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ExclusiveShutdownPhase {
        Cleanup,
        Save,
        Stop,
        ServerReturn,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RegistryOp {
        List,
        Register,
        Update,
        Heartbeat,
        Delete,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RecordedCall {
        op: RegistryOp,
        app_id: Option<String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RegisterBehavior {
        Succeed,
        AcceptButLoseResponse,
        TimeoutNoCreate,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ListBehavior {
        Live,
        Timeout,
    }

    struct FakeState {
        rows: Vec<AppRow>,
        calls: Vec<RecordedCall>,
        next_id: u32,
        register_behavior: RegisterBehavior,
        list_behavior: ListBehavior,
        heartbeat_missing: Vec<String>,
        list_hang: Option<tokio::sync::oneshot::Receiver<()>>,
    }

    #[derive(Clone)]
    struct FakeRegistry {
        state: Arc<Mutex<FakeState>>,
    }

    impl FakeRegistry {
        fn new() -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeState {
                    rows: Vec::new(),
                    calls: Vec::new(),
                    next_id: 1,
                    register_behavior: RegisterBehavior::Succeed,
                    list_behavior: ListBehavior::Live,
                    heartbeat_missing: Vec::new(),
                    list_hang: None,
                })),
            }
        }

        fn seed_exact(&self, id: &str) {
            self.state.lock().expect("fake lock").rows.push(AppRow {
                id: id.to_string(),
                name: APP_NAME.to_string(),
            });
        }

        fn seed_other(&self, id: &str, name: &str) {
            self.state.lock().expect("fake lock").rows.push(AppRow {
                id: id.to_string(),
                name: name.to_string(),
            });
        }

        fn set_register_behavior(&self, behavior: RegisterBehavior) {
            self.state.lock().expect("fake lock").register_behavior = behavior;
        }

        fn set_list_behavior(&self, behavior: ListBehavior) {
            self.state.lock().expect("fake lock").list_behavior = behavior;
        }

        fn set_list_hang(&self, hang: tokio::sync::oneshot::Receiver<()>) {
            self.state.lock().expect("fake lock").list_hang = Some(hang);
        }

        fn mark_heartbeat_missing(&self, id: &str) {
            self.state
                .lock()
                .expect("fake lock")
                .heartbeat_missing
                .push(id.to_string());
        }

        fn calls(&self) -> Vec<RecordedCall> {
            self.state.lock().expect("fake lock").calls.clone()
        }

        fn ops(&self) -> Vec<RegistryOp> {
            self.calls().into_iter().map(|call| call.op).collect()
        }

        fn mutation_ops(&self) -> Vec<RegistryOp> {
            self.ops()
                .into_iter()
                .filter(|op| *op != RegistryOp::List)
                .collect()
        }

        fn rows(&self) -> Vec<AppRow> {
            self.state.lock().expect("fake lock").rows.clone()
        }

        fn record(&self, op: RegistryOp, app_id: Option<&str>) {
            self.state
                .lock()
                .expect("fake lock")
                .calls
                .push(RecordedCall {
                    op,
                    app_id: app_id.map(ToOwned::to_owned),
                });
        }
    }

    impl RegistryApi for FakeRegistry {
        async fn list_apps(&self) -> Result<Vec<AppRow>, LeaseError> {
            let hang = self.state.lock().expect("fake lock").list_hang.take();
            if let Some(hang) = hang {
                let _ = hang.await;
            }
            self.record(RegistryOp::List, None);
            let state = self.state.lock().expect("fake lock");
            match state.list_behavior {
                ListBehavior::Live => Ok(state.rows.clone()),
                ListBehavior::Timeout => Err(LeaseError::Timeout),
            }
        }

        async fn register(&self, manifest: &Value) -> Result<String, LeaseError> {
            self.record(RegistryOp::Register, None);
            let mut state = self.state.lock().expect("fake lock");
            let name = manifest
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(APP_NAME)
                .to_string();
            match state.register_behavior {
                RegisterBehavior::Succeed => {
                    let id = format!("lease-{}", state.next_id);
                    state.next_id += 1;
                    state.rows.push(AppRow {
                        id: id.clone(),
                        name,
                    });
                    Ok(id)
                }
                RegisterBehavior::AcceptButLoseResponse => {
                    let id = format!("lease-{}", state.next_id);
                    state.next_id += 1;
                    state.rows.push(AppRow { id, name });
                    Err(LeaseError::Timeout)
                }
                RegisterBehavior::TimeoutNoCreate => Err(LeaseError::Timeout),
            }
        }

        async fn update(&self, app_id: &str, _manifest: &Value) -> Result<(), LeaseError> {
            self.record(RegistryOp::Update, Some(app_id));
            let state = self.state.lock().expect("fake lock");
            if state.rows.iter().any(|row| row.id == app_id) {
                Ok(())
            } else {
                Err(LeaseError::NotFound)
            }
        }

        async fn heartbeat(&self, app_id: &str) -> Result<HeartbeatOutcome, LeaseError> {
            self.record(RegistryOp::Heartbeat, Some(app_id));
            let mut state = self.state.lock().expect("fake lock");
            if let Some(idx) = state.heartbeat_missing.iter().position(|id| id == app_id) {
                state.heartbeat_missing.remove(idx);
                return Ok(HeartbeatOutcome::NotFound);
            }
            if state.rows.iter().any(|row| row.id == app_id) {
                Ok(HeartbeatOutcome::Ok)
            } else {
                Ok(HeartbeatOutcome::NotFound)
            }
        }

        async fn delete(&self, app_id: &str) -> Result<(), LeaseError> {
            self.record(RegistryOp::Delete, Some(app_id));
            let mut state = self.state.lock().expect("fake lock");
            let before = state.rows.len();
            state.rows.retain(|row| row.id != app_id);
            if state.rows.len() == before {
                Err(LeaseError::NotFound)
            } else {
                Ok(())
            }
        }
    }

    fn sample_manifest() -> Value {
        ms3_lease_manifest(9090, "0.1.0", 120)
    }

    async fn wait_for_exact_lease(fake: &FakeRegistry) -> String {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let rows = exact_name_rows(&fake.rows(), APP_NAME);
                if rows.len() == 1 {
                    return rows[0].id.clone();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("lease acquire should finish quickly")
    }

    /// Production composition used by `main`:
    /// `run_lease_until_shutdown` → retained `JoinHandle` → bounded await → coordinator.
    async fn run_production_lease_through_coordinator(
        fake: FakeRegistry,
        heartbeat_timeout_secs: u64,
        cleanup_bound: Duration,
        after_acquire: impl FnOnce(&FakeRegistry),
        on_save: impl FnOnce(&FakeRegistry) + Send + 'static,
    ) -> ShutdownCoordinatorReport {
        let phases = Arc::new(Mutex::new(Vec::new()));
        let (lease_tx, lease_rx) = tokio::sync::watch::channel(false);
        let (sig_tx, mut sig_rx) = tokio::sync::watch::channel(false);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();

        let task_fake = fake.clone();
        let lease_task = tokio::spawn(async move {
            run_lease_until_shutdown(
                &task_fake,
                APP_NAME,
                &sample_manifest(),
                heartbeat_timeout_secs,
                lease_rx,
            )
            .await
        });

        let _held = wait_for_exact_lease(&fake).await;
        after_acquire(&fake);

        let phases_cleanup = phases.clone();
        let phases_save = phases.clone();
        let phases_stop = phases.clone();
        let phases_server = phases.clone();
        let fake_save = fake.clone();

        let coordinator = tokio::spawn(async move {
            run_exclusive_ctrl_c_shutdown(
                async {
                    let _ = sig_rx.changed().await;
                },
                async {
                    let _ = lease_tx.send(true);
                    let result =
                        await_optional_cleanup_task_bound(Some(lease_task), cleanup_bound).await;
                    phases_cleanup
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::Cleanup);
                    result
                },
                async {
                    on_save(&fake_save);
                    phases_save
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::Save);
                    Ok(())
                },
                async {
                    phases_stop
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::Stop);
                    let _ = release_tx.send(());
                    Ok(())
                },
                async {
                    let _ = release_rx.await;
                    phases_server
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::ServerReturn);
                    Ok(())
                },
            )
            .await
        });

        sig_tx.send(true).expect("signal ctrl-c");
        let report = coordinator.await.expect("coordinator");
        assert_eq!(
            phases.lock().expect("phase lock").clone(),
            vec![
                ExclusiveShutdownPhase::Cleanup,
                ExclusiveShutdownPhase::Save,
                ExclusiveShutdownPhase::Stop,
                ExclusiveShutdownPhase::ServerReturn,
            ]
        );
        report
    }

    #[test]
    fn lease_manifest_sets_heartbeat_required_below_timeout() {
        let manifest = sample_manifest();
        assert_eq!(manifest["name"], APP_NAME);
        assert_eq!(manifest["lifecycle"]["heartbeat_required"], true);
        assert_eq!(manifest["lifecycle"]["heartbeat_timeout_secs"], 120);
        let interval = heartbeat_interval_secs(120);
        assert!(interval > 0);
        assert!(interval < 120);
    }

    #[test]
    fn parse_list_body_rejects_malformed() {
        assert!(matches!(
            parse_list_body(&json!([])),
            Err(LeaseError::Malformed(_))
        ));
        assert!(matches!(
            parse_list_body(&json!({"apps": [{"name": APP_NAME}]})),
            Err(LeaseError::EmptyAppId)
        ));
        let rows = parse_list_body(&json!({
            "apps": [
                {"id": "keep-1", "name": APP_NAME},
                {"app_id": "keep-2", "name": "other"}
            ]
        }))
        .expect("valid list");
        assert_eq!(
            exact_name_rows(&rows, APP_NAME),
            vec![AppRow {
                id: "keep-1".into(),
                name: APP_NAME.into()
            }]
        );
    }

    #[tokio::test]
    async fn lost_register_response_is_reused_without_second_post() {
        let fake = FakeRegistry::new();
        fake.seed_other("decoy-1", "someone_else");
        fake.set_register_behavior(RegisterBehavior::AcceptButLoseResponse);

        let lease = acquire_lease(&fake, APP_NAME, &sample_manifest())
            .await
            .expect("lost POST must recover the accepted row");

        assert_eq!(lease.app_id, "lease-1");
        assert_eq!(
            fake.ops(),
            vec![
                RegistryOp::List,
                RegistryOp::Register,
                RegistryOp::List,
                RegistryOp::Update,
                RegistryOp::Heartbeat,
            ]
        );
        assert_eq!(
            fake.calls()
                .into_iter()
                .filter(|call| call.op == RegistryOp::Register)
                .count(),
            1
        );
        assert_eq!(
            exact_name_rows(&fake.rows(), APP_NAME)
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["lease-1"]
        );
    }

    #[tokio::test]
    async fn singleton_exact_name_is_reused_then_heartbeat() {
        let fake = FakeRegistry::new();
        fake.seed_exact("existing-1");
        fake.seed_other("decoy-1", "someone_else");

        let lease = acquire_lease(&fake, APP_NAME, &sample_manifest())
            .await
            .expect("singleton must be reused");

        assert_eq!(lease.app_id, "existing-1");
        assert_eq!(
            fake.ops(),
            vec![RegistryOp::List, RegistryOp::Update, RegistryOp::Heartbeat,]
        );
        assert!(!fake.ops().contains(&RegistryOp::Register));
        assert_eq!(exact_name_rows(&fake.rows(), APP_NAME).len(), 1);
    }

    #[tokio::test]
    async fn duplicate_exact_name_is_fail_closed_with_zero_mutations() {
        let fake = FakeRegistry::new();
        fake.seed_exact("dup-a");
        fake.seed_exact("dup-b");
        fake.seed_other("decoy-1", "someone_else");

        let err = acquire_lease(&fake, APP_NAME, &sample_manifest())
            .await
            .expect_err("duplicates must fail closed");

        assert_eq!(err, LeaseError::DuplicateExactName { count: 2 });
        assert_eq!(fake.ops(), vec![RegistryOp::List]);
        assert!(fake.mutation_ops().is_empty());
        assert_eq!(exact_name_rows(&fake.rows(), APP_NAME).len(), 2);
    }

    #[tokio::test]
    async fn singleton_shutdown_deletes_exact_id_only() {
        let fake = FakeRegistry::new();
        fake.seed_exact("held-1");
        fake.seed_other("decoy-1", "someone_else");

        let outcome = shutdown_cleanup(&fake, APP_NAME, "held-1")
            .await
            .expect("sole exact-name row may be deleted");

        assert_eq!(outcome, CleanupOutcome::Deleted);
        assert_eq!(fake.ops(), vec![RegistryOp::List, RegistryOp::Delete]);
        assert_eq!(
            fake.calls()
                .into_iter()
                .find(|call| call.op == RegistryOp::Delete)
                .and_then(|call| call.app_id),
            Some("held-1".into())
        );
        assert!(exact_name_rows(&fake.rows(), APP_NAME).is_empty());
        assert_eq!(
            fake.rows()
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["decoy-1"]
        );
    }

    #[tokio::test]
    async fn duplicate_shutdown_does_not_delete() {
        let fake = FakeRegistry::new();
        fake.seed_exact("held-1");
        fake.seed_exact("other-1");

        let outcome = shutdown_cleanup(&fake, APP_NAME, "held-1")
            .await
            .expect("duplicate cleanup is non-destructive");

        assert_eq!(outcome, CleanupOutcome::NoDestructiveAction);
        assert_eq!(fake.ops(), vec![RegistryOp::List]);
        assert!(!fake.ops().contains(&RegistryOp::Delete));
        assert_eq!(exact_name_rows(&fake.rows(), APP_NAME).len(), 2);
    }

    #[tokio::test]
    async fn heartbeat_404_reacquires_through_get_not_blind_post() {
        let fake = FakeRegistry::new();
        fake.seed_other("decoy-1", "someone_else");
        let lease = HeldLease {
            app_id: "ghost-1".into(),
        };
        fake.mark_heartbeat_missing("ghost-1");

        let next = beat_or_reacquire(&fake, APP_NAME, &sample_manifest(), lease)
            .await
            .expect("404 must GET then POST once");

        assert_eq!(next.app_id, "lease-1");
        assert_eq!(
            fake.ops(),
            vec![
                RegistryOp::Heartbeat,
                RegistryOp::List,
                RegistryOp::Register,
            ]
        );
        assert_eq!(
            fake.calls()
                .into_iter()
                .filter(|call| call.op == RegistryOp::Register)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn list_timeout_is_fail_closed_with_zero_mutations() {
        let fake = FakeRegistry::new();
        fake.set_list_behavior(ListBehavior::Timeout);

        let err = acquire_lease(&fake, APP_NAME, &sample_manifest())
            .await
            .expect_err("list timeout must fail closed");

        assert_eq!(err, LeaseError::Timeout);
        assert!(fake.mutation_ops().is_empty());
    }

    #[tokio::test]
    async fn ctrl_c_stops_heartbeat_then_deletes_singleton() {
        let fake = FakeRegistry::new();
        fake.seed_other("decoy-1", "someone_else");
        let (tx, rx) = tokio::sync::watch::channel(false);
        let task_fake = fake.clone();
        let handle = tokio::spawn(async move {
            run_lease_until_shutdown(&task_fake, APP_NAME, &sample_manifest(), 9, rx)
                .await
                .expect("successful Ctrl-C cleanup");
        });

        let started = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if exact_name_rows(&fake.rows(), APP_NAME).len() == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(started.is_ok(), "lease acquire should finish quickly");
        let held = exact_name_rows(&fake.rows(), APP_NAME)[0].id.clone();
        tx.send(true).expect("signal shutdown");
        handle.await.expect("lease task");

        assert!(exact_name_rows(&fake.rows(), APP_NAME).is_empty());
        assert!(fake.calls().iter().any(|call| {
            call.op == RegistryOp::Delete && call.app_id.as_deref() == Some(held.as_str())
        }));
        assert!(
            !fake.ops().contains(&RegistryOp::Delete)
                || fake.rows().iter().all(|row| row.name != APP_NAME)
        );
    }

    #[tokio::test]
    async fn exclusive_ctrl_c_cleanup_and_save_finish_before_runtime_return() {
        let phases = Arc::new(Mutex::new(Vec::new()));
        let fake = FakeRegistry::new();
        fake.seed_exact("held-1");
        fake.seed_other("decoy-1", "someone_else");

        let (sig_tx, mut sig_rx) = tokio::sync::watch::channel(false);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let raced_return = Arc::new(AtomicBool::new(false));

        let mut raced_rx = sig_rx.clone();
        let raced_flag = raced_return.clone();
        let raced_owner = tokio::spawn(async move {
            let _ = raced_rx.changed().await;
            raced_flag.store(true, Ordering::SeqCst);
        });

        let phases_task = phases.clone();
        let fake_task = fake.clone();
        let exclusive = tokio::spawn(async move {
            run_exclusive_ctrl_c_shutdown(
                async {
                    let _ = sig_rx.changed().await;
                },
                async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    let outcome = shutdown_cleanup(&fake_task, APP_NAME, "held-1")
                        .await
                        .expect("cleanup");
                    assert_eq!(outcome, CleanupOutcome::Deleted);
                    phases_task
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::Cleanup);
                    Ok(())
                },
                async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    phases_task
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::Save);
                    Ok(())
                },
                async {
                    phases_task
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::Stop);
                    let _ = release_tx.send(());
                    Ok(())
                },
                async {
                    let _ = release_rx.await;
                    phases_task
                        .lock()
                        .expect("phase lock")
                        .push(ExclusiveShutdownPhase::ServerReturn);
                    Ok(())
                },
            )
            .await
        });

        sig_tx.send(true).expect("signal ctrl-c");
        raced_owner.await.expect("raced owner");
        assert!(
            raced_return.load(Ordering::SeqCst),
            "second signal owner (Actix default) returns immediately"
        );
        assert!(
            phases.lock().expect("phase lock").is_empty(),
            "cleanup/save must not have finished if a second owner returned first"
        );

        let report = exclusive.await.expect("exclusive owner");
        assert_eq!(
            phases.lock().expect("phase lock").clone(),
            vec![
                ExclusiveShutdownPhase::Cleanup,
                ExclusiveShutdownPhase::Save,
                ExclusiveShutdownPhase::Stop,
                ExclusiveShutdownPhase::ServerReturn,
            ]
        );
        assert_eq!(report.cleanup, ShutdownStepResult::Completed);
        assert_eq!(report.save, ShutdownStepResult::Completed);
        assert_eq!(report.stop, ShutdownStepResult::Completed);
        assert_eq!(report.server_return, ShutdownStepResult::Completed);
        assert!(exact_name_rows(&fake.rows(), APP_NAME).is_empty());
        assert!(fake.rows().iter().any(|row| row.id == "decoy-1"));
    }

    #[tokio::test]
    async fn graceful_shutdown_coordinator_order_is_cleanup_save_stop_then_server_return() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let server_polled = Arc::new(AtomicBool::new(false));
        let (sig_tx, mut sig_rx) = tokio::sync::watch::channel(false);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();

        let log_cleanup = log.clone();
        let log_save = log.clone();
        let log_stop = log.clone();
        let log_server = log.clone();
        let polled = server_polled.clone();
        let coordinator = tokio::spawn(async move {
            run_exclusive_ctrl_c_shutdown(
                async {
                    let _ = sig_rx.changed().await;
                },
                async {
                    log_cleanup
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::Cleanup);
                    Ok(())
                },
                async {
                    log_save
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::Save);
                    Ok(())
                },
                async {
                    log_stop
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::Stop);
                    let _ = release_tx.send(());
                    Ok(())
                },
                async {
                    polled.store(true, Ordering::SeqCst);
                    let _ = release_rx.await;
                    log_server
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::ServerReturn);
                    Ok(())
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(1), async {
            while !server_polled.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("server future must be polled before signal so HttpServer stays live");
        assert!(
            log.lock().expect("log lock").is_empty(),
            "cleanup/save/stop must not start before the shutdown signal"
        );

        sig_tx.send(true).expect("signal ctrl-c");
        let report = coordinator.await.expect("coordinator");

        assert_eq!(
            log.lock().expect("log lock").clone(),
            vec![
                ExclusiveShutdownPhase::Cleanup,
                ExclusiveShutdownPhase::Save,
                ExclusiveShutdownPhase::Stop,
                ExclusiveShutdownPhase::ServerReturn,
            ]
        );
        assert_eq!(report.cleanup, ShutdownStepResult::Completed);
        assert_eq!(report.save, ShutdownStepResult::Completed);
        assert_eq!(report.stop, ShutdownStepResult::Completed);
        assert_eq!(report.server_return, ShutdownStepResult::Completed);
    }

    #[tokio::test]
    async fn graceful_shutdown_coordinator_fail_closed_still_stops_and_awaits_server() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (sig_tx, mut sig_rx) = tokio::sync::watch::channel(false);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();

        let log_cleanup = log.clone();
        let log_save = log.clone();
        let log_stop = log.clone();
        let log_server = log.clone();
        let coordinator = tokio::spawn(async move {
            run_exclusive_ctrl_c_shutdown(
                async {
                    let _ = sig_rx.changed().await;
                },
                async {
                    log_cleanup
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::Cleanup);
                    Err(ShutdownStepError("cleanup failed".into()))
                },
                async {
                    log_save
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::Save);
                    Err(ShutdownStepError("save failed".into()))
                },
                async {
                    log_stop
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::Stop);
                    let _ = release_tx.send(());
                    Err(ShutdownStepError("stop failed".into()))
                },
                async {
                    let _ = release_rx.await;
                    log_server
                        .lock()
                        .expect("log lock")
                        .push(ExclusiveShutdownPhase::ServerReturn);
                    Ok(())
                },
            )
            .await
        });

        sig_tx.send(true).expect("signal ctrl-c");
        let report = coordinator.await.expect("coordinator");

        assert_eq!(
            log.lock().expect("log lock").clone(),
            vec![
                ExclusiveShutdownPhase::Cleanup,
                ExclusiveShutdownPhase::Save,
                ExclusiveShutdownPhase::Stop,
                ExclusiveShutdownPhase::ServerReturn,
            ],
            "cleanup/save/stop errors must not skip later ordered steps"
        );
        assert_eq!(
            report.cleanup,
            ShutdownStepResult::Failed(ShutdownStepError("cleanup failed".into()))
        );
        assert_eq!(
            report.save,
            ShutdownStepResult::Failed(ShutdownStepError("save failed".into()))
        );
        assert_eq!(
            report.stop,
            ShutdownStepResult::Failed(ShutdownStepError("stop failed".into()))
        );
        assert_eq!(
            report.server_return,
            ShutdownStepResult::Completed,
            "server future must still complete so main can return"
        );
    }

    #[tokio::test]
    async fn production_coordinator_timeout_aborts_cleanup_before_save_with_no_late_registry_effect(
    ) {
        let fake = FakeRegistry::new();
        fake.seed_other("decoy-1", "someone_else");
        let (late_gate_tx, late_gate_rx) = tokio::sync::oneshot::channel::<()>();
        let ops_at_save = Arc::new(Mutex::new(None::<Vec<RegistryOp>>));
        let ops_at_save_slot = ops_at_save.clone();

        let report = run_production_lease_through_coordinator(
            fake.clone(),
            120,
            Duration::from_millis(40),
            |fake| {
                fake.set_list_hang(late_gate_rx);
            },
            move |fake_at_save| {
                *ops_at_save_slot.lock().expect("ops lock") = Some(fake_at_save.ops());
                assert!(
                    !fake_at_save.ops().contains(&RegistryOp::Delete),
                    "late Registry delete must already be impossible when save starts"
                );
            },
        )
        .await;

        assert_eq!(
            report.cleanup,
            ShutdownStepResult::Failed(ShutdownStepError(CLEANUP_TIMEOUT_MESSAGE.into()))
        );
        assert_eq!(report.save, ShutdownStepResult::Completed);
        assert_eq!(report.stop, ShutdownStepResult::Completed);
        assert_eq!(report.server_return, ShutdownStepResult::Completed);

        assert!(
            late_gate_tx.send(()).is_err(),
            "aborted cleanup must be joined before save, dropping the late-effect gate"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !fake.ops().contains(&RegistryOp::Delete),
            "blocked cleanup must not emit a post-timeout Registry delete"
        );
        assert_eq!(
            fake.ops(),
            ops_at_save
                .lock()
                .expect("ops lock")
                .clone()
                .expect("save must snapshot Registry ops"),
            "Registry call log must be frozen before save"
        );
        assert_eq!(exact_name_rows(&fake.rows(), APP_NAME).len(), 1);
        assert!(fake.rows().iter().any(|row| row.id == "decoy-1"));
    }

    #[tokio::test]
    async fn production_coordinator_cleanup_success_joins_before_save() {
        let fake = FakeRegistry::new();
        fake.seed_other("decoy-1", "someone_else");

        let report = run_production_lease_through_coordinator(
            fake.clone(),
            120,
            CLEANUP_BOUND,
            |_fake| {},
            |fake_save| {
                assert!(
                    exact_name_rows(&fake_save.rows(), APP_NAME).is_empty(),
                    "successful cleanup must finish its Registry delete before save"
                );
            },
        )
        .await;

        assert_eq!(report.cleanup, ShutdownStepResult::Completed);
        assert_eq!(report.save, ShutdownStepResult::Completed);
        assert_eq!(report.stop, ShutdownStepResult::Completed);
        assert_eq!(report.server_return, ShutdownStepResult::Completed);
        assert!(fake.ops().contains(&RegistryOp::Delete));
        assert!(exact_name_rows(&fake.rows(), APP_NAME).is_empty());
        assert!(fake.rows().iter().any(|row| row.id == "decoy-1"));
    }

    #[tokio::test]
    async fn production_coordinator_cleanup_error_joins_before_save() {
        let fake = FakeRegistry::new();
        fake.seed_other("decoy-1", "someone_else");

        let report = run_production_lease_through_coordinator(
            fake.clone(),
            120,
            CLEANUP_BOUND,
            |fake| {
                fake.set_list_behavior(ListBehavior::Timeout);
            },
            |fake_save| {
                assert!(
                    !fake_save.ops().contains(&RegistryOp::Delete),
                    "failed cleanup must be joined before save, with no later mutation"
                );
                assert_eq!(exact_name_rows(&fake_save.rows(), APP_NAME).len(), 1);
            },
        )
        .await;

        assert_eq!(
            report.cleanup,
            ShutdownStepResult::Failed(ShutdownStepError(LeaseError::Timeout.to_string())),
            "Registry cleanup failure must not be reported as Completed"
        );
        assert_eq!(report.save, ShutdownStepResult::Completed);
        assert_eq!(report.stop, ShutdownStepResult::Completed);
        assert_eq!(report.server_return, ShutdownStepResult::Completed);
        assert!(!fake.ops().contains(&RegistryOp::Delete));
        assert_eq!(exact_name_rows(&fake.rows(), APP_NAME).len(), 1);
        assert!(fake.rows().iter().any(|row| row.id == "decoy-1"));
    }

    #[tokio::test]
    async fn lost_register_with_no_row_stays_fail_closed() {
        let fake = FakeRegistry::new();
        fake.set_register_behavior(RegisterBehavior::TimeoutNoCreate);

        let err = acquire_lease(&fake, APP_NAME, &sample_manifest())
            .await
            .expect_err("lost POST with zero rows must not retry POST");

        assert_eq!(err, LeaseError::Timeout);
        assert_eq!(
            fake.ops(),
            vec![RegistryOp::List, RegistryOp::Register, RegistryOp::List]
        );
        assert_eq!(
            fake.calls()
                .into_iter()
                .filter(|call| call.op == RegistryOp::Register)
                .count(),
            1
        );
        assert!(exact_name_rows(&fake.rows(), APP_NAME).is_empty());
    }
}
