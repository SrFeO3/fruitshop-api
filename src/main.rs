//! # Fruit Shop Backend API Server
//!
//! A lightweight utility for development and testing, compatible with apiserver.py.
//!
//! [API Endpoints]
//!   GET */api/add/<x>/<y> (No DB required)
//!   - Returns: {"result": x+y, "comment": "...", "user": "..."}
//!
//!   GET */api/fruits (DB required)
//!   - Returns: {"fruits": [{id, name, ...}, ...]}
//!
//!   GET */api/fruits/<id> (DB required)
//!   - Returns: {id, name, ...}
//!
//! [Environment Variables]
//!   - API_HOST: Bind address (default: 0.0.0.0)
//!   - API_PORTS: Comma/space separated list of ports (default: 7444)
//!   - DATABASE_URL: PostgreSQL connection string (or use DB_HOST, DB_PORT, etc.)
//!   - OIDC_ISSUER_URL_INTERNAL: Discovery URL for JWKS.
//!   - OIDC_ISSUER_URL_EXTERNAL: Allowed 'iss' claims in tokens.
//!   - CORS_ALLOWED_ORIGINS: Permitted origins for CORS.
//!
//! [Token Processing]
//!   - Requires Bearer token in Authorization header.
//!   - JWS (Signed): Validates RS256 or EdDSA signatures using JWKS from OIDC Discovery.
//!   - JWE (Encrypted): Detected but returns 401 (Not configured).
//!   - Opaque: Tokens not in JWT format are treated as valid for testing.
//!
//! [Features]
//!   - Flexible Path Configuration: Handles arbitrary path prefixes (wildcard *).
//!   - Resilient DB Connection: Starts in DB-less mode if connection fails and retries on demand.

use std::{collections::HashMap, sync::Arc};

use axum::{
    Router,
    extract::{Path, Request, State},
    http::{self, StatusCode, Uri, header},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::get,
};
use deadpool_postgres::{Config, Pool, Runtime};
use jsonwebtoken::{DecodingKey, Validation, decode, decode_header};
use serde::{Deserialize, Serialize};
use tokio::{sync::RwLock, time::Duration};
use tokio_postgres::NoTls;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("Database pool error: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),
    #[error("Postgres error: {0}")]
    Postgres(#[from] tokio_postgres::Error),
    #[error("JWT error: {0}")]
    Jwt(#[from] jsonwebtoken::errors::Error),
    #[error("Resource not found: {0}")]
    NotFound(String),
    #[error("Request error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("Authentication failed: {0}")]
    Auth(String),
    #[error("Database not configured or connection failed")]
    DbDisabled,
    #[error("OIDC Discovery failed: {0}")]
    Discovery(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Auth(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AppError::Jwt(err) => (
                StatusCode::UNAUTHORIZED,
                format!("JWT Error: {:?}", err.kind()),
            ),
            AppError::DbDisabled | AppError::Discovery(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            AppError::Pool(_) | AppError::Postgres(_) | AppError::Reqwest(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };
        error!(
            "Responding with error: status={}, message='{}', details='{:?}'",
            status, message, self
        );
        (status, message).into_response()
    }
}

#[derive(Serialize)]
struct Fruit {
    id: String,
    name: String,
    origin: String,
    price: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    origin_latitude: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    origin_longitude: Option<f64>,
}

#[derive(Serialize)]
struct FruitsResponse {
    fruits: Vec<Fruit>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Claims {
    iss: String,
    aud: String,
    sub: String,
    exp: usize,
    iat: Option<usize>,
    jti: Option<String>,
    scope: Option<String>,
    client_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Jwks {
    keys: Vec<JwkKey>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JwkKey {
    kty: String,
    kid: String,
    x: Option<String>,
    crv: Option<String>,
    n: Option<String>,
    e: Option<String>,
    #[serde(rename = "use")]
    key_use: String,
}

struct JwksClient {
    jwks_uri: String,
    keys: RwLock<HashMap<String, DecodingKey>>,
    http_client: reqwest::Client,
}

/// Extracts claims without signature verification for logging purposes.
fn decode_claims_unverified(token: &str) -> Option<Claims> {
    use base64::prelude::*;
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let decoded = BASE64_URL_SAFE_NO_PAD.decode(parts[1]).ok()?;
    serde_json::from_slice(&decoded).ok()
}

impl JwksClient {
    fn new(jwks_uri: &str) -> Self {
        Self {
            jwks_uri: jwks_uri.to_string(),
            keys: RwLock::new(HashMap::new()),
            http_client: reqwest::Client::new(),
        }
    }

    async fn get_decoding_key(&self, kid: &str) -> Result<DecodingKey, AppError> {
        if let Some(key) = self.keys.read().await.get(kid) {
            info!("JWKS Client: Found key with kid '{}' in cache.", kid);
            return Ok(key.clone());
        }

        info!(
            "JWKS Client: Key with kid '{}' not in cache. Fetching...",
            kid
        );
        let resp_text = self
            .http_client
            .get(&self.jwks_uri)
            .send()
            .await?
            .text()
            .await?;

        let jwks: Jwks =
            serde_json::from_str(&resp_text).map_err(|e| AppError::Discovery(e.to_string()))?;

        let mut key_map = self.keys.write().await;

        // Double-check if another thread updated the cache while waiting for the lock
        if let Some(key) = key_map.get(kid) {
            return Ok(key.clone());
        }

        key_map.clear();

        for key in jwks.keys {
            if key.key_use == "sig" {
                if key.kty == "OKP" && key.crv.as_deref() == Some("Ed25519") {
                    if let Some(x) = &key.x {
                        key_map.insert(key.kid.clone(), DecodingKey::from_ed_components(x)?);
                    }
                } else if key.kty == "RSA" {
                    if let (Some(n), Some(e)) = (&key.n, &key.e) {
                        key_map.insert(key.kid.clone(), DecodingKey::from_rsa_components(n, e)?);
                    }
                }
            }
        }

        if let Some(key) = key_map.get(kid) {
            Ok(key.clone())
        } else {
            Err(AppError::Auth(format!("Unknown key ID '{}'", kid)))
        }
    }
}

#[derive(Clone)]
struct AppState {
    db_url: Option<String>,
    db_pool: Arc<RwLock<Option<Pool>>>,
    jwks_client: Arc<JwksClient>,
    jwt_validation: Arc<Validation>,
}

/// Returns the existing DB pool or attempts to initialize a new one if missing.
async fn get_or_init_db_pool(state: &AppState) -> Result<Pool, AppError> {
    if let Some(pool) = state.db_pool.read().await.as_ref() {
        return Ok(pool.clone());
    }

    let url = state.db_url.as_ref().ok_or(AppError::DbDisabled)?;

    let mut lock = state.db_pool.write().await;

    // Double-check pattern
    if let Some(pool) = lock.as_ref() {
        return Ok(pool.clone());
    }

    info!("Database pool is missing. Attempting to initialize pool...");
    let mut cfg = Config::new();
    cfg.url = Some(url.clone());
    cfg.pool = Some(deadpool_postgres::PoolConfig {
        max_size: 10,
        timeouts: deadpool_postgres::Timeouts {
            wait: Some(Duration::from_secs(5)),
            ..Default::default()
        },
        ..Default::default()
    });

    match cfg.create_pool(Some(Runtime::Tokio1), NoTls) {
        Ok(pool) => {
            *lock = Some(pool.clone());
            info!("Database pool initialized successfully.");
            Ok(pool)
        }
        Err(e) => {
            error!("Failed to initialize database pool: {}", e);
            Err(AppError::DbDisabled)
        }
    }
}

#[derive(Clone)]
struct TokenValidationInfo(String);

async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, AppError> {
    if request.method() == http::Method::OPTIONS {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Default::default())
            .unwrap());
    }

    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());
    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| AppError::Auth("Missing or invalid Bearer token".to_string()))?;

    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() == 5 {
        return Err(AppError::Auth(
            "Received JWE token but JWE_DECRYPTION_KEY is not configured.".to_string(),
        ));
    } else if parts.len() != 3 {
        request.extensions_mut().insert(TokenValidationInfo(
            "Key: Opaque, Validation: Skipped".to_string(),
        ));
        return Ok(next.run(request).await);
    }

    if token == "dummy_token" {
        request.extensions_mut().insert(TokenValidationInfo(
            "Key: Opaque, Validation: Skipped".to_string(),
        ));
        return Ok(next.run(request).await);
    }

    let header = decode_header(token)?;
    let kid = header
        .kid
        .ok_or_else(|| AppError::Auth("Token missing 'kid' in header".to_string()))?;
    let decoding_key = state.jwks_client.get_decoding_key(&kid).await?;

    if let Some(claims) = decode_claims_unverified(token) {
        let exp = claims.exp;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as usize;
        debug!(
            " > Token exp={}, remaining={}",
            exp,
            exp as isize - now as isize
        );
    }

    let _token_data = match decode::<Claims>(token, &decoding_key, &state.jwt_validation) {
        Ok(data) => data,
        Err(err) => {
            if let jsonwebtoken::errors::ErrorKind::InvalidIssuer = err.kind() {
                let actual_issuer = decode_claims_unverified(token)
                    .map(|c| c.iss)
                    .unwrap_or_else(|| "unknown".to_string());
                error!(
                    " > VALIDATION FAILED: Invalid issuer. Expected: {:?}, Got: '{}'",
                    state.jwt_validation.iss, actual_issuer
                );
            }
            return Err(err.into());
        }
    };

    request.extensions_mut().insert(TokenValidationInfo(
        "Key: JWT, Validation: Signature Verified".to_string(),
    ));
    Ok(next.run(request).await)
}

async fn root_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "message": "Backend is running!" }))
}

async fn get_fruits_handler(
    State(state): State<AppState>,
) -> Result<Json<FruitsResponse>, AppError> {
    let pool = get_or_init_db_pool(&state).await?;
    let client = pool.get().await?;
    let rows = client
        .query("SELECT id, name, origin, price FROM fruit ORDER BY id", &[])
        .await?;
    let fruits: Vec<Fruit> = rows
        .iter()
        .map(|row| Fruit {
            id: row.get("id"),
            name: row.get("name"),
            origin: row.get("origin"),
            price: row.get("price"),
            description: None,
            origin_latitude: None,
            origin_longitude: None,
        })
        .collect();

    Ok(Json(FruitsResponse { fruits }))
}

async fn get_fruit_by_id_handler(
    State(state): State<AppState>,
    Path(fruit_id): Path<String>,
) -> Result<Json<Fruit>, AppError> {
    let pool = get_or_init_db_pool(&state).await?;
    let client = pool.get().await?;
    let row_opt = client.query_opt(
        "SELECT id, name, origin, price, description, origin_latitude::FLOAT8 AS origin_latitude, origin_longitude::FLOAT8 AS origin_longitude FROM fruit WHERE id = $1",
        &[&fruit_id],
    ).await?;

    if let Some(row) = row_opt {
        Ok(Json(Fruit {
            id: row.get("id"),
            name: row.get("name"),
            origin: row.get("origin"),
            price: row.get("price"),
            description: row.get("description"),
            origin_latitude: row.get("origin_latitude"),
            origin_longitude: row.get("origin_longitude"),
        }))
    } else {
        Err(AppError::NotFound(format!(
            "Fruit with id '{}' not found",
            fruit_id
        )))
    }
}

async fn fallback_handler(uri: http::Uri) -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "Not Found",
            "message": format!("The requested path '{}' does not exist on this server.", uri.path())
        })),
    )
}

async fn add_handler(Path((x, y)): Path<(i32, i32)>, request: Request) -> Json<serde_json::Value> {
    let bff_user = request
        .headers()
        .get("X-BFF-IDToken-Sub")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("None");
    let comment = request
        .extensions()
        .get::<TokenValidationInfo>()
        .map(|info| info.0.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    Json(serde_json::json!({ "result": x + y, "comment": comment, "user": bff_user }))
}

/// Normalizes the request URI by stripping any prefix before '/api/'.
fn rewrite_api_path(mut req: Request) -> Request {
    let path = req.uri().path();

    if let Some(pos) = path.find("/api/") {
        if pos != 0 {
            let new_path = &path[pos..];

            // Preserve query parameters (e.g., ?foo=bar)
            let query = req
                .uri()
                .query()
                .map(|q| format!("?{}", q))
                .unwrap_or_default();
            let new_pq = format!("{}{}", new_path, query);

            let mut parts = req.uri().clone().into_parts();
            parts.path_and_query = Some(new_pq.parse().unwrap());

            *req.uri_mut() = Uri::from_parts(parts).unwrap();
        }
    }
    req
}

async fn path_normalization_middleware(req: Request, next: axum::middleware::Next) -> Response {
    let req = rewrite_api_path(req);
    next.run(req).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=info".into());
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(&rust_log))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let api_host = std::env::var("API_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let api_ports_str = std::env::var("API_PORTS").unwrap_or_else(|_| "7444".to_string());
    let ports: Vec<String> = api_ports_str
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    let db_url = std::env::var("DATABASE_URL").ok().or_else(|| {
        let host = std::env::var("DB_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("DB_PORT").unwrap_or_else(|_| "5432".to_string());
        let name = std::env::var("DB_NAME").unwrap_or_else(|_| "shopdb".to_string());
        let user = std::env::var("DB_USER").unwrap_or_else(|_| "shopuser".to_string());
        let password = std::env::var("DB_PASSWORD").unwrap_or_else(|_| "myShopPwABC".to_string());
        Some(format!(
            "postgresql://{user}:{password}@{host}:{port}/{name}"
        ))
    });

    let oidc_issuer_internal =
        std::env::var("OIDC_ISSUER_URL_INTERNAL").expect("OIDC_ISSUER_URL_INTERNAL must be set");
    let oidc_issuer_external =
        std::env::var("OIDC_ISSUER_URL_EXTERNAL").expect("OIDC_ISSUER_URL_EXTERNAL must be set");
    let allowed_origins_str =
        std::env::var("CORS_ALLOWED_ORIGINS").expect("CORS_ALLOWED_ORIGINS must be set");
    let oidc_audience = std::env::var("OIDC_AUDIENCE").unwrap_or_else(|_| "fruit-shop".to_string());
    let oidc_algo_str = std::env::var("OIDC_ALGORITHM").unwrap_or_else(|_| "RS256".to_string());
    let algorithm = match oidc_algo_str.as_str() {
        "EdDSA" => jsonwebtoken::Algorithm::EdDSA,
        _ => jsonwebtoken::Algorithm::RS256,
    };

    let discovery_client = reqwest::Client::new();
    let discovery_resp: serde_json::Value = discovery_client
        .get(format!(
            "{}/.well-known/openid-configuration",
            oidc_issuer_internal.trim_end_matches('/')
        ))
        .send()
        .await?
        .json()
        .await?;
    let jwks_uri = discovery_resp["jwks_uri"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let (db_pool, db_url_to_store) = if let Some(url) = db_url {
        let mut cfg = Config::new();
        cfg.url = Some(url.clone());
        cfg.pool = Some(deadpool_postgres::PoolConfig {
            max_size: 10,
            timeouts: deadpool_postgres::Timeouts {
                wait: Some(Duration::from_secs(5)),
                ..Default::default()
            },
            ..Default::default()
        });
        match cfg.create_pool(Some(Runtime::Tokio1), NoTls) {
            Ok(pool) => (Some(pool), Some(url)),
            Err(e) => {
                error!(
                    "Database connection failed at startup: {}. The server will start in DB-less mode and retry connection on demand.",
                    e
                );
                (None, Some(url))
            }
        }
    } else {
        info!("Database is not configured. DB features will be disabled.");
        (None, None)
    };

    let external_issuers: Vec<String> = oidc_issuer_external
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let mut validation = Validation::new(algorithm);
    validation.set_issuer(&external_issuers);
    validation.set_audience(&[&oidc_audience]);

    let app_state = AppState {
        db_url: db_url_to_store,
        db_pool: Arc::new(RwLock::new(db_pool)),
        jwks_client: Arc::new(JwksClient::new(&jwks_uri)),
        jwt_validation: Arc::new(validation),
    };

    let allowed_origins = allowed_origins_str
        .split(',')
        .map(|origin| origin.trim().parse().expect("Failed to parse origin"))
        .collect::<Vec<_>>();

    let cors_layer = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods(Any)
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]);

    let api_router = Router::new()
        .route("/", get(root_handler))
        .route("/api/add/{x}/{y}", get(add_handler))
        .route("/api/fruits", get(get_fruits_handler))
        .route("/api/fruits/{id}", get(get_fruit_by_id_handler))
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth_middleware,
        ))
        .fallback(fallback_handler)
        .with_state(app_state);

    // Apply path normalization and CORS to the entire application
    let app = Router::new()
        .fallback_service(api_router)
        .layer(middleware::from_fn(path_normalization_middleware))
        .layer(cors_layer)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let mut handles = Vec::new();
    for port in ports {
        let server_address = format!("{}:{}", api_host, port);
        let listener = tokio::net::TcpListener::bind(&server_address).await?;
        let app_clone = app.clone();

        info!("Backend server listening on {}", server_address);
        let handle = tokio::spawn(async move {
            axum::serve(listener, app_clone)
                .with_graceful_shutdown(shutdown_signal())
                .await
        });
        handles.push(handle);
    }

    for handle in handles {
        let _ = handle.await?;
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! { _ = ctrl_c => {}, _ = terminate => {}, }
    info!("Signal received, starting graceful shutdown");
}
