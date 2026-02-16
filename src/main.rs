//! # Fruit Shop Backend API Server
//!
//! This file contains the main logic for the Fruit Shop Single Page Application (SPA) backend.
//! It provides a RESTful API for managing fruits.
//!
//! ## Dependencies
//! This backend relies on a PostgreSQL database for data storage.
//! The connection details are configured via environment variables.
//!
//! ## Run
//! ```sh
//! DATABASE_URL="postgres://shopuser:mysecretpassword534@127.0.0.1:5432/fruitdb" \
//! OIDC_ISSUER_URL_INTERNAL="http://authserver:8082" \
//! OIDC_ISSUER_URL_EXTERNAL="http://localhost:8082" \
//! CORS_ALLOWED_ORIGINS="http://localhost:8080" \
//! SERVER_ADDRESS="0.0.0.0:8000" \
//! RUST_LOG="info" \
//! cargo run
//! ```
//!
//! ## Testing with Dummy Token
//! For development and testing purposes, you can bypass JWT authentication by using a hardcoded "dummy_token".
//!
//! ```sh
//! curl -H "Authorization: Bearer dummy_token" http://localhost:8000/api/fruits
//! ```
//!
//! ## Configuration (Environment Variables)
//! The application is configured using the following environment variables.
//!
//! - `SERVER_ADDRESS`:
//!   - **Meaning**: The address to bind the server to.
//!   - **Purpose**: Specifies the IP and port for the backend to listen on.
//!   - **Example**: `0.0.0.0:5000`
//!
//! - `DATABASE_URL`:
//!   - **Meaning**: The connection string for the PostgreSQL database.
//!   - **Purpose**: Used to connect to the database for all data operations.
//!   - **Example**: `postgres://user:password@db:5432/fruitdb`
//!
//! - `OIDC_ISSUER_URL_INTERNAL`:
//!   - **Meaning**: The internal network address of the OIDC authentication server.
//!   - **Purpose**: Used by this backend service to fetch the JWKS (JSON Web Key Set) for token validation. This URL should be accessible from within the Docker network.
//!   - **Example**: `http://authserver:8082`
//!
//! - `OIDC_ISSUER_URL_EXTERNAL`:
//!   - **Meaning**: The public-facing, external address of the OIDC authentication server.
//!   - **Purpose**: Used to validate the `iss` (issuer) claim in the JWT. This URL must match the issuer URL that clients (like the frontend) use and that is embedded in the tokens.
//!   - **Example**: `http://localhost:8082`
//!   - **Note**: Can be a comma-separated list to support multiple hostnames (e.g., `http://localhost:8082,http://192.168.10.130:8082`).
//!
//! - `CORS_ALLOWED_ORIGINS`:
//!   - **Meaning**: A comma-separated list of allowed origins for Cross-Origin Resource Sharing (CORS).
//!   - **Purpose**: Specifies which frontend URLs are permitted to make requests to this backend API.
//!   - **Example**: `http://localhost:8080,http://192.168.10.130:8080`
//!

use std::{collections::HashMap, sync::Arc};

use axum::{
    Router,
    extract::{Path, Request, State},
    http::{self, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::get,
};
use base64::{Engine as _, engine::general_purpose};
use deadpool_postgres::{Config, Pool, Runtime};
use jsonwebtoken::{DecodingKey, Validation, dangerous::insecure_decode, decode, decode_header};
use serde::{Deserialize, Serialize};
use tokio::{sync::RwLock, time::Duration};
use tokio_postgres::NoTls;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// A custom error type to handle various error kinds in the application.
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
}

/// Converts our custom AppError into an HTTP response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Auth(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AppError::Jwt(err) => (
                StatusCode::UNAUTHORIZED,
                format!("JWT Error: {:?}", err.kind()),
            ),
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

/// Represents a fruit record from the database.
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

/// Represents the claims we expect in the JWT.
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields are used by the `jsonwebtoken` library for validation
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

/// Represents the structure of the JWKS response from the auth server.
#[derive(Debug, Deserialize)]
struct Jwks {
    keys: Vec<JwkKey>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Some fields are part of the JWK standard but not used in our logic
struct JwkKey {
    kty: String,
    kid: String,
    n: String,
    e: String,
    alg: String,
    #[serde(rename = "use")]
    key_use: String,
}

/// A client to fetch and cache JWKS keys.
struct JwksClient {
    jwks_uri: String,
    keys: RwLock<HashMap<String, DecodingKey>>,
    http_client: reqwest::Client,
}

impl JwksClient {
    fn new(jwks_uri: &str) -> Self {
        Self {
            jwks_uri: jwks_uri.to_string(),
            keys: RwLock::new(HashMap::new()),
            http_client: reqwest::Client::new(),
        }
    }

    /// Gets a decoding key by its ID (kid). Fetches and caches if not present.
    async fn get_decoding_key(&self, kid: &str) -> Result<DecodingKey, AppError> {
        // First, check if the key is already in our cache.
        if let Some(key) = self.keys.read().await.get(kid) {
            info!(
                "JWKS Client: Found key with kid '{}' in cache (from {}).",
                kid, self.jwks_uri
            );
            return Ok(key.clone()); // Return the cached key.
        }

        // If not in cache, fetch the entire JWKS from the auth server.
        info!(
            "JWKS Client: Key with kid '{}' not in cache. Fetching JWKS from URL: {}",
            kid, self.jwks_uri
        );
        let jwks: Jwks = self
            .http_client
            .get(&self.jwks_uri)
            .send()
            .await?
            .json()
            .await?;
        info!(
            "JWKS Client: Successfully fetched {} keys.",
            jwks.keys.len()
        );

        // Write the fetched keys to the cache.
        let mut key_map = self.keys.write().await;
        key_map.clear();

        for key in jwks.keys {
            info!("JWKS Client: Processing key details: {:?}", key);
            if key.kty == "RSA" && key.key_use == "sig" {
                let decoding_key = DecodingKey::from_rsa_components(&key.n, &key.e)?;
                key_map.insert(key.kid.clone(), decoding_key);
            }
        }
        info!("JWKS Client: Cache updated with new keys.");

        // Try to get the key from the now-populated cache.
        if let Some(key) = key_map.get(kid) {
            info!(
                "JWKS Client: Successfully retrieved new key with kid '{}' from {}.",
                kid, self.jwks_uri
            );
            Ok(key.clone())
        } else {
            Err(AppError::Auth(format!("Unknown key ID '{}'", kid)))
        }
    }
}

/// The shared application state.
#[derive(Clone)]
struct AppState {
    db_pool: Pool,
    jwks_client: Arc<JwksClient>,
    jwt_validation: Arc<Validation>,
}

/// Axum middleware for JWT authentication.
///
/// This middleware intercepts requests to protected routes and performs two main tasks:
/// handling CORS preflight requests and validating JWTs.
///
/// ### 1. CORS Preflight Handling
/// It first checks if the request is a CORS preflight request (`OPTIONS`). If so, it immediately
/// returns an empty `200 OK` response, bypassing all authentication logic. This allows the
/// `CorsLayer` (applied later) to attach the necessary CORS headers and respond to the browser.
///
/// ### 2. JWT Validation
/// For all other requests, it performs a full JWT validation based on OIDC specifications:
///   2-1. **Extract Token**: Extracts the JWT from the `Authorization: Bearer <token>` header.
///   2-2. **Fetch Validation Key**: Decodes the token's header to get the `kid` (Key ID), then uses it to fetch the corresponding public key from the JWKS client (which includes caching).
///   2-3. **Validate Claims**: Verifies the token's signature, issuer (`iss`), audience (`aud`), and expiration time (`exp`).
///   2-4. **Pass to Next Handler**: If the token is valid, the request is passed to the next handler. If any step fails, a `401 Unauthorized` error is returned.
async fn auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    // 1. CORS Preflight Handling:
    // For OPTIONS requests, bypass authentication and return an empty OK response.
    // The `CorsLayer` will then add the necessary CORS headers.
    if request.method() == http::Method::OPTIONS {
        info!("Auth middleware: Handling CORS preflight request (OPTIONS)");
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Default::default())
            .unwrap());
    }

    // 2-1. Extract JWT Token
    info!("Auth middleware: Validating token...");
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());
    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or_else(|| {
            warn!("Auth middleware: Missing or invalid Bearer token");
            AppError::Auth("Missing or invalid Bearer token".to_string())
        })?;

    info!("Auth middleware: Received token: {}", token);

    // Test bypass: Allow a specific dummy token to skip validation
    if token == "dummy_token" {
        info!("Auth middleware: Dummy token detected. Bypassing validation.");
        return Ok(next.run(request).await);
    }

    // For debugging, decode and display the payload before validation.
    if let Some(payload_b64) = token.split('.').nth(1) {
        match general_purpose::URL_SAFE_NO_PAD.decode(payload_b64) {
            Ok(decoded_payload_bytes) => {
                if let Ok(payload_str) = String::from_utf8(decoded_payload_bytes) {
                    info!(
                        "Auth middleware: Decoded payload (for inspection): {}",
                        payload_str
                    );
                } else {
                    warn!("Auth middleware: Payload is not valid UTF-8");
                }
            }
            Err(e) => {
                warn!("Auth middleware: Failed to Base64 decode payload: {}", e);
            }
        }
    }

    // 2-2. Fetch Validation Key (Part 1: Get Key ID from header)
    let header = decode_header(token)?;
    let kid = header
        .kid
        .ok_or_else(|| AppError::Auth("Token missing 'kid' in header".to_string()))?;
    info!("Auth middleware: Found token with kid '{}'", kid);

    // 2-2. Fetch Validation Key (Part 2: Get public key from JWKS client)
    let decoding_key = state.jwks_client.get_decoding_key(&kid).await?;

    // Log token expiration details
    if let Ok(token_data) = insecure_decode::<Claims>(token) {
        let exp = token_data.claims.exp;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as usize;
        let remaining = if exp > now {
            (exp - now) as isize
        } else {
            -((now - exp) as isize)
        };

        info!(
            "Auth middleware: Token Expiration Check: exp={}, now={}, remaining={}s",
            exp, now, remaining
        );
    }

    // 2-3. Validate Claims
    // This single `decode` call verifies the signature, issuer, audience, and expiration time
    // based on the `Validation` settings configured in `main`.
    info!(
        "Auth middleware: Validating token with expected values: issuers={:?}, audience={:?}, validate_exp={}",
        state.jwt_validation.iss, state.jwt_validation.aud, state.jwt_validation.validate_exp
    );

    // Decode the token and handle potential validation errors for better logging.
    let _token_data = match decode::<Claims>(token, &decoding_key, &state.jwt_validation) {
        Ok(data) => {
            info!(
                "Auth middleware: Token validated successfully with claims: {:?}",
                data.claims
            );
            data
        }
        Err(err) => {
            // Log the kid to make debugging signature errors easier.
            if let jsonwebtoken::errors::ErrorKind::InvalidSignature = err.kind() {
                error!(
                    "Auth middleware: Invalid signature for token with kid '{}'",
                    kid
                );
            }
            // If the error is an invalid issuer, log the expected vs actual values.
            if let jsonwebtoken::errors::ErrorKind::InvalidIssuer = err.kind() {
                // To inspect claims for logging, decode the token without verifying the signature.
                let unverified_claims = insecure_decode::<Claims>(token)
                    .map(|data| data.claims)
                    .ok();
                let actual_issuer = unverified_claims
                    .map(|c| c.iss)
                    .unwrap_or_else(|| "unknown".to_string());

                error!(
                    "Auth middleware: Invalid issuer. Expected one of: {:?}, but got: '{}'",
                    state.jwt_validation.iss, actual_issuer
                );
            }
            // Propagate the original error.
            return Err(err.into());
        }
    };

    // 2-4. Pass to Next Handler
    Ok(next.run(request).await)
}

/// Handler for the root endpoint.
async fn root_handler() -> &'static str {
    "Backend is running!"
}

/// Handler for the protected /api/fruits endpoint.
async fn get_fruits_handler(State(state): State<AppState>) -> Result<Json<Vec<Fruit>>, AppError> {
    let client = state.db_pool.get().await?;
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
    Ok(Json(fruits))
}

/// Handler for the protected /api/fruits/:id endpoint.
async fn get_fruit_by_id_handler(
    State(state): State<AppState>,
    Path(fruit_id): Path<String>,
) -> Result<Json<Fruit>, AppError> {
    info!("Fetching fruit with id: {}", fruit_id);
    let client = state.db_pool.get().await?;
    let row_opt = client
        .query_opt(
            "SELECT id, name, origin, price, description, origin_latitude::FLOAT8 AS origin_latitude, origin_longitude::FLOAT8 AS origin_longitude FROM fruit WHERE id = $1",
            &[&fruit_id],
        )
        .await?;

    if let Some(row) = row_opt {
        let fruit = Fruit {
            id: row.get("id"),
            name: row.get("name"),
            origin: row.get("origin"),
            price: row.get("price"),
            description: row.get("description"),
            origin_latitude: row.get("origin_latitude"),
            origin_longitude: row.get("origin_longitude"),
        };
        Ok(Json(fruit))
    } else {
        Err(AppError::NotFound(format!(
            "Fruit with id '{}' not found",
            fruit_id
        )))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file if it exists.
    dotenvy::dotenv().ok();

    // Setup tracing subscriber for logging.
    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug".into());
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(&rust_log))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // --- Configuration ---
    // These settings are configured via environment variables, typically set in a Docker Compose file.
    let db_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let oidc_issuer_internal =
        std::env::var("OIDC_ISSUER_URL_INTERNAL").expect("OIDC_ISSUER_URL_INTERNAL must be set");
    let oidc_issuer_external =
        std::env::var("OIDC_ISSUER_URL_EXTERNAL").expect("OIDC_ISSUER_URL_EXTERNAL must be set");
    let allowed_origins_str =
        std::env::var("CORS_ALLOWED_ORIGINS").expect("CORS_ALLOWED_ORIGINS must be set");
    let server_address = std::env::var("SERVER_ADDRESS").expect("SERVER_ADDRESS must be set");
    let jwks_uri = format!("{}/jwks.json", oidc_issuer_internal);

    info!("--- Application Configuration ---");
    info!("RUST_LOG: {}", rust_log);
    info!("DATABASE_URL: {}", db_url);
    info!("OIDC_ISSUER_URL_INTERNAL: {}", oidc_issuer_internal);
    info!("OIDC_ISSUER_URL_EXTERNAL: {}", oidc_issuer_external);
    info!("CORS_ALLOWED_ORIGINS: '{}'", allowed_origins_str);
    info!("SERVER_ADDRESS: {}", server_address);
    info!("---------------------------------");

    // --- Database Pool ---
    let mut cfg = Config::new();
    cfg.url = Some(db_url);
    cfg.pool = Some(deadpool_postgres::PoolConfig {
        max_size: 10,
        timeouts: deadpool_postgres::Timeouts {
            wait: Some(Duration::from_secs(5)), // Set the timeout for getting a connection from the pool.
            ..Default::default()
        },
        ..Default::default()
    });
    // For simplicity in this demo, we use NoTls. For production, configure TLS.
    let db_pool = cfg.create_pool(Some(Runtime::Tokio1), NoTls)?;
    info!("Database connection pool established.");

    // --- JWT Validation Setup ---
    let external_issuers: Vec<String> = oidc_issuer_external
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let mut validation = Validation::new(jsonwebtoken::Algorithm::RS256);
    validation.set_issuer(&external_issuers);
    validation.set_audience(&["fruit-shop"]);
    info!(
        "JWT validation configured: issuers={:?}, audience='{}'",
        external_issuers, "fruit-shop"
    );

    // --- App State ---
    let app_state = AppState {
        db_pool,
        jwks_client: Arc::new(JwksClient::new(&jwks_uri)),
        jwt_validation: Arc::new(validation),
    };

    // --- CORS Layer ---
    let allowed_origins = allowed_origins_str
        .split(',')
        .map(|origin| origin.trim().parse().expect("Failed to parse origin"))
        .collect::<Vec<_>>();

    let cors_layer = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods(Any)
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]);

    // --- Router ---
    let api_routes = Router::new()
        .route("/fruits", get(get_fruits_handler)) // For the list
        .route("/fruits/{id}", get(get_fruit_by_id_handler)) // For a single item
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth_middleware,
        ));

    let app = Router::new()
        .route("/", get(root_handler))
        .nest("/api", api_routes)
        .with_state(app_state)
        .layer(cors_layer)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    // --- Start Server ---
    let listener = tokio::net::TcpListener::bind(&server_address).await?;
    info!("Backend server listening on {}", server_address);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Listens for the OS shutdown signal (e.g., Ctrl+C or SIGTERM).
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

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Signal received, starting graceful shutdown");
}
