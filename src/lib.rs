//! layer-kit — shared infrastructure for MeiSei layer crates (torii, satori,
//! enma, yatagarasu, fujin). Each layer stays its own deploy unit with its
//! own domain (RawItem, SensingItem, Decision, PlanBrief, ActionPacket); this
//! crate holds only the plumbing duplicated across all of them.
//!
//! `ai` is the lib-level AI provider seam (torii/satori/yatagarasu depend on
//! it from their `lib` crates) — deliberately zero heavy deps (serde-level
//! only), so a lib crate can use it without pulling axum/tokio/reqwest into
//! its tree. `error` and `id` are `#[macro_export]` macros (`layer_error!`,
//! `newtype_id!`) that generate each layer's local error enum and id
//! newtypes; `time` is the shared `Timestamp`/`now()` wall-clock primitive.
//! `auth` is the platform→tool token contract, independently available behind
//! its own feature. `openai` / `serve` are server-only infra (an OpenAI
//! Responses client and the axum/tokio server scaffold), gated behind the
//! `server` feature. `store` (feature `storage`) is SQLite object persistence a
//! layer opts into when it must survive a restart.

pub mod ai;
pub mod test_support;
pub mod time;

mod error;
mod id;

#[cfg(feature = "auth")]
pub mod auth;
#[cfg(feature = "server")]
pub mod openai;
#[cfg(feature = "server")]
pub mod serve;
#[cfg(feature = "storage")]
pub mod store;
