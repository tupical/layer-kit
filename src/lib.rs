//! layer-kit — shared infrastructure for MeiSei layer crates (torii, satori,
//! enma, yatagarasu, fujin). Each layer stays its own deploy unit with its
//! own domain (RawItem, SensingItem, Decision, PlanBrief, ActionPacket); this
//! crate holds only the plumbing duplicated across all of them.
//!
//! `ai` is the lib-level AI provider seam (torii/satori/yatagarasu depend on
//! it from their `lib` crates) — deliberately zero heavy deps (serde-level
//! only), so a lib crate can use it without pulling axum/tokio/reqwest into
//! its tree. `auth` / `openai` / `serve` are server-only infra (the
//! platform→tool token contract, an OpenAI Responses client, and the
//! axum/tokio server scaffold), gated behind the `server` feature and used
//! only from each layer's `server` binary crate.

pub mod ai;

#[cfg(feature = "server")]
pub mod auth;
#[cfg(feature = "server")]
pub mod openai;
#[cfg(feature = "server")]
pub mod serve;
