//! Shared wall-clock primitive for MeiSei layer crates.
//!
//! Every layer owns its own `Timestamp` so the skeleton has zero dependency
//! on daruma's `shared`; the host maps to/from daruma's timestamp when
//! wiring the layer (they are the same `chrono` type, so it is a no-op).
//! The shape is identical across layers, so it lives here once instead of
//! being hand-copied per layer.

use chrono::{DateTime, Utc};

/// Canonical UTC timestamp used across MeiSei layers.
pub type Timestamp = DateTime<Utc>;

/// Current UTC wall-clock time.
#[inline]
pub fn now() -> Timestamp {
    Utc::now()
}
