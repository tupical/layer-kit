//! Platform→tool auth contract (C3). The mcpbox.ru platform mints a
//! short-lived token; a layer server validates it OFFLINE with a configured
//! shared key and never reads the platform's database. Boundary-clean: no
//! mcpbox dependency.
//!
//! Token format (stub): `<claims_b64url>.<sig_b64url>` where
//! `sig = HMAC-SHA256(secret, claims_b64url)`. Claims are JSON.
//!
//! NOTE (hardening path): swap the shared-secret HMAC for an asymmetric
//! signature (Ed25519) so the platform holds the private key and tools only
//! carry the public key. Until then a layer server's port MUST be bound to
//! localhost so only the co-located platform can reach it.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Claims {
    /// Cloud workspace id.
    pub workspace: String,
    /// Project id within the workspace (optional for workspace-scoped calls).
    #[serde(default)]
    pub project: Option<String>,
    /// Tool this token is audience-scoped to — must equal this service's tool.
    pub tool: String,
    /// Unix-seconds expiry.
    pub exp: i64,
}

fn sign(secret: &[u8], claims_b64: &str) -> HmacSha256 {
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(claims_b64.as_bytes());
    mac
}

/// Mint a token. The platform owns this; included here so a layer server's
/// tests (and a future `--mint` admin helper) can exercise the exact
/// contract.
// ponytail: used by tests + the planned `--mint` helper, not any bin yet.
#[allow(dead_code)]
pub fn mint(secret: &[u8], claims: &Claims) -> String {
    let claims_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).expect("claims serialize"));
    let sig = URL_SAFE_NO_PAD.encode(sign(secret, &claims_b64).finalize().into_bytes());
    format!("{claims_b64}.{sig}")
}

/// Validate signature, expiry, and audience (`tool`). `now` is unix-seconds.
/// Signature comparison is constant-time via `hmac::Mac::verify_slice`.
pub fn verify(secret: &[u8], expected_tool: &str, now: i64, token: &str) -> Option<Claims> {
    let (claims_b64, sig_b64) = token.split_once('.')?;
    let sig = URL_SAFE_NO_PAD.decode(sig_b64).ok()?;
    sign(secret, claims_b64).verify_slice(&sig).ok()?;
    let claims: Claims = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(claims_b64).ok()?).ok()?;
    if claims.exp < now || claims.tool != expected_tool {
        return None;
    }
    Some(claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims(tool: &str, exp: i64) -> Claims {
        Claims {
            workspace: "ws1".into(),
            project: Some("p1".into()),
            tool: tool.into(),
            exp,
        }
    }

    #[test]
    fn valid_token_round_trips() {
        let secret = b"platform-secret";
        let t = mint(secret, &claims("test-tool", 10_000));
        assert_eq!(
            verify(secret, "test-tool", 9_999, &t).unwrap().workspace,
            "ws1"
        );
    }

    #[test]
    fn rejects_wrong_secret_expiry_and_audience() {
        let secret = b"platform-secret";
        let t = mint(secret, &claims("test-tool", 10_000));
        assert!(verify(b"other", "test-tool", 1, &t).is_none(), "wrong secret");
        assert!(
            verify(secret, "test-tool", 10_001, &t).is_none(),
            "expired"
        );
        assert!(
            verify(secret, "other-tool", 1, &t).is_none(),
            "wrong audience"
        );
        assert!(
            verify(secret, "test-tool", 1, "garbage").is_none(),
            "malformed"
        );
    }
}
