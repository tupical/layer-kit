//! Shared UUIDv7 newtype-id macro for MeiSei layer crates.
//!
//! Each layer mints its own ids locally (never depending on
//! `daruma-shared::ids`), but the shape — UUIDv7 + human-readable prefix,
//! `serde(transparent)`, a `"<prefix>_<uuid>"` `Display`, and a `FromStr`
//! that accepts either the prefixed or bare UUID string — is identical
//! everywhere. This macro generates it instead of it being hand-copied per
//! layer.

/// Define a strongly-typed UUIDv7 id newtype.
///
/// ```ignore
/// layer_kit::newtype_id! {
///     /// Strongly-typed UUIDv7 identifier for a [`RawItem`].
///     pub struct RawItemId("ri");
/// }
/// ```
///
/// Generates `new`/`from_uuid`/`as_uuid`/`prefix`, `Default`, a `Display`
/// that renders `"<prefix>_<uuid>"`, and a `FromStr` that accepts either the
/// prefixed or bare UUID string. Requires `serde` (with the `derive`
/// feature) and `uuid` (with the `v7` feature) as direct dependencies of the
/// invoking crate.
#[macro_export]
macro_rules! newtype_id {
    ($(#[$meta:meta])* pub struct $name:ident($prefix:literal);) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd,
            serde::Serialize, serde::Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub uuid::Uuid);

        impl $name {
            #[inline]
            pub fn new() -> Self {
                Self(uuid::Uuid::now_v7())
            }

            #[inline]
            pub fn from_uuid(uuid: uuid::Uuid) -> Self {
                Self(uuid)
            }

            #[inline]
            pub fn as_uuid(&self) -> uuid::Uuid {
                self.0
            }

            #[inline]
            pub const fn prefix() -> &'static str {
                $prefix
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}_{}", $prefix, self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = uuid::Error;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let trimmed = s.strip_prefix(concat!($prefix, "_")).unwrap_or(s);
                Ok(Self(uuid::Uuid::parse_str(trimmed)?))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    newtype_id! {
        /// Test id.
        pub struct TestId("tst");
    }

    #[test]
    fn display_and_parse_roundtrip() {
        let id = TestId::new();
        let shown = id.to_string();
        assert!(shown.starts_with("tst_"), "got {shown}");
        let back: TestId = shown.parse().unwrap();
        assert_eq!(id, back);
        assert_eq!(TestId::prefix(), "tst");
    }

    #[test]
    fn from_str_accepts_bare_uuid() {
        let bare = uuid::Uuid::now_v7().to_string();
        let parsed: TestId = bare.parse().unwrap();
        assert_eq!(parsed.as_uuid().to_string(), bare);
    }

    #[test]
    fn from_uuid_wraps_given_uuid() {
        let raw = uuid::Uuid::now_v7();
        assert_eq!(TestId::from_uuid(raw).as_uuid(), raw);
    }

    #[test]
    fn default_and_as_uuid_agree_with_new() {
        let id = TestId::default();
        assert_eq!(id.as_uuid(), id.0);
    }
}
