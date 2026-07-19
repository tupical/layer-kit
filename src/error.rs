//! Shared local-error-type macro for MeiSei layer crates.
//!
//! Every layer replaces `daruma_shared::CoreError` with its own
//! dependency-free error enum so the skeleton compiles without pulling in
//! daruma. The shape — `String`-payload variants, a snake_case constructor
//! per variant, a matching lowercase `Display`, and `std::error::Error` — is
//! identical across layers; only the enum name and variant set differ. This
//! macro generates it instead of it being hand-copied per layer.

/// Define a layer-local error enum.
///
/// ```ignore
/// layer_kit::layer_error!(IntakeError {
///     Ai(ai, "AI provider failed or returned an unusable response."),
///     Serde(serde, "(De)serialization failure."),
///     Validation(validation, "Output failed validation (missing or invalid fields)."),
/// });
/// ```
///
/// Generates, per variant, a `$name::$ctor(impl Into<String>) -> Self`
/// constructor; a `Display` that renders `"<ctor>: <message>"`; and
/// `std::error::Error`.
#[macro_export]
macro_rules! layer_error {
    ($name:ident { $($variant:ident($ctor:ident, $doc:literal)),+ $(,)? }) => {
        #[derive(Debug)]
        pub enum $name {
            $(
                #[doc = $doc]
                $variant(String),
            )+
        }

        impl $name {
            $(
                pub fn $ctor(msg: impl Into<String>) -> Self {
                    Self::$variant(msg.into())
                }
            )+
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( Self::$variant(m) => write!(f, "{}: {m}", stringify!($ctor)), )+
                }
            }
        }

        impl std::error::Error for $name {}
    };
}

#[cfg(test)]
mod tests {
    layer_error!(TestError {
        Ai(ai, "ai failure"),
        Serde(serde, "serde failure"),
        Validation(validation, "validation failure"),
    });

    #[test]
    fn ctor_and_display_roundtrip() {
        assert_eq!(TestError::ai("boom").to_string(), "ai: boom");
        assert_eq!(TestError::serde("bad json").to_string(), "serde: bad json");
        assert_eq!(
            TestError::validation("bad field").to_string(),
            "validation: bad field"
        );
    }

    #[test]
    fn implements_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<TestError>();
    }

    #[test]
    fn subset_of_variants_is_allowed() {
        layer_error!(SmallError {
            Validation(validation, "validation failure"),
            Serde(serde, "serde failure"),
        });
        assert_eq!(SmallError::validation("x").to_string(), "validation: x");
        assert_eq!(SmallError::serde("y").to_string(), "serde: y");
    }
}
