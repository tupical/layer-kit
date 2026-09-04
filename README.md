# Layer Kit

## Updating to 0.2.0

Before the first restart, back up every service's `{TOOL}_DB`. Migration 0002
irreversibly drops the unused `events` table and has no down migration.

Dropping the table does not shrink the database file. After the restart, run
`VACUUM` manually on each database if reclaiming disk space is desired; it is
intentionally not part of the transactional sqlx migration.

## Strict structured output

Function tools opt in with `"strict": true`. Layer Kit preserves that flag for
both Responses and Chat Completions and closes every object schema recursively.
Tools without the flag are sent unchanged for compatibility with providers
that do not implement OpenAI strict schemas. Token-budget exhaustion and
invalid response schemas are distinguished by `AiError::kind()`.
