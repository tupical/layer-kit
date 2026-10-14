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
invalid response schemas are distinguished by `AiError::kind()`, an explicit
field set by the constructor — never parsed from the message. Wrap with
`AiError::context()` to keep the kind; a body that fails to decode is a
transport fault (`Other`), not a schema error.

## AI protocols

`AiConfig.api_protocol` (wire value of `ai.api_protocol` or env
`OPENAI_API_PROTOCOL`) selects the provider dialect: `responses` (default),
`chat_completions` or `anthropic_messages`. An unknown value rejects the AI
config instead of falling back. `anthropic_messages` posts to
`{base}/v1/messages` (base is the bare host, default
`https://api.anthropic.com`; a trailing `/v1` is tolerated) with `x-api-key`
and `anthropic-version: 2023-06-01` instead of Bearer, always sends
`max_tokens`, and lifts `system` messages of an array input into the
top-level `system` field. `text` and `tool_use` blocks become outputs; a
`stop_reason` of `max_tokens` is an `OutputBudget` error. Prompt caching and
message batches are not supported here.
