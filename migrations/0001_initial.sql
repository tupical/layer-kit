-- Generic per-layer persistence: the current state of every object the layer
-- owns, plus the append-only log of the writes that produced it. A layer's
-- domain types stay in its own crate — here they are opaque JSON under a
-- `kind` discriminator, so one schema serves all five layers.

CREATE TABLE IF NOT EXISTS objects (
    kind       TEXT NOT NULL,
    id         TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);

CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    object_id   TEXT NOT NULL,
    payload     TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_objects_kind_updated ON objects (kind, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events (kind, seq);
