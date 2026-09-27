-- Tenant scope (daruma 01a0d3bc): one SQLite serves every workspace, so each
-- object carries the workspace/project of the platform token that wrote it,
-- and the key includes the workspace — otherwise one tenant's upsert could
-- overwrite another's row with the same (kind, id). Rows written before this
-- migration have no known owner: they land in workspace '' (the unscoped
-- namespace), which no platform token can reach.
CREATE TABLE objects_scoped (
    workspace  TEXT NOT NULL DEFAULT '',
    project    TEXT,
    kind       TEXT NOT NULL,
    id         TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (workspace, kind, id)
);
INSERT INTO objects_scoped (kind, id, payload, created_at, updated_at)
    SELECT kind, id, payload, created_at, updated_at FROM objects;
DROP TABLE objects;
ALTER TABLE objects_scoped RENAME TO objects;
CREATE INDEX idx_objects_scope_kind_updated ON objects (workspace, kind, updated_at DESC);
