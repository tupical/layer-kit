//! SQLite object persistence shared by every MeiSei layer server.
//!
//! Deliberately domain-blind — a layer stores its own typed object as JSON
//! under its own `kind` (`"raw_item"`, `"decision"`, ...) and deserializes on
//! read. That keeps one schema (and one set of migrations) serving all five
//! layers; a layer that outgrows it can still open [`Store::pool`] and run
//! its own SQL.
//!
//! Every read and write is confined to one workspace (and, when the caller
//! has one, one project): a layer server is shared by all tenants, so a
//! handler must work through [`Store::scoped`] with the verified token
//! claims. The bare store is the unscoped namespace `''` — tests and rows
//! that predate scoping; no platform token reaches it. README «Migration
//! 0003» gives operators the SQL to hand such rows to their workspace.
//!
//! Env: `{TOOL}_DB` (see [`Store::from_env`]) — the file survives restarts,
//! which is the whole point of this module.

use serde::de::DeserializeOwned;
use serde::Serialize;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;

/// A layer's SQLite store. Cheap to clone (the pool is shared).
#[derive(Clone)]
pub struct Store {
    pool: SqlitePool,
    workspace: String,
    project: Option<String>,
}

impl Store {
    /// Open (creating if absent) the database at `path` and run migrations.
    pub async fn open(path: &str) -> Result<Self, sqlx::Error> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            // WAL + NORMAL: concurrent readers during a write, without the
            // fsync-per-commit cost. A layer server is a single writer.
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self {
            pool,
            workspace: String::new(),
            project: None,
        })
    }

    /// Open the path in `{TOOL}_DB`, defaulting to `./{tool}.db`.
    pub async fn from_env(tool: &str) -> Result<Self, sqlx::Error> {
        let path = std::env::var(format!("{}_DB", tool.to_uppercase()))
            .unwrap_or_else(|_| format!("{tool}.db"));
        Self::open(&path).await
    }

    /// The same store confined to one tenant: pass the verified token's
    /// `workspace` / `project`. Objects are keyed by workspace; a
    /// project-scoped caller also sees workspace-wide objects (no project),
    /// but not other projects'. On write the project is provenance, not
    /// isolation: the key is (workspace, kind, id), so within one workspace a
    /// caller of another project upserting the same id overwrites the payload.
    pub fn scoped(&self, workspace: &str, project: Option<&str>) -> Self {
        Self {
            pool: self.pool.clone(),
            workspace: workspace.to_owned(),
            project: project.map(str::to_owned),
        }
    }

    /// Escape hatch for a layer whose queries outgrow this module.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Upsert an object. `created_at` survives an overwrite; `updated_at` moves.
    pub async fn put<T: Serialize>(
        &self,
        kind: &str,
        id: &str,
        object: &T,
    ) -> Result<(), sqlx::Error> {
        let payload = serde_json::to_string(object).map_err(encode_err)?;
        let now = crate::time::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO objects (kind, id, payload, created_at, updated_at, workspace, project)
             VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6)
             ON CONFLICT (workspace, kind, id) DO UPDATE SET payload = ?3, updated_at = ?4",
        )
        .bind(kind)
        .bind(id)
        .bind(&payload)
        .bind(&now)
        .bind(&self.workspace)
        .bind(&self.project)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Fetch one object, `None` when absent.
    pub async fn get<T: DeserializeOwned>(
        &self,
        kind: &str,
        id: &str,
    ) -> Result<Option<T>, sqlx::Error> {
        let row = sqlx::query(&format!(
            "SELECT payload FROM objects WHERE kind = ?1 AND id = ?2 AND {SCOPE}"
        ))
        .bind(kind)
        .bind(id)
        .bind(&self.workspace)
        .bind(&self.project)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| serde_json::from_str(r.get("payload")).map_err(decode_err))
            .transpose()
    }

    /// Objects of one kind, most recently updated first.
    pub async fn list<T: DeserializeOwned>(
        &self,
        kind: &str,
        limit: i64,
    ) -> Result<Vec<T>, sqlx::Error> {
        sqlx::query(&format!(
            "SELECT payload FROM objects WHERE kind = ?1 AND {SCOPE}
             ORDER BY updated_at DESC, id DESC LIMIT ?2"
        ))
        .bind(kind)
        .bind(limit)
        .bind(&self.workspace)
        .bind(&self.project)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|r| serde_json::from_str(r.get("payload")).map_err(decode_err))
        .collect()
    }

    /// Delete an object. `true` when a row was removed.
    pub async fn delete(&self, kind: &str, id: &str) -> Result<bool, sqlx::Error> {
        let done = sqlx::query(&format!(
            "DELETE FROM objects WHERE kind = ?1 AND id = ?2 AND {SCOPE}"
        ))
        .bind(kind)
        .bind(id)
        .bind(&self.workspace)
        .bind(&self.project)
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected() > 0)
    }
}

/// Tenant filter shared by every read/delete; binds `?3` = workspace,
/// `?4` = project.
const SCOPE: &str = "workspace = ?3 AND (?4 IS NULL OR project IS NULL OR project = ?4)";

fn encode_err(e: serde_json::Error) -> sqlx::Error {
    sqlx::Error::Encode(Box::new(e))
}

fn decode_err(e: serde_json::Error) -> sqlx::Error {
    sqlx::Error::Decode(Box::new(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Item {
        id: String,
        body: String,
    }

    fn temp_db() -> String {
        std::env::temp_dir()
            .join(format!("layer-kit-{}.db", uuid::Uuid::now_v7()))
            .to_string_lossy()
            .into_owned()
    }

    /// The task's acceptance criterion: a standalone layer survives a restart.
    #[tokio::test]
    async fn objects_survive_reopen() {
        let path = temp_db();
        let first = Store::open(&path).await.unwrap();
        let item = Item {
            id: "raw_1".into(),
            body: "hello".into(),
        };
        first.put("raw_item", &item.id, &item).await.unwrap();
        drop(first);

        // Reopen the same file — migrations are idempotent, data is still there.
        let reopened = Store::open(&path).await.unwrap();
        assert_eq!(
            reopened.get::<Item>("raw_item", "raw_1").await.unwrap(),
            Some(item)
        );
        let n: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='events'",
        )
        .fetch_one(reopened.pool())
        .await
        .unwrap();
        assert_eq!(n, 0, "0002 must drop the events table");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn upsert_lists_by_kind_and_deletes() {
        let path = temp_db();
        let store = Store::open(&path).await.unwrap();
        store
            .put(
                "raw_item",
                "a",
                &Item {
                    id: "a".into(),
                    body: "one".into(),
                },
            )
            .await
            .unwrap();
        store
            .put(
                "decision",
                "d",
                &Item {
                    id: "d".into(),
                    body: "other kind".into(),
                },
            )
            .await
            .unwrap();
        // Overwrite the existing object.
        store
            .put(
                "raw_item",
                "a",
                &Item {
                    id: "a".into(),
                    body: "two".into(),
                },
            )
            .await
            .unwrap();

        let raw: Vec<Item> = store.list("raw_item", 10).await.unwrap();
        assert_eq!(raw.len(), 1, "upsert must not duplicate the object");
        assert_eq!(raw[0].body, "two");
        assert!(store.delete("raw_item", "a").await.unwrap());
        assert!(!store.delete("raw_item", "a").await.unwrap(), "already gone");
        assert!(store.get::<Item>("raw_item", "a").await.unwrap().is_none());
        assert_eq!(
            store.list::<Item>("decision", 10).await.unwrap().len(),
            1,
            "delete must not cross kinds"
        );
        std::fs::remove_file(&path).ok();
    }

    /// daruma 01a0d3bc: one SQLite serves every tenant — a workspace sees only
    /// its own objects, a project only its own plus workspace-wide ones, and a
    /// colliding (kind, id) from another tenant never overwrites.
    #[tokio::test]
    async fn scoped_store_isolates_workspaces_and_projects() {
        let path = temp_db();
        let store = Store::open(&path).await.unwrap();
        let item = |body: &str| Item {
            id: "x".into(),
            body: body.into(),
        };
        let a = store.scoped("ws_a", None);
        let a_p1 = store.scoped("ws_a", Some("p1"));
        let a_p2 = store.scoped("ws_a", Some("p2"));
        let b = store.scoped("ws_b", None);
        a.put("raw_item", "x", &item("a")).await.unwrap();
        b.put("raw_item", "x", &item("b")).await.unwrap();
        a_p1.put("raw_item", "y", &item("a-p1")).await.unwrap();
        store.put("raw_item", "z", &item("legacy")).await.unwrap();

        let bodies = |v: Vec<Item>| v.into_iter().map(|i| i.body).collect::<Vec<_>>();
        assert_eq!(bodies(b.list("raw_item", 10).await.unwrap()), ["b"]);
        assert_eq!(
            bodies(a.list("raw_item", 10).await.unwrap()),
            ["a-p1", "a"],
            "workspace-wide caller sees every project of its workspace"
        );
        assert_eq!(
            bodies(a_p2.list("raw_item", 10).await.unwrap()),
            ["a"],
            "another project's object is hidden"
        );
        assert_eq!(
            b.get::<Item>("raw_item", "x").await.unwrap(),
            Some(item("b"))
        );
        assert!(b.get::<Item>("raw_item", "y").await.unwrap().is_none());
        assert!(
            !b.delete("raw_item", "y").await.unwrap(),
            "cannot delete across tenants"
        );
        assert_eq!(
            a.get::<Item>("raw_item", "x").await.unwrap(),
            Some(item("a"))
        );
        assert!(
            a.get::<Item>("raw_item", "z").await.unwrap().is_none(),
            "unscoped (legacy) rows are unreachable with a workspace"
        );
        std::fs::remove_file(&path).ok();
    }

    /// daruma 01a1441c: the README upgrade step for 0003, run verbatim on a
    /// database that was really written before 0003. Without it legacy rows
    /// are unreachable; with it they belong to the workspace and a row the
    /// tenant wrote after the upgrade is not overwritten.
    #[tokio::test]
    async fn readme_backfill_hands_pre_0003_rows_to_their_workspace() {
        const ADOPT: &str =
            "UPDATE OR IGNORE objects SET workspace = '<workspace-uuid>' WHERE workspace = '';";
        const LEFT: &str = "SELECT kind, id FROM objects WHERE workspace = '';";
        let readme = include_str!("../README.md");
        assert!(
            readme.contains(ADOPT) && readme.contains(LEFT),
            "README must carry exactly the backfill this test runs"
        );

        // A database at 0002, built from the shipped migration files.
        let path = temp_db();
        let old = std::env::temp_dir().join(format!("layer-kit-0002-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir(&old).unwrap();
        for f in ["0001_initial.sql", "0002_drop_events.sql"] {
            std::fs::copy(
                format!("{}/migrations/{f}", env!("CARGO_MANIFEST_DIR")),
                old.join(f),
            )
            .unwrap();
        }
        let pool = SqlitePool::connect_with(
            SqliteConnectOptions::new()
                .filename(&path)
                .create_if_missing(true),
        )
        .await
        .unwrap();
        sqlx::migrate::Migrator::new(old.as_path())
            .await
            .unwrap()
            .run(&pool)
            .await
            .unwrap();
        for (id, body) in [("a", "legacy-a"), ("b", "legacy-b")] {
            let payload = serde_json::to_string(&Item {
                id: id.into(),
                body: body.into(),
            })
            .unwrap();
            sqlx::query(
                "INSERT INTO objects (kind, id, payload, created_at, updated_at)
                 VALUES ('raw_item', ?1, ?2, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            )
            .bind(id)
            .bind(payload)
            .execute(&pool)
            .await
            .unwrap();
        }
        pool.close().await;
        std::fs::remove_dir_all(&old).ok();

        // The restart applies 0003: the rows survive but no tenant sees them.
        let store = Store::open(&path).await.unwrap();
        let ws = "019e5f3f-ab10-7451-8455-6f3807545eb9";
        let tenant = store.scoped(ws, None);
        let body = |i: Option<Item>| i.map(|i| i.body);
        assert_eq!(body(tenant.get("raw_item", "a").await.unwrap()), None);
        assert_eq!(
            body(store.get("raw_item", "a").await.unwrap()).as_deref(),
            Some("legacy-a"),
            "0003 keeps the row in the unscoped namespace"
        );
        // Written by the tenant after the upgrade, before the backfill ran.
        tenant
            .put(
                "raw_item",
                "b",
                &Item {
                    id: "b".into(),
                    body: "tenant-b".into(),
                },
            )
            .await
            .unwrap();

        sqlx::query(&ADOPT.replace("<workspace-uuid>", ws))
            .execute(store.pool())
            .await
            .unwrap();
        let left: Vec<(String, String)> =
            sqlx::query_as(LEFT).fetch_all(store.pool()).await.unwrap();

        assert_eq!(
            body(tenant.get("raw_item", "a").await.unwrap()).as_deref(),
            Some("legacy-a")
        );
        assert_eq!(
            body(
                store
                    .scoped(ws, Some("p1"))
                    .get("raw_item", "a")
                    .await
                    .unwrap()
            )
            .as_deref(),
            Some("legacy-a"),
            "an adopted row is workspace-wide"
        );
        assert_eq!(
            body(tenant.get("raw_item", "b").await.unwrap()).as_deref(),
            Some("tenant-b"),
            "the backfill never overwrites the tenant's own row"
        );
        assert_eq!(left, [("raw_item".to_owned(), "b".to_owned())]);
        assert!(
            store
                .scoped("other-ws", None)
                .get::<Item>("raw_item", "a")
                .await
                .unwrap()
                .is_none(),
            "only the named workspace adopts the rows"
        );
        std::fs::remove_file(&path).ok();
    }
}
