//! SQLite persistence shared by every MeiSei layer server: a `kind`-keyed
//! object table plus the append-only event log of the writes behind it.
//!
//! Deliberately domain-blind — a layer stores its own typed object as JSON
//! under its own `kind` (`"raw_item"`, `"decision"`, ...) and deserializes on
//! read. That keeps one schema (and one set of migrations) serving all five
//! layers; a layer that outgrows it can still open [`Store::pool`] and run
//! its own SQL.
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
}

/// One row of the append-only write log.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StoredEvent {
    pub seq: i64,
    pub kind: String,
    pub object_id: String,
    pub payload: serde_json::Value,
    pub occurred_at: String,
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
        Ok(Self { pool })
    }

    /// Open the path in `{TOOL}_DB`, defaulting to `./{tool}.db`.
    pub async fn from_env(tool: &str) -> Result<Self, sqlx::Error> {
        let path = std::env::var(format!("{}_DB", tool.to_uppercase()))
            .unwrap_or_else(|_| format!("{tool}.db"));
        Self::open(&path).await
    }

    /// Escape hatch for a layer whose queries outgrow this module.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Upsert an object and append the matching event, atomically. Returns the
    /// event's `seq`. `created_at` survives an overwrite; `updated_at` moves.
    pub async fn put<T: Serialize>(
        &self,
        kind: &str,
        id: &str,
        object: &T,
    ) -> Result<i64, sqlx::Error> {
        let payload = serde_json::to_string(object).map_err(encode_err)?;
        let now = crate::time::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "INSERT INTO objects (kind, id, payload, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT (kind, id) DO UPDATE SET payload = ?3, updated_at = ?4",
        )
        .bind(kind)
        .bind(id)
        .bind(&payload)
        .bind(&now)
        .execute(&mut *tx)
        .await?;
        let seq = sqlx::query(
            "INSERT INTO events (kind, object_id, payload, occurred_at)
             VALUES (?1, ?2, ?3, ?4) RETURNING seq",
        )
        .bind(kind)
        .bind(id)
        .bind(&payload)
        .bind(&now)
        .fetch_one(&mut *tx)
        .await?
        .get::<i64, _>("seq");
        tx.commit().await?;
        Ok(seq)
    }

    /// Fetch one object, `None` when absent.
    pub async fn get<T: DeserializeOwned>(
        &self,
        kind: &str,
        id: &str,
    ) -> Result<Option<T>, sqlx::Error> {
        let row = sqlx::query("SELECT payload FROM objects WHERE kind = ?1 AND id = ?2")
            .bind(kind)
            .bind(id)
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
        sqlx::query(
            "SELECT payload FROM objects WHERE kind = ?1 ORDER BY updated_at DESC, id DESC LIMIT ?2",
        )
        .bind(kind)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|r| serde_json::from_str(r.get("payload")).map_err(decode_err))
        .collect()
    }

    /// Delete an object. The events that produced it are kept — the log is
    /// append-only. `true` when a row was removed.
    pub async fn delete(&self, kind: &str, id: &str) -> Result<bool, sqlx::Error> {
        let done = sqlx::query("DELETE FROM objects WHERE kind = ?1 AND id = ?2")
            .bind(kind)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    /// Events with `seq > after`, oldest first — the cursor a follower (a
    /// projection, the platform) polls with.
    pub async fn events_since(
        &self,
        after: i64,
        limit: i64,
    ) -> Result<Vec<StoredEvent>, sqlx::Error> {
        sqlx::query(
            "SELECT seq, kind, object_id, payload, occurred_at FROM events
             WHERE seq > ?1 ORDER BY seq LIMIT ?2",
        )
        .bind(after)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|r| {
            Ok(StoredEvent {
                seq: r.get("seq"),
                kind: r.get("kind"),
                object_id: r.get("object_id"),
                payload: serde_json::from_str(r.get("payload")).map_err(decode_err)?,
                occurred_at: r.get("occurred_at"),
            })
        })
        .collect()
    }
}

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
    async fn objects_and_events_survive_reopen() {
        let path = temp_db();
        let first = Store::open(&path).await.unwrap();
        let item = Item {
            id: "raw_1".into(),
            body: "hello".into(),
        };
        let seq = first.put("raw_item", &item.id, &item).await.unwrap();
        assert_eq!(seq, 1, "event log starts at 1");
        drop(first);

        // Reopen the same file — migrations are idempotent, data is still there.
        let reopened = Store::open(&path).await.unwrap();
        assert_eq!(
            reopened.get::<Item>("raw_item", "raw_1").await.unwrap(),
            Some(item)
        );
        let events = reopened.events_since(0, 10).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["body"], "hello");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn upsert_lists_by_kind_and_deletes() {
        let path = temp_db();
        let store = Store::open(&path).await.unwrap();
        store
            .put("raw_item", "a", &Item { id: "a".into(), body: "one".into() })
            .await
            .unwrap();
        store
            .put("decision", "d", &Item { id: "d".into(), body: "other kind".into() })
            .await
            .unwrap();
        // Overwrite: one object, but a second event for it.
        store
            .put("raw_item", "a", &Item { id: "a".into(), body: "two".into() })
            .await
            .unwrap();

        let raw: Vec<Item> = store.list("raw_item", 10).await.unwrap();
        assert_eq!(raw.len(), 1, "upsert must not duplicate the object");
        assert_eq!(raw[0].body, "two");
        assert_eq!(store.events_since(0, 10).await.unwrap().len(), 3);
        assert_eq!(store.events_since(2, 10).await.unwrap()[0].seq, 3);

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
}
