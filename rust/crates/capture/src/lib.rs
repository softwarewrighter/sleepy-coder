//! Episode capture and SQLite storage for sleepy-coder.
//!
//! This crate provides persistent storage for episodes,
//! supporting insert, query, and export operations.

use core_types::{Episode, EvalResult, SCHEMA_VERSION};
use rusqlite::{Connection, Result as SqlResult, params};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

/// Errors from the capture crate.
#[derive(Error, Debug)]
pub enum CaptureError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for capture operations.
pub type Result<T> = std::result::Result<T, CaptureError>;

/// SQLite-backed storage for episodes.
pub struct EpisodeStore {
    conn: Connection,
}

impl EpisodeStore {
    /// Open or create a database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                run_id TEXT,
                task_id TEXT NOT NULL,
                attempt_idx INTEGER NOT NULL,
                prompt_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                error_signature TEXT,
                diff_unified TEXT,
                passed INTEGER NOT NULL,
                steps_to_green INTEGER NOT NULL,
                wall_clock_ms INTEGER NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_error ON episodes(error_signature);

            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                cycle INTEGER NOT NULL,
                repeat_error_rate REAL NOT NULL,
                median_steps_to_green REAL NOT NULL,
                pass_rate REAL NOT NULL,
                frozen_pass_rate REAL NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_eval_run ON eval_results(run_id);
            "#,
        )?;

        // Store schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        Ok(())
    }

    /// Get the current schema version.
    pub fn schema_version(&self) -> Result<u32> {
        let version: String = self.conn.query_row(
            "SELECT value FROM schema_info WHERE key = 'version'",
            [],
            |row| row.get(0),
        )?;
        Ok(version.parse().unwrap_or(0))
    }

    /// Insert an episode without a run ID.
    pub fn insert(&self, episode: &Episode) -> Result<i64> {
        self.insert_with_run("", episode)
    }

    /// Insert an episode with a run ID.
    pub fn insert_with_run(&self, run_id: &str, episode: &Episode) -> Result<i64> {
        self.conn.execute(
            r#"
            INSERT INTO episodes (
                run_id, task_id, attempt_idx, prompt_hash, model_id,
                error_signature, diff_unified, passed, steps_to_green,
                wall_clock_ms, tokens_in, tokens_out, timestamp
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            "#,
            params![
                run_id,
                episode.task_id,
                episode.attempt_idx,
                episode.prompt_hash,
                episode.model_id,
                episode.error_signature,
                episode.diff_unified,
                episode.passed as i32,
                episode.steps_to_green,
                episode.wall_clock_ms,
                episode.tokens_in,
                episode.tokens_out,
                episode.timestamp.to_rfc3339(),
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Get an episode by ID.
    pub fn get(&self, id: i64) -> Result<Option<Episode>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT task_id, attempt_idx, prompt_hash, model_id,
                   error_signature, diff_unified, passed, steps_to_green,
                   wall_clock_ms, tokens_in, tokens_out, timestamp
            FROM episodes WHERE id = ?1
            "#,
        )?;

        let result = stmt.query_row(params![id], |row| self.row_to_episode(row));

        match result {
            Ok(episode) => Ok(Some(episode)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Query episodes by run ID.
    pub fn query_by_run(&self, run_id: &str) -> Result<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT task_id, attempt_idx, prompt_hash, model_id,
                   error_signature, diff_unified, passed, steps_to_green,
                   wall_clock_ms, tokens_in, tokens_out, timestamp
            FROM episodes WHERE run_id = ?1
            "#,
        )?;

        let episodes = stmt
            .query_map(params![run_id], |row| self.row_to_episode(row))?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(episodes)
    }

    /// Query episodes by task ID.
    pub fn query_by_task(&self, task_id: &str) -> Result<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT task_id, attempt_idx, prompt_hash, model_id,
                   error_signature, diff_unified, passed, steps_to_green,
                   wall_clock_ms, tokens_in, tokens_out, timestamp
            FROM episodes WHERE task_id = ?1
            "#,
        )?;

        let episodes = stmt
            .query_map(params![task_id], |row| self.row_to_episode(row))?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(episodes)
    }

    /// Query episodes where a task failed then was fixed.
    pub fn query_failed_then_fixed(&self) -> Result<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT task_id, attempt_idx, prompt_hash, model_id,
                   error_signature, diff_unified, passed, steps_to_green,
                   wall_clock_ms, tokens_in, tokens_out, timestamp
            FROM episodes e1
            WHERE passed = 1
              AND EXISTS (
                  SELECT 1 FROM episodes e2
                  WHERE e2.task_id = e1.task_id
                    AND e2.passed = 0
                    AND e2.attempt_idx < e1.attempt_idx
              )
            "#,
        )?;

        let episodes = stmt
            .query_map([], |row| self.row_to_episode(row))?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(episodes)
    }

    /// Count episodes by error signature.
    pub fn count_by_error_signature(&self) -> Result<HashMap<String, i32>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT error_signature, COUNT(*) as cnt
            FROM episodes
            WHERE error_signature IS NOT NULL
            GROUP BY error_signature
            "#,
        )?;

        let mut counts = HashMap::new();
        let rows = stmt.query_map([], |row| {
            let sig: String = row.get(0)?;
            let cnt: i32 = row.get(1)?;
            Ok((sig, cnt))
        })?;

        for row in rows {
            let (sig, cnt) = row?;
            counts.insert(sig, cnt);
        }

        Ok(counts)
    }

    /// Export all episodes to a JSONL file.
    pub fn export_jsonl(&self, path: &Path) -> Result<usize> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT task_id, attempt_idx, prompt_hash, model_id,
                   error_signature, diff_unified, passed, steps_to_green,
                   wall_clock_ms, tokens_in, tokens_out, timestamp
            FROM episodes
            "#,
        )?;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut count = 0;

        let rows = stmt.query_map([], |row| self.row_to_episode(row))?;

        for row in rows {
            let episode = row?;
            let json = serde_json::to_string(&episode)?;
            writeln!(writer, "{json}")?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Insert an evaluation result.
    pub fn insert_eval_result(&self, run_id: &str, cycle: u32, result: &EvalResult) -> Result<i64> {
        self.conn.execute(
            r#"
            INSERT INTO eval_results (
                run_id, cycle, repeat_error_rate, median_steps_to_green,
                pass_rate, frozen_pass_rate, timestamp
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                run_id,
                cycle,
                result.repeat_error_rate,
                result.median_steps_to_green,
                result.pass_rate,
                result.frozen_pass_rate,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Query evaluation results by run ID.
    pub fn query_eval_results(&self, run_id: &str) -> Result<Vec<EvalResult>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT cycle, repeat_error_rate, median_steps_to_green,
                   pass_rate, frozen_pass_rate
            FROM eval_results WHERE run_id = ?1
            ORDER BY cycle
            "#,
        )?;

        let results = stmt
            .query_map(params![run_id], |row| {
                Ok(EvalResult {
                    cycle: row.get(0)?,
                    repeat_error_rate: row.get(1)?,
                    median_steps_to_green: row.get(2)?,
                    pass_rate: row.get(3)?,
                    frozen_pass_rate: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(results)
    }

    /// Convert a database row to an Episode.
    fn row_to_episode(&self, row: &rusqlite::Row) -> SqlResult<Episode> {
        let timestamp_str: String = row.get(11)?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        Ok(Episode {
            task_id: row.get(0)?,
            attempt_idx: row.get(1)?,
            prompt_hash: row.get(2)?,
            model_id: row.get(3)?,
            error_signature: row.get(4)?,
            diff_unified: row.get(5)?,
            passed: row.get::<_, i32>(6)? != 0,
            steps_to_green: row.get(7)?,
            wall_clock_ms: row.get(8)?,
            tokens_in: row.get(9)?,
            tokens_out: row.get(10)?,
            timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{Episode, EvalResult};
    use tempfile::tempdir;

    #[test]
    fn test_store_creation() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");

        let store = EpisodeStore::open(&db_path).unwrap();

        assert!(db_path.exists());
        drop(store);
    }

    #[test]
    fn test_insert_and_retrieve_episode() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        let mut episode = Episode::new("task_001".to_string(), 0, "model_v1".to_string());
        episode.set_error("E0382:borrow_of_moved_value".to_string());

        let id = store.insert(&episode).unwrap();
        assert!(id > 0);

        let retrieved = store.get(id).unwrap().unwrap();
        assert_eq!(retrieved.task_id, "task_001");
        assert_eq!(
            retrieved.error_signature,
            Some("E0382:borrow_of_moved_value".to_string())
        );
    }

    #[test]
    fn test_query_by_run_id() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        // Insert episodes for different runs
        let ep1 = Episode::new("task_001".to_string(), 0, "model".to_string());
        let ep2 = Episode::new("task_002".to_string(), 0, "model".to_string());
        let ep3 = Episode::new("task_001".to_string(), 0, "model".to_string());

        store.insert_with_run("run_001", &ep1).unwrap();
        store.insert_with_run("run_001", &ep2).unwrap();
        store.insert_with_run("run_002", &ep3).unwrap();

        let run1_episodes = store.query_by_run("run_001").unwrap();
        assert_eq!(run1_episodes.len(), 2);

        let run2_episodes = store.query_by_run("run_002").unwrap();
        assert_eq!(run2_episodes.len(), 1);
    }

    #[test]
    fn test_query_by_task() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        let ep1 = Episode::new("task_001".to_string(), 0, "model".to_string());
        let ep2 = Episode::new("task_001".to_string(), 1, "model".to_string());
        let ep3 = Episode::new("task_002".to_string(), 0, "model".to_string());

        store.insert(&ep1).unwrap();
        store.insert(&ep2).unwrap();
        store.insert(&ep3).unwrap();

        let task1_episodes = store.query_by_task("task_001").unwrap();
        assert_eq!(task1_episodes.len(), 2);
    }

    #[test]
    fn test_query_failed_then_fixed() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        // Episode that failed then was fixed
        let mut ep1_fail = Episode::new("task_001".to_string(), 0, "model".to_string());
        ep1_fail.set_error("E0382:error".to_string());

        let mut ep1_pass = Episode::new("task_001".to_string(), 1, "model".to_string());
        ep1_pass.mark_passed(2);
        ep1_pass.diff_unified = Some("--- a\n+++ b".to_string());

        // Episode that never passed
        let mut ep2_fail = Episode::new("task_002".to_string(), 0, "model".to_string());
        ep2_fail.set_error("E0382:error".to_string());

        store.insert(&ep1_fail).unwrap();
        store.insert(&ep1_pass).unwrap();
        store.insert(&ep2_fail).unwrap();

        let fixed = store.query_failed_then_fixed().unwrap();
        assert_eq!(fixed.len(), 1);
        assert_eq!(fixed[0].task_id, "task_001");
        assert!(fixed[0].passed);
    }

    #[test]
    fn test_export_to_jsonl() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        let ep1 = Episode::new("task_001".to_string(), 0, "model".to_string());
        let ep2 = Episode::new("task_002".to_string(), 0, "model".to_string());

        store.insert(&ep1).unwrap();
        store.insert(&ep2).unwrap();

        let export_path = temp.path().join("export.jsonl");
        let count = store.export_jsonl(&export_path).unwrap();
        assert_eq!(count, 2);

        let content = std::fs::read_to_string(&export_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line should be valid JSON
        for line in lines {
            let _: Episode = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn test_count_by_error_signature() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        let mut ep1 = Episode::new("task_001".to_string(), 0, "model".to_string());
        ep1.set_error("E0382:borrow".to_string());

        let mut ep2 = Episode::new("task_002".to_string(), 0, "model".to_string());
        ep2.set_error("E0382:borrow".to_string());

        let mut ep3 = Episode::new("task_003".to_string(), 0, "model".to_string());
        ep3.set_error("E0277:trait".to_string());

        store.insert(&ep1).unwrap();
        store.insert(&ep2).unwrap();
        store.insert(&ep3).unwrap();

        let counts = store.count_by_error_signature().unwrap();
        assert_eq!(counts.get("E0382:borrow"), Some(&2));
        assert_eq!(counts.get("E0277:trait"), Some(&1));
    }

    #[test]
    fn test_store_eval_result() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");
        let store = EpisodeStore::open(&db_path).unwrap();

        let result = EvalResult::new(0.25, 3.5, 0.85, 0.90);
        store.insert_eval_result("run_001", 0, &result).unwrap();

        let results = store.query_eval_results("run_001").unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].repeat_error_rate - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_schema_version_tracking() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("test.db");

        let store = EpisodeStore::open(&db_path).unwrap();
        let version = store.schema_version().unwrap();

        assert_eq!(version, core_types::SCHEMA_VERSION);
    }
}
