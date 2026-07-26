-- Curated database of rare / high-value Euro issues.
--
-- One row per issue (country + denomination + year + variant). Reference images
-- live in a separate table so an issue can have several, and so an embedding
-- backend has somewhere to cache vectors.

CREATE TABLE IF NOT EXISTS rare_coin (
    id              INTEGER PRIMARY KEY,
    country         TEXT    NOT NULL,
    denomination    INTEGER NOT NULL,      -- cents: 1, 2, 5, 10, 20, 50, 100, 200
    year            INTEGER,               -- NULL when the issue spans years
    variant         TEXT    NOT NULL DEFAULT 'standard',
                                           -- 'standard' | 'commemorative' | 'error'
    name            TEXT    NOT NULL,      -- e.g. "Monaco 2007 Grace Kelly"
    mintage         INTEGER,               -- NULL when unknown
    est_value_eur   REAL,                  -- rough catalogue value, for triage only
    notes           TEXT    NOT NULL DEFAULT '',
    source_url      TEXT,                  -- where the figures came from
    UNIQUE (country, denomination, year, variant, name)
);

CREATE INDEX IF NOT EXISTS idx_rare_coin_denomination
    ON rare_coin (denomination);
CREATE INDEX IF NOT EXISTS idx_rare_coin_country_year
    ON rare_coin (country, year);

CREATE TABLE IF NOT EXISTS reference_image (
    id              INTEGER PRIMARY KEY,
    rare_coin_id    INTEGER NOT NULL REFERENCES rare_coin (id) ON DELETE CASCADE,
    path            TEXT    NOT NULL UNIQUE,  -- relative to the image store root
    face            TEXT    NOT NULL DEFAULT 'reverse',  -- 'obverse' | 'reverse'
    embedding       BLOB,                     -- float32 vector, populated by the
                                              -- embedding backend when used
    embedding_model TEXT                      -- which model produced it
);

CREATE INDEX IF NOT EXISTS idx_reference_image_coin
    ON reference_image (rare_coin_id);

-- One row per flagged coin, so scan history is queryable over time.
CREATE TABLE IF NOT EXISTS scan_flag (
    id              INTEGER PRIMARY KEY,
    scanned_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    source_image    TEXT    NOT NULL,
    coin_index      INTEGER NOT NULL,
    crop_path       TEXT,
    denomination    INTEGER,
    rare_coin_id    INTEGER REFERENCES rare_coin (id),
    score           REAL    NOT NULL DEFAULT 0.0,
    notes           TEXT    NOT NULL DEFAULT '',
    reviewed        INTEGER NOT NULL DEFAULT 0  -- 0 = pending, 1 = checked by hand
);

CREATE INDEX IF NOT EXISTS idx_scan_flag_pending
    ON scan_flag (reviewed, scanned_at);
