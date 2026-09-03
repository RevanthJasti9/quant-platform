-- Core V1 tables. New scalar columns get added on the fly by
-- src/data/db.py's upsert_wide() helper (ALTER TABLE ... ADD COLUMN IF NOT
-- EXISTS) so new features/fundamentals fields never require a migration.
-- New *sources* (a whole new table) still get their DDL added here.

CREATE TABLE IF NOT EXISTS prices (
    ticker      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE,
    volume      BIGINT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker      VARCHAR NOT NULL,
    as_of       DATE    NOT NULL,
    sector      VARCHAR,
    industry    VARCHAR,
    PRIMARY KEY (ticker, as_of)
);

CREATE TABLE IF NOT EXISTS news_events (
    ticker          VARCHAR NOT NULL,
    url             VARCHAR NOT NULL,
    published_at    TIMESTAMP,
    headline        VARCHAR,
    source          VARCHAR,
    provider        VARCHAR,
    received_at     TIMESTAMP,
    event_type      VARCHAR,
    reliability_score DOUBLE,
    duplicate_group VARCHAR,
    PRIMARY KEY (ticker, url)
);

-- Existing local databases need an explicit upgrade because CREATE TABLE IF
-- NOT EXISTS does not add new columns to a table that is already present.
ALTER TABLE news_events ADD COLUMN IF NOT EXISTS provider VARCHAR;
ALTER TABLE news_events ADD COLUMN IF NOT EXISTS received_at TIMESTAMP;
ALTER TABLE news_events ADD COLUMN IF NOT EXISTS event_type VARCHAR;
ALTER TABLE news_events ADD COLUMN IF NOT EXISTS reliability_score DOUBLE;
ALTER TABLE news_events ADD COLUMN IF NOT EXISTS duplicate_group VARCHAR;

CREATE TABLE IF NOT EXISTS sec_filings (
    accession_number    VARCHAR NOT NULL,
    ticker              VARCHAR,
    cik                 VARCHAR,
    filing_type         VARCHAR,
    filing_date         DATE,
    url                 VARCHAR,
    PRIMARY KEY (accession_number)
);

CREATE TABLE IF NOT EXISTS insider_transactions (
    ticker              VARCHAR NOT NULL,
    insider_name        VARCHAR NOT NULL,
    transaction_date    DATE NOT NULL,
    transaction_code    VARCHAR NOT NULL,
    shares              DOUBLE NOT NULL,
    role                VARCHAR,
    price               DOUBLE,
    value               DOUBLE,
    shares_owned_after  DOUBLE,
    filing_url          VARCHAR,
    -- When the Form 4 actually became public (always >= transaction_date --
    -- insiders get up to 2 business days to file). Point-in-time features
    -- must key off this, not transaction_date, or they'd occasionally use
    -- information before it was actually known.
    filing_date         DATE,
    PRIMARY KEY (ticker, insider_name, transaction_date, transaction_code, shares)
);
ALTER TABLE insider_transactions ADD COLUMN IF NOT EXISTS filing_date DATE;

CREATE TABLE IF NOT EXISTS features (
    ticker      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version   VARCHAR NOT NULL,
    trained_at      TIMESTAMP NOT NULL,
    algo            VARCHAR,
    horizon_days    INTEGER,
    params_json     VARCHAR,
    train_start     DATE,
    train_end       DATE,
    feature_cols    VARCHAR,
    PRIMARY KEY (model_version, horizon_days)
);

CREATE TABLE IF NOT EXISTS predictions (
    ticker                  VARCHAR NOT NULL,
    prediction_date         DATE    NOT NULL,
    horizon_days            INTEGER NOT NULL,
    model_version           VARCHAR,
    expected_relative_return DOUBLE,
    probability_outperform  DOUBLE,
    confidence              DOUBLE,
    actual_relative_return  DOUBLE,
    error                   DOUBLE,
    evaluated_at            TIMESTAMP,
    -- Which ingest_runs row produced the data this prediction was built on,
    -- so any prediction can be traced back to exactly what was known and
    -- whether that run's sources were healthy. NULL for predictions made
    -- before this column existed, or via a path that doesn't have a run_id
    -- to hand (e.g. a manual one-off python -m src.models.predict).
    run_id                  VARCHAR,
    PRIMARY KEY (ticker, prediction_date, horizon_days)
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS run_id VARCHAR;

CREATE TABLE IF NOT EXISTS backtests (
    backtest_id     VARCHAR NOT NULL,
    run_at          TIMESTAMP,
    start_date      DATE,
    end_date        DATE,
    horizon_days    INTEGER,
    sharpe          DOUBLE,
    cagr            DOUBLE,
    max_drawdown    DOUBLE,
    win_rate        DOUBLE,
    turnover        DOUBLE,
    alpha           DOUBLE,
    volatility      DOUBLE,
    n_trades        INTEGER,
    params_json     VARCHAR,
    equity_curve_json VARCHAR,
    -- How many walk-forward folds actually contributed simulated days
    -- (some can be skipped if a fold's train/test window has no data), and
    -- a per-regime (bull/bear, high/low volatility) performance breakdown
    -- as JSON -- so a strong aggregate Sharpe/alpha can't hide a strategy
    -- that only worked in one regime or one lucky fold. See
    -- src/backtesting/metrics.py:classify_regimes/regime_breakdown_metrics.
    n_folds         INTEGER,
    regime_breakdown_json VARCHAR,
    -- calibration = information coefficient (correlation between predicted
    -- score and realized return); quantile_spread = top-quantile-minus-
    -- bottom-quantile mean return, a more direct read of ranking skill
    -- than the correlation alone. Both were computed before but silently
    -- discarded -- now persisted. See src/backtesting/metrics.py.
    calibration     DOUBLE,
    quantile_spread DOUBLE,
    -- Two-tailed p-value for "this IC could just be noise" (H0: true IC = 0).
    -- See src/stats.py's ic_significance.
    calibration_pvalue DOUBLE,
    PRIMARY KEY (backtest_id)
);
ALTER TABLE backtests ADD COLUMN IF NOT EXISTS n_folds INTEGER;
ALTER TABLE backtests ADD COLUMN IF NOT EXISTS regime_breakdown_json VARCHAR;
ALTER TABLE backtests ADD COLUMN IF NOT EXISTS calibration DOUBLE;
ALTER TABLE backtests ADD COLUMN IF NOT EXISTS quantile_spread DOUBLE;
ALTER TABLE backtests ADD COLUMN IF NOT EXISTS calibration_pvalue DOUBLE;

CREATE TABLE IF NOT EXISTS holdings (
    ticker      VARCHAR NOT NULL,
    shares      DOUBLE NOT NULL,
    cost_basis  DOUBLE NOT NULL,   -- average price paid per share
    added_at    TIMESTAMP,
    PRIMARY KEY (ticker)
);

-- Read-only broker values are kept separately from research prices so the
-- portfolio screen can match the broker's account total exactly.
ALTER TABLE holdings ADD COLUMN IF NOT EXISTS broker_market_price DOUBLE;
ALTER TABLE holdings ADD COLUMN IF NOT EXISTS broker_previous_close DOUBLE;
ALTER TABLE holdings ADD COLUMN IF NOT EXISTS broker_price_at TIMESTAMP;
ALTER TABLE holdings ADD COLUMN IF NOT EXISTS market_price_source VARCHAR;
-- Marks whether a row is maintained manually or by a complete read-only
-- broker snapshot. This lets a broker sync remove sold positions without
-- touching manually tracked positions.
ALTER TABLE holdings ADD COLUMN IF NOT EXISTS position_source VARCHAR;

CREATE TABLE IF NOT EXISTS broker_portfolio_snapshots (
    provider        VARCHAR PRIMARY KEY,
    synced_at       TIMESTAMP NOT NULL,
    total_value     DOUBLE NOT NULL,
    equity_value    DOUBLE NOT NULL,
    cash            DOUBLE NOT NULL,
    crypto_value    DOUBLE NOT NULL
);

-- LLM-generated (optional -- see src/llm/) plain-English summary of each
-- ticker's recent headlines, plus an LLM-derived sentiment score from the
-- same call. Refreshed once per ingest run, one row per ticker (latest
-- snapshot only, for display) -- a missing row just means no LLM was
-- available or there was no recent news to summarize. For a trainable
-- history of sentiment over time, see news_sentiment below.
CREATE TABLE IF NOT EXISTS news_digests (
    ticker          VARCHAR NOT NULL,
    generated_at    TIMESTAMP,
    summary         VARCHAR,
    headline_count  INTEGER,
    PRIMARY KEY (ticker)
);
ALTER TABLE news_digests ADD COLUMN IF NOT EXISTS sentiment_score DOUBLE;

-- One row per (ticker, date) so sentiment has an actual history to train
-- on, unlike news_digests above which only ever keeps the latest snapshot.
-- sentiment_score is LLM-derived (-1.0 very negative to 1.0 very positive,
-- 0 neutral/routine) -- unlike every other feature in this app, it's not a
-- deterministic, auditable calculation, just the model's best read of the
-- same headlines a person would see. Treat it as a soft signal: useful if
-- the trained model's feature importance actually bears that out (checked,
-- not assumed), not a validated NLP classifier against labeled data.
CREATE TABLE IF NOT EXISTS news_sentiment (
    ticker          VARCHAR NOT NULL,
    as_of           DATE NOT NULL,
    sentiment_score DOUBLE,
    headline_count  INTEGER,
    generated_at    TIMESTAMP,
    PRIMARY KEY (ticker, as_of)
);

-- Data control plane: one row per run_ingest() call, plus one row per
-- source attempted within that call. This is what lets the app say "this
-- source failed" or "this data is stale" instead of silently serving a
-- confident-looking number built on missing/broken data.
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id              VARCHAR NOT NULL,
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    status              VARCHAR,   -- 'success' | 'partial' | 'failed'
    sources_requested   INTEGER,
    sources_succeeded   INTEGER,
    sources_failed      INTEGER,
    critical_failure    BOOLEAN,
    PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS source_runs (
    run_id      VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,
    status      VARCHAR,   -- 'success' | 'failed'
    row_count   INTEGER,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    error       VARCHAR,
    PRIMARY KEY (run_id, source)
);

-- Checks against the *content* of the data (as opposed to source_runs,
-- which only knows whether a fetch succeeded) -- stale prices, missing
-- benchmark/holdings coverage, abnormal price jumps, features that came out
-- unexpectedly empty. See src/data/quality.py.
CREATE TABLE IF NOT EXISTS data_quality_results (
    run_id      VARCHAR NOT NULL,
    check_name  VARCHAR NOT NULL,
    status      VARCHAR,   -- 'pass' | 'warn' | 'fail'
    detail      VARCHAR,
    checked_at  TIMESTAMP,
    PRIMARY KEY (run_id, check_name)
);
