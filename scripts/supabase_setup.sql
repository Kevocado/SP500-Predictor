-- ============================================================
-- SP500 Predictor — Supabase Schema
-- Run this in the Supabase SQL Editor (Dashboard → SQL)
-- ============================================================

-- Live scanner opportunities (replaces Azure LiveOpportunities)
CREATE TABLE IF NOT EXISTS live_opportunities (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    engine          TEXT NOT NULL,           -- Weather, Macro, Quant, etc.
    asset           TEXT,
    market_title    TEXT,
    market_ticker   TEXT,
    event_ticker    TEXT,
    action          TEXT,                    -- BUY YES / BUY NO
    model_prob      REAL,
    market_price    REAL,
    edge            REAL,
    confidence      REAL,
    reasoning       TEXT,
    data_source     TEXT,
    kalshi_url      TEXT,
    market_date     TEXT,
    expiration      TEXT,
    ai_approved     BOOLEAN DEFAULT TRUE,
    ai_reasoning    TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Paper trading signals from Quant engine (replaces Azure PaperTradingSignals)
CREATE TABLE IF NOT EXISTS paper_signals (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    ticker          TEXT NOT NULL,           -- SPY, QQQ
    predicted_price REAL,
    current_price   REAL,
    direction       TEXT,                    -- UP / DOWN
    model_prob      REAL,
    kelly_bet       REAL,
    edge            REAL,
    rmse            REAL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Historical trade log for backtesting & PnL tracking
CREATE TABLE IF NOT EXISTS trade_history (
    id              BIGSERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    predicted_price REAL,
    current_price   REAL,
    actual_price    REAL,
    model_rmse      REAL,
    best_edge       REAL,
    best_action     TEXT,
    best_strike     TEXT,
    brier_score     REAL,
    pnl_cents       REAL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Scanner run metadata (new — tracks timing + health)
CREATE TABLE IF NOT EXISTS scanner_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT UNIQUE NOT NULL,
    status          TEXT DEFAULT 'running',  -- running, completed, failed
    engines_run     TEXT[],
    total_opps      INTEGER DEFAULT 0,
    duration_sec    REAL,
    error_msg       TEXT,
    wipe_date       TIMESTAMPTZ,            -- records the hard reset date
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_live_opps_run ON live_opportunities(run_id);
CREATE INDEX IF NOT EXISTS idx_live_opps_engine ON live_opportunities(engine);
CREATE INDEX IF NOT EXISTS idx_paper_signals_ticker ON paper_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_trade_history_ticker ON trade_history(ticker);
CREATE INDEX IF NOT EXISTS idx_scanner_runs_status ON scanner_runs(status);
