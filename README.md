# AI Quant Platform — Cloud-First V1.5

A provider-independent research platform that preserves the existing V1 local
pipeline while adding a cloud-first control plane for recurring-free compute,
storage, and orchestration. The repo still runs locally for development and
tests, but the runtime contracts now separate the app from any one GPU, data,
or storage provider.

No live trading is included. This phase does not add Robinhood or execution
workflows. See `config/settings.yaml` for the editable universe, model params,
scheduling, and cloud-provider routing policy.

## Architecture

```text
Mac / Codex / Git
        |
        v
      GitHub
        |
        v
Cloud control plane / job manager
        |
        +--> metadata storage (Postgres-compatible)
        +--> object storage (S3-compatible / Parquet / model artifacts)
        +--> data router --> market/news/SEC providers
        +--> GPU router --> Modal / Kaggle / HF ZeroGPU
        |
        v
Specialist models
  - XGBoost / LightGBM / CatBoost
  - Chronos-Bolt / Chronos-2 / TimesFM 2.5
  - FinBERT
        |
        v
Ensemble / ranking / prediction journal / dashboard
```

The cloud-control modules live in `src/cloud/`:

- `contracts.py`: provider-neutral interfaces for data, GPU, storage, quotas, and model execution
- `control_plane.py`: job manager with priorities, deadlines, dependencies, retries, and stale publish states
- `routers.py`: free-compute router, data fallback router, and partial-refresh publication planner
- `providers.py`: pluggable recurring-free GPU provider stubs for Modal, Kaggle, and Hugging Face ZeroGPU
- `model_registry.py`: specialist model catalog for XGBoost, LightGBM, CatBoost, Chronos-2, TimesFM 2.5, Chronos-Bolt, and FinBERT
- `scheduler.py`: event-driven job graph for data capture, EOD features, inference, ranking, retraining, model arena, and journal refresh
- `scripts/cloudctl.py`: lightweight status, jobs, GPU-quota, refresh, predict, train, backtest, and deploy-planning commands

## Setup

```bash
cd quant-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC_EDGAR_USER_AGENT with "Your Name you@example.com"
```

## Run the current local pipeline once

Ingests prices/fundamentals/news/SEC filings for the universe in
`config/settings.yaml`, builds features, trains models, walk-forward
backtests them, and generates + evaluates predictions:

```bash
python scripts/run_pipeline.py
```

First run pulls several years of history for ~30 tickers — expect it to
take a few minutes, mostly spent on yfinance/EDGAR HTTP calls.

Use `--skip-backtest` for faster iteration once the pipeline has run once.

## Cloud control-plane commands

These commands do not consume provider credits by themselves; they show the
planned orchestration and recurring-free routing policy:

```bash
python scripts/cloudctl.py status
python scripts/cloudctl.py jobs
python scripts/cloudctl.py gpu-quota
python scripts/cloudctl.py predict
python scripts/cloudctl.py deploy
```

Deployment preparation, runtime settings, and post-deploy verification are in [DEPLOYMENT.md](DEPLOYMENT.md). The included GitHub Actions workflow tests every push and builds the deployable container; it does not deploy to a paid service or submit GPU work.

## Run the dashboard

```bash
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/dashboard, or double-click **"AI Quant
Dashboard.command"** on the Desktop — it starts the server if it isn't
already running and opens the dashboard in your browser.

`/stock/{TICKER}` shows a price chart + latest features + prediction
history for one name, with plain-language explanations throughout (this
app assumes no finance background); `/backtests` shows every backtest run
and the latest equity curve, in plain language too.

### My Holdings vs. Currently Outperforming

The dashboard splits into two distinct things, deliberately kept separate:

- **My Holdings** — stocks you actually own. Manual entries are supported,
  and a complete read-only Robinhood snapshot can also maintain positions,
  cost basis, and current quotes without triggering any research or trading
  workflow. A complete broker snapshot removes sold broker-managed positions;
  it never places orders. Adding a manual position averages into the existing
  ticker (`src/holdings.py`).
- **Currently Outperforming** — a live, unpersisted scan: the top stocks by
  modeled probability of beating the market over the configured horizon
  (`outperformers` in `config/settings.yaml`). Pure research, not tied to
  your holdings at all.

Holdings tickers are folded into every future `run_ingest()` call
automatically, same as the universe.

The app also starts a background scheduler (`src/scheduler/jobs.py`) that
re-runs ingest → features → quality checks → predict → evaluate → news
digests. Default cadence is 4:30pm ET, weekdays (after the US market
close), but that's a config value, not a code change — see `scheduler` in
`config/settings.yaml`. Retraining stays manual — run
`python -m src.models.train` (or the full pipeline) whenever you want a
fresh model version; `predict.py` always uses whichever model has the most
recent `trained_at` for each horizon.

## Why a prediction is what it is

Every prediction gets a real explanation, not a canned one. `src/models/explain.py`
runs SHAP (`TreeExplainer`) against the actual trained XGBoost/LightGBM
regressors for that specific ticker/date, and keeps the top features that
actually pushed the number up or down — stored as `reasons_json` on the
`predictions` row. That's shown as +/- chips on the stock page and in
Forecast History, and it works with zero extra setup.

If a local LLM is available (see below), `src/models/narrate.py` turns
those same SHAP reasons into one plain-English sentence (`reasons_summary`)
— the LLM only rephrases facts already computed, it never invents its own
reasoning or sees anything the SHAP output didn't already say.

## Optional: local LLM (for plain-English summaries)

Nothing in this app requires an LLM — predictions, backtests, and the SHAP
reasons chips all work without one. If [Ollama](https://ollama.com) is
installed and running, two things light up automatically:

- **Reasons, as a sentence** — `reasons_summary` on the stock page and in
  Forecast History (see above).
- **Recent news, summarized** — `src/models/news_digest.py` turns each
  ticker's recent headlines (from whichever of the three news sources are
  active) into a 1-2 sentence "what's happening" blurb, shown in a "Recent
  news" section on the stock page alongside the raw headlines/links.
- **News sentiment, as a feature** — the same LLM call that writes the
  summary also scores the same headlines from -1.0 (very negative) to +1.0
  (very positive), stored per (ticker, date) in `news_sentiment` and fed
  into the model as `news_sentiment_score`. This is a genuinely different
  kind of input than everything else in `features`: it's the LLM's own
  read of the headlines, not a deterministic, auditable calculation like
  the SHAP reasons or the rule-based event counts. It's calibrated by hand
  against a few known cases, not validated against a labeled dataset —
  treat it as a candidate signal. Check the trained model's
  `feature_importances_` for `news_sentiment_score` before trusting it;
  don't assume an LLM-derived number is useful just because it exists.

All three are generated during the pipeline run (`run_predictions()` /
`build_news_digests()`), not live on page load, so browsing the dashboard
never waits on the LLM. If Ollama isn't running, all three are silently
skipped (`src/llm/ollama_client.py` fails soft) and the app falls back to
its structured (chip/number) display.

Setup (already done on this machine): install Ollama, then

```bash
ollama pull llama3.2:3b
```

`llama3.2:3b` was picked specifically for its footprint on an 8GB Mac
(~2.5GB resident while generating; Ollama unloads it automatically after
~5 minutes idle — never sits in memory while you're just browsing). Swap
the model in `config/settings.yaml`'s `llm.model` if you want something
different; set `llm.enabled: false` to turn this off entirely.

### Freeing RAM

The app automatically releases the optional local LLM after every pipeline
run, including a run that stops early because of a data-quality problem. The
dashboard scheduler also checks every five minutes while no data/model task is
active and clears unused local-model memory. Delayed scheduler events are
coalesced, so sleep or a restart cannot create a pile of overlapping runs.

The dashboard stays online while this happens. It never force-stops a current
ingest, feature build, training run, or database write: interrupting those
halfway through is riskier than briefly retaining memory. The Live activity
panel reports any active pipeline task.

For an immediate manual cleanup of the optional local LLM, run:

```bash
./scripts/free_memory.sh
```

The automatic cleanup covers normal use. Avoid force-stopping an active
pipeline unless you specifically need to abandon that run; Ollama reloads the
model automatically the next time it is needed.

## Adding a new data source

Drop a new module in `src/data/`, subclass `DataSource` (see
`src/data/base.py`), implement `fetch()`, and decorate the class with
`@register_source("your_name")`. `src/data/__init__.py` auto-imports every
module in that folder, so nothing else needs to change — `run_ingest()`
(and the scheduler) picks it up automatically. If it needs a new table, add
its DDL to `src/data/schema.sql`; if it just adds scalar columns to an
existing table (a new fundamentals field, a new feature), the `upsert_wide`
helper in `src/data/db.py` adds those columns on the fly.

### News sources

Yahoo Finance remains available as a lower-confidence coverage feed. Add
`FINNHUB_API_KEY` and/or `POLYGON_API_KEY` to `.env` to enable the optional
company-news providers (Finnhub free tier: 60 requests/min; Polygon/Massive
free tier: 5 requests/min — both providers pace themselves to actually fit
under their own limit across the full universe rather than bursting and
getting the rest of the run silently rate-limited). Every article is stored
with its publisher, provider,
original publication time, receipt time, broad event type, reliability score,
and a duplicate group for syndicated versions of the same headline. SEC
filings and Form 4 insider transactions remain separately ingested as the
official-event record. To add an official company press-release feed, put its
RSS or Atom URL under `news.company_press_release_feeds` in
`config/settings.yaml`; those articles are marked `company_ir` with the
highest non-SEC reliability score.

## Data quality and lineage

Every `run_ingest()` call records an `ingest_runs` row (one per call) and a
`source_runs` row per source attempted — status, row count, error. If a
source listed under `data_quality.critical_sources` in `settings.yaml`
(`prices` by default) fails or comes back empty, the run is marked
`critical_failure` and the scheduler/pipeline stop before rebuilding
features or predictions on missing data, rather than silently refreshing on
top of a bad ingest.

Separately, `src/data/quality.py` checks the *content* of what landed —
stale prices, missing/stale benchmark or holdings data, tickers that never
got any data, single-day price moves that look like bad data rather than a
real move, and core features that came out unexpectedly empty — into
`data_quality_results`. These don't block predictions (a stale price for one
obscure holding shouldn't block the other 30 tickers) but are logged and
shown on the dashboard's collapsible "System health" section.

Every prediction also carries the `run_id` of the ingest run its underlying
data came from (`predictions.run_id`), so any forecast can be traced back to
exactly what was known, and whether that run's sources were healthy, when it
was made.

News and insider-trading data feed the model directly, not just the UI —
`src/features/events.py` turns `news_events`/`insider_transactions`/
`sec_filings` into point-in-time-safe rolling features (event counts,
reliability-weighted counts, insider buy/sell $ value, 8-K filing recency).
Insider features key off `filing_date` — when the Form 4 actually became
public — not `transaction_date`, since insiders get up to 2 business days to
file; using the trade date directly would occasionally let a feature see a
transaction before the market actually could have.

## What is implemented vs. stubbed

Implemented in code now:

- Provider contracts for GPU, financial data, storage, and model execution
- Central job definitions with priority, deadline, dependency, retry, and stale publish states
- Free GPU router with pluggable Modal, Kaggle, and HF ZeroGPU adapters
- Quota-aware queue behavior with no paid fallback
- Specialist model catalog spanning tree models, Chronos variants, TimesFM 2.5, and FinBERT
- Lightweight cloud CLI commands and a scheduler-visible cloud job plan

Still stubbed until external accounts or credentials are configured:

- Real Modal, Kaggle, and Hugging Face job submission
- Real FMP or other premium market-data adapters
- Real Postgres-compatible and S3-compatible storage backends
- Real GitHub-to-cloud deployment wiring

## Tests

```bash
pytest
```

Covers feature/target leakage guarantees (including the event and sentiment
features' point-in-time cutoffs), the walk-forward fold builder and its
purge-gap safety check, the dynamic-column upsert mechanism, ingest
run-tracking and data quality checks, prediction journal grading, the
accuracy page's grading math, the scheduler's config-driven cadence, the
Finnhub/Polygon rate-limit pacing, LLM sentiment-response parsing, and an
end-to-end feature-build integration test. Does not hit the network or a
real LLM or cloud GPU — those are exercised by actually running
`scripts/run_pipeline.py` with Ollama running.

## What's not here yet (by design)

News/event sentiment classification, the book-strategy library
(Minervini/CANSLIM/Weinstein/etc.), the experiment/model-comparison
registry, market regime detection, and Portfolio Guardian are V2.
Robinhood execution, the risk engine, and the trade-proposal/review
workflow are V3. ChatGPT-as-research-reviewer hooks are V4.
