"""One-shot CLI: ingest -> features -> train -> backtest -> predict -> evaluate.

This is what the scheduler's after-close job effectively does too (minus
training/backtesting, which stay manual in V1 — see src/scheduler/jobs.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting.engine import run_backtest
from src.config import get_settings
from src.data.ingest import run_ingest
from src.data.quality import has_blocking_failure, run_data_quality_checks
from src.features.build import build_features
from src.journal.evaluate import evaluate_predictions
from src.llm.client import unload as unload_llm
from src.models.news_digest import build_news_digests
from src.models.predict import run_predictions
from src.models.train import train_models
from src.observability.runtime import track_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backtest", action="store_true", help="Skip the walk-forward backtest (faster iteration).")
    args = parser.parse_args()

    try:
        with track_task("full research pipeline"):
            logger.info("Step 1/8: ingesting data (prices, fundamentals, news, SEC filings/insiders)")
            ingest_result = run_ingest()
            logger.info("Ingested rows: %s", ingest_result.results)
            if ingest_result.critical_failure:
                logger.error(
                    "Stopping: critical source(s) failed this run (%s). Fix the data source and re-run "
                    "rather than train/predict on missing or broken data.",
                    ingest_result.failed_sources,
                )
                sys.exit(1)

            logger.info("Step 2/8: generating current news sentiment and digests")
            build_news_digests()

            logger.info("Step 3/8: building features")
            build_features()

            logger.info("Step 4/8: checking data quality (staleness, coverage, abnormal moves)")
            quality_results = run_data_quality_checks(ingest_result.run_id)
            for r in quality_results:
                if r["status"] != "pass":
                    logger.warning("  [%s] %s: %s", r["status"], r["check_name"], r["detail"])
            if has_blocking_failure(quality_results):
                logger.error("Stopping: blocking data-quality check failed; not training, backtesting, or predicting.")
                sys.exit(1)

            logger.info("Step 5/8: training models")
            train_models()

            if not args.skip_backtest:
                logger.info("Step 6/8: walk-forward backtesting")
                for h in get_settings()["models"]["horizons_days"]:
                    try:
                        logger.info("Backtest (%sd): %s", h, run_backtest(h))
                    except RuntimeError as e:
                        logger.warning("Backtest for horizon %sd skipped: %s", h, e)
            else:
                logger.info("Step 6/8: skipped (--skip-backtest)")

            logger.info("Step 7/8: running predictions (includes LLM reasons_summary if Ollama is running)")
            run_predictions(run_id=ingest_result.run_id)

            logger.info("Step 8/8: evaluating past predictions")
            evaluate_predictions()
            logger.info("Pipeline complete. Run `uvicorn app.main:app --reload` and open http://127.0.0.1:8000/dashboard")
    finally:
        # This includes failed/aborted runs, not just the successful path.
        unload_llm(get_settings())


if __name__ == "__main__":
    main()
