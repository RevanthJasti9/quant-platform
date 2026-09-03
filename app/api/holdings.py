from __future__ import annotations

import logging
import socket
from contextlib import contextmanager
from urllib.parse import quote

import yfinance as yf
from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

from src.config import get_settings
from src.data.ingest import run_ingest
from src.features.build import build_features
from src.holdings import add_or_update_holding, remove_holding, update_holding
from src.models.predict import run_predictions

router = APIRouter()
logger = logging.getLogger(__name__)

VALIDATION_TIMEOUT_SECONDS = 8
FETCH_TIMEOUT_SECONDS = 20


@contextmanager
def _bounded_yfinance_call(timeout_seconds: float):
    # yfinance itself has no timeout parameter, and Yahoo's unofficial API
    # occasionally rate-limits/stalls under heavy use (this session has
    # made a lot of calls) -- bound the wait so a request fails fast with a
    # clear error instead of hanging for a minute. socket.setdefaulttimeout
    # is process-global for its duration; fine here since this is a small
    # local single-user app, not a concurrent multi-user server.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _ticker_has_data(ticker: str) -> bool:
    try:
        with _bounded_yfinance_call(VALIDATION_TIMEOUT_SECONDS):
            return not yf.Ticker(ticker).history(period="5d").empty
    except Exception:
        return False


def _error_redirect(message: str) -> RedirectResponse:
    return RedirectResponse(f"/dashboard?holdings_error={quote(message)}", status_code=303)


@router.post("/holdings/add")
def holdings_add(ticker: str = Form(...), shares: float = Form(...), total_paid: float = Form(...)):
    ticker = ticker.upper().strip()
    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        return _error_redirect(f'"{ticker}" doesn\'t look like a valid ticker symbol')
    if shares <= 0 or total_paid <= 0:
        return _error_redirect("Shares and total amount paid both need to be positive numbers")
    if not _ticker_has_data(ticker):
        return _error_redirect(f'Couldn\'t find a stock with the symbol "{ticker}"')

    add_or_update_holding(ticker, shares, total_paid / shares)

    # Best-effort: fetch this ticker's data and get it a forecast right away,
    # instead of making the user wait for the next scheduled run. If this
    # fails, the holding is still recorded and will be picked up by the next
    # full pipeline run.
    try:
        settings = get_settings()
        with _bounded_yfinance_call(FETCH_TIMEOUT_SECONDS):
            ingest_result = run_ingest(sources=["prices", "fundamentals"], tickers=[ticker, settings["benchmark"]])
        build_features()
        # Scoped to just this ticker -- without this, predicting "the
        # whole universe" (including SHAP + LLM reasons for every other
        # ticker) just to onboard one new holding made "Add" take minutes.
        run_predictions(tickers=[ticker], run_id=ingest_result.run_id)
    except Exception:
        logger.warning("On-demand fetch for newly added holding %s failed", ticker, exc_info=True)

    return RedirectResponse("/dashboard", status_code=303)


@router.post("/holdings/update")
def holdings_update(ticker: str = Form(...), shares: float = Form(...), total_paid: float = Form(...)):
    ticker = ticker.upper().strip()
    if shares <= 0 or total_paid <= 0:
        return _error_redirect("Shares and total amount paid both need to be positive numbers")
    if not update_holding(ticker, shares, total_paid / shares):
        return _error_redirect(f'Could not find a saved holding for "{ticker}"')
    return RedirectResponse("/dashboard", status_code=303)


@router.post("/holdings/remove")
def holdings_remove(ticker: str = Form(...)):
    remove_holding(ticker)
    return RedirectResponse("/dashboard", status_code=303)
