"""SEC EDGAR ingestion: company filings (10-K/10-Q/8-K/4) and, for Form 4s,
best-effort parsed insider transaction details.

Two DataSources share the CIK lookup + submissions fetch below:
- SecFilingsSource   -> sec_filings table (every filing, all types)
- InsiderTransactionsSource -> insider_transactions table (Form 4 detail)

EDGAR requires a descriptive User-Agent on every request and asks for a
light request rate — both are respected here (see env.sec_edgar_user_agent
and the sleep between per-filing detail fetches).
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from functools import lru_cache

import httpx
import pandas as pd

from src.data.base import DataSource, register_source

logger = logging.getLogger(__name__)

_FILING_TYPES = ("10-K", "10-Q", "8-K", "4")
_MAX_FILINGS_PER_TICKER = 40
_MAX_FORM4_DETAILS_PER_TICKER = 10


def _headers(env) -> dict:
    return {"User-Agent": env.sec_edgar_user_agent}


@lru_cache
def _cik_map(base_www_url: str, user_agent: str) -> dict[str, str]:
    resp = httpx.get(
        f"{base_www_url}/files/company_tickers.json",
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}


def _get_submissions(ticker: str, settings: dict, env) -> dict | None:
    cik_map = _cik_map(settings["data"]["sec_edgar_www_url"], env.sec_edgar_user_agent)
    cik = cik_map.get(ticker.upper())
    if cik is None:
        return None
    url = f"{settings['data']['sec_edgar_base_url']}/submissions/CIK{cik}.json"
    resp = httpx.get(url, headers=_headers(env), timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    data["_cik"] = cik
    return data


def _recent_filings_df(ticker: str, submissions: dict) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    cik = submissions["_cik"]
    df = pd.DataFrame(
        {
            "accession_number": recent.get("accessionNumber", []),
            "filing_type": recent.get("form", []),
            "filing_date": recent.get("filingDate", []),
            "primary_document": recent.get("primaryDocument", []),
        }
    )
    if df.empty:
        return df
    df = df[df["filing_type"].isin(_FILING_TYPES)].head(_MAX_FILINGS_PER_TICKER).copy()
    if df.empty:
        # pandas .apply(axis=1) misbehaves (returns a DataFrame, not a Series)
        # on an empty frame -- bail out before hitting that.
        return df
    df["ticker"] = ticker
    df["cik"] = cik
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df["url"] = df.apply(
        lambda r: f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{r['accession_number'].replace('-', '')}/{r['primary_document']}",
        axis=1,
    )
    return df[["accession_number", "ticker", "cik", "filing_type", "filing_date", "url"]]


def _raw_xml_url(url: str) -> str:
    """EDGAR's `primaryDocument` for Form 4s points at the XSLT-rendered HTML
    viewer (e.g. .../xslF345X06/form4.xml, served as text/html despite the
    extension). The raw ownership XML lives one directory up, same filename.
    """
    directory, filename = url.rsplit("/", 1)
    parent, last_segment = directory.rsplit("/", 1)
    if last_segment.lower().startswith("xsl"):
        directory = parent
    return f"{directory}/{filename}"


def _parse_form4(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)

    def find_text(el, path):
        node = el.find(path)
        return node.text.strip() if node is not None and node.text else None

    owner_name = find_text(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
    role_bits = []
    rel = root.find(".//reportingOwner/reportingOwnerRelationship")
    if rel is not None:
        for tag, label in [
            ("isDirector", "Director"),
            ("isOfficer", "Officer"),
            ("isTenPercentOwner", "10% Owner"),
        ]:
            node = rel.find(tag)
            if node is not None and node.text == "1":
                role_bits.append(label)
    role = ", ".join(role_bits) or None

    rows = []
    for tx in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        date = find_text(tx, "./transactionDate/value")
        code = find_text(tx, "./transactionCoding/transactionCode")
        shares = find_text(tx, "./transactionAmounts/transactionShares/value")
        price = find_text(tx, "./transactionAmounts/transactionPricePerShare/value")
        shares_after = find_text(tx, "./postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        if not date or not code or not shares:
            continue
        shares_f = float(shares)
        price_f = float(price) if price else None
        rows.append(
            {
                "insider_name": owner_name,
                "role": role,
                "transaction_date": date,
                "transaction_code": code,
                "shares": shares_f,
                "price": price_f,
                "value": (shares_f * price_f) if price_f else None,
                "shares_owned_after": float(shares_after) if shares_after else None,
            }
        )
    return rows


@register_source("sec_filings")
class SecFilingsSource(DataSource):
    table = "sec_filings"
    key_cols = ("accession_number",)

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        frames = []
        for t in tickers:
            try:
                submissions = _get_submissions(t, settings, env)
            except Exception:
                logger.warning("EDGAR submissions fetch failed for %s", t, exc_info=True)
                continue
            if submissions is None:
                continue
            df = _recent_filings_df(t, submissions)
            if not df.empty:
                frames.append(df)
            time.sleep(0.15)
        if not frames:
            return pd.DataFrame(columns=["accession_number", "ticker", "cik", "filing_type", "filing_date", "url"])
        return pd.concat(frames, ignore_index=True)


@register_source("insider_transactions")
class InsiderTransactionsSource(DataSource):
    table = "insider_transactions"
    key_cols = ("ticker", "insider_name", "transaction_date", "transaction_code", "shares")

    def fetch(self, tickers: list[str], settings: dict, env) -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                submissions = _get_submissions(t, settings, env)
            except Exception:
                logger.warning("EDGAR submissions fetch failed for %s", t, exc_info=True)
                continue
            if submissions is None:
                continue
            filings = _recent_filings_df(t, submissions)
            form4s = filings[filings["filing_type"] == "4"].head(_MAX_FORM4_DETAILS_PER_TICKER)
            for _, filing in form4s.iterrows():
                try:
                    raw_url = _raw_xml_url(filing["url"])
                    resp = httpx.get(raw_url, headers=_headers(env), timeout=30)
                    if resp.status_code != 200 or "xml" not in resp.headers.get("content-type", ""):
                        continue
                    parsed = _parse_form4(resp.text)
                except Exception:
                    logger.debug("Form 4 parse failed for %s (%s)", t, filing["accession_number"], exc_info=True)
                    continue
                for r in parsed:
                    r["ticker"] = t
                    r["filing_url"] = filing["url"]
                    # The transaction itself isn't public until this Form 4
                    # is actually filed -- insiders get up to 2 business days
                    # to file, so a feature that treats transaction_date as
                    # "known by" would occasionally use information before it
                    # was actually public. filing_date is EDGAR's own record
                    # of when the document became public, always >= the
                    # transaction date, so it's the safe point-in-time cutoff.
                    r["filing_date"] = filing["filing_date"]
                    rows.append(r)
                time.sleep(0.15)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker", "insider_name", "transaction_date", "transaction_code", "shares",
                    "role", "price", "value", "shares_owned_after", "filing_url", "filing_date",
                ]
            )
        df = pd.DataFrame(rows)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
        # Two distinct line items (different footnotes/prices) can otherwise
        # collide on our (ticker, insider, date, code, shares) key -- keep
        # the first and move on rather than fail the whole batch on it.
        return df.drop_duplicates(subset=list(self.key_cols), keep="first")
