from src.data.news_intelligence import classify_event, duplicate_group, normalize_news_event
from src.data.company_ir_news import parse_press_release_feed


def test_classifier_assigns_expected_event_types():
    assert classify_event("CEO sells shares after filing Form 4") == "insider_transaction"
    assert classify_event("Company raises full-year guidance after earnings") == "earnings_guidance"
    assert classify_event("Company launches a new product") == "product"


def test_normalized_event_keeps_source_lineage_and_duplicate_group():
    event = normalize_news_event(
        ticker="amzn",
        headline="Amazon launches a new service",
        url="https://example.com/article",
        published_at="2026-08-29T12:00:00Z",
        source="Example News",
        provider="polygon",
    )

    assert event["ticker"] == "AMZN"
    assert event["provider"] == "polygon"
    assert event["event_type"] == "product"
    assert event["reliability_score"] == 0.85
    assert event["duplicate_group"] == duplicate_group("Amazon launches a new service")
    assert event["received_at"] >= event["published_at"]


def test_company_ir_rss_feed_is_stored_as_an_official_source():
    rows = parse_press_release_feed(
        """<rss><channel><item><title>Company launches a new product</title>
        <link>https://investor.example.com/release</link>
        <pubDate>Fri, 29 Aug 2026 12:00:00 GMT</pubDate></item></channel></rss>""",
        "amzn",
    )

    assert rows[0]["provider"] == "company_ir"
    assert rows[0]["reliability_score"] == 0.95
    assert rows[0]["event_type"] == "product"
