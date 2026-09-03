from app.api import control_plane
from app.main import app


def _route(path: str):
    return next(route for route in app.routes if getattr(route, "path", None) == path)


def test_health_endpoint_is_available_without_starting_the_scheduler():
    response = _route("/health").endpoint()

    assert response == {"status": "ok", "service": "quant-platform"}


def test_control_plane_status_is_read_only_and_queues_default_work():
    response = control_plane.status()

    assert response == {"queued": 10}


def test_control_plane_quota_endpoint_enforces_recurring_free_only_defaults():
    response = control_plane.gpu_quota()

    assert response
    assert all(provider["recurring_free_only"] for provider in response)
    assert all(provider["hard_stop_on_exhaustion"] for provider in response)


def test_control_plane_router_is_registered_with_the_application():
    assert any(getattr(route, "original_router", None) is control_plane.router for route in app.routes)


def test_model_catalog_exposes_specialists_without_provider_specific_details():
    models = control_plane.models()

    assert {model["name"] for model in models} >= {"xgboost", "finbert", "chronos-2", "timesfm-2.5"}
    assert next(model for model in models if model["name"] == "xgboost")["tier"] == "publish_critical"
