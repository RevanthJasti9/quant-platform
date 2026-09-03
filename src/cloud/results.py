"""Shared specialist-result store and publish gatekeeper.

Specialists write independently. The gatekeeper is the only component that
combines their outputs, so replacing a GPU provider or model never changes
the dashboard/prediction contract.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

from src.cloud.contracts import ModelSpec, ModelTier, PublishedPrediction, ResultStatus, SpecialistResult, StructuredStorage


class SpecialistResultStore:
    def __init__(self, storage: StructuredStorage) -> None:
        self.storage = storage

    def write(self, result: SpecialistResult) -> None:
        record = asdict(result)
        record["as_of"] = result.as_of.isoformat()
        record["data_timestamp"] = result.data_timestamp.isoformat() if result.data_timestamp else None
        record["status"] = result.status.value
        self.storage.write_record("specialist_results", record)

    def for_prediction(self, ticker: str, as_of: datetime, horizon_days: int) -> list[SpecialistResult]:
        results = []
        for record in self.storage.read_records("specialist_results"):
            if record["ticker"] != ticker or record["horizon_days"] != horizon_days or record["as_of"] != as_of.isoformat():
                continue
            results.append(
                SpecialistResult(
                    ticker=record["ticker"],
                    as_of=as_of,
                    model_name=record["model_name"],
                    model_version=record["model_version"],
                    specialist=record["specialist"],
                    horizon_days=record["horizon_days"],
                    prediction=record.get("prediction"),
                    probability=record.get("probability"),
                    confidence=record.get("confidence"),
                    uncertainty=record.get("uncertainty"),
                    features_version=record.get("features_version"),
                    data_timestamp=datetime.fromisoformat(record["data_timestamp"]) if record.get("data_timestamp") else None,
                    status=ResultStatus(record["status"]),
                    artifact_uri=record.get("artifact_uri"),
                    metadata=record.get("metadata", {}),
                )
            )
        return results


class PredictionGatekeeper:
    def __init__(self, specs: list[ModelSpec]) -> None:
        self.specs = {spec.name: spec for spec in specs}

    def publishable(self, results: list[SpecialistResult]) -> list[PublishedPrediction]:
        grouped: dict[tuple[str, datetime, int], list[SpecialistResult]] = defaultdict(list)
        for result in results:
            grouped[(result.ticker, result.as_of, result.horizon_days)].append(result)
        return [self._publish_one(group) for group in grouped.values() if self._has_price_result(group)]

    def _publish_one(self, results: list[SpecialistResult]) -> PublishedPrediction:
        ready = [result for result in results if result.status == ResultStatus.READY]
        price = [
            result
            for result in ready
            if self.specs.get(result.model_name) and self.specs[result.model_name].specialist == "price"
        ]
        if not price:
            raise ValueError("A prediction needs at least one ready price specialist")

        weights = [self.specs[result.model_name].ensemble_weight or 1.0 for result in price]
        weight_by_model = {result.model_name: weight for result, weight in zip(price, weights)}
        total_weight = sum(weights)
        prediction = sum((result.prediction or 0.0) * weight for result, weight in zip(price, weights)) / total_weight
        probability_values = [result for result in price if result.probability is not None]
        confidence_values = [result for result in price if result.confidence is not None]
        probability = self._weighted_value(probability_values, weight_by_model, "probability")
        confidence = self._weighted_value(confidence_values, weight_by_model, "confidence")
        expected_price = {spec.name for spec in self.specs.values() if spec.specialist == "price" and spec.tier <= ModelTier.ENHANCEMENT}
        contributing = tuple(sorted(result.model_name for result in ready))
        pending = tuple(sorted(expected_price - set(contributing)))
        latest_data = max((result.data_timestamp for result in ready if result.data_timestamp), default=None)
        return PublishedPrediction(
            ticker=price[0].ticker,
            as_of=price[0].as_of,
            horizon_days=price[0].horizon_days,
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            freshness="current" if not pending else "partial",
            contributing_models=contributing,
            pending_models=pending,
            refresh_required=bool(pending),
            data_timestamp=latest_data,
        )

    def _has_price_result(self, results: list[SpecialistResult]) -> bool:
        return any(result.status == ResultStatus.READY and self.specs.get(result.model_name) and self.specs[result.model_name].specialist == "price" for result in results)

    @staticmethod
    def _weighted_value(results: list[SpecialistResult], weights: dict[str, float], field: str) -> float | None:
        if not results:
            return None
        values = [getattr(result, field) for result in results]
        selected_weights = [weights[result.model_name] for result in results]
        return sum(value * weight for value, weight in zip(values, selected_weights)) / sum(selected_weights)
