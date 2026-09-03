from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from src.cloud.contracts import JobDefinition, JobState, StructuredStorage
from src.cloud.storage import InMemoryStructuredStorage


class ControlPlane:
    def __init__(self, storage: StructuredStorage | None = None) -> None:
        self.storage = storage or InMemoryStructuredStorage()
        self.jobs: dict[str, JobDefinition] = {}

    def submit(self, job: JobDefinition) -> JobDefinition:
        self.jobs[job.job_id] = job
        self.storage.write_record("jobs", self._job_record(job))
        return job

    def submit_many(self, jobs: list[JobDefinition]) -> list[JobDefinition]:
        return [self.submit(job) for job in jobs]

    def get(self, job_id: str) -> JobDefinition:
        return self.jobs[job_id]

    def ready_jobs(self, now: datetime | None = None) -> list[JobDefinition]:
        now = now or datetime.now(UTC)
        jobs = [job for job in self.jobs.values() if self._is_ready(job, now)]
        return sorted(jobs, key=lambda job: (job.priority, job.deadline_at or datetime.max.replace(tzinfo=UTC), job.created_at))

    def mark_running(self, job_id: str) -> JobDefinition:
        job = replace(self.jobs[job_id], state=JobState.RUNNING, attempts=self.jobs[job_id].attempts + 1)
        self.jobs[job_id] = job
        self.storage.write_record("job_events", self._event(job, "running"))
        return job

    def mark_succeeded(self, job_id: str, result: dict | None = None) -> JobDefinition:
        result = result or {}
        job = replace(self.jobs[job_id], state=JobState.SUCCEEDED, result=result, last_error=None)
        self.jobs[job_id] = job
        self.storage.write_record("job_events", self._event(job, "succeeded"))
        return job

    def mark_partial_publish(self, job_id: str, result: dict | None = None) -> JobDefinition:
        result = result or {}
        job = replace(self.jobs[job_id], state=JobState.PUBLISHED_STALE, result=result, last_error=None)
        self.jobs[job_id] = job
        self.storage.write_record("job_events", self._event(job, "published_stale"))
        return job

    def mark_failed(self, job_id: str, error: str, now: datetime | None = None) -> JobDefinition:
        now = now or datetime.now(UTC)
        job = self.jobs[job_id]
        if job.attempts <= job.max_retries and (job.deadline_at is None or now <= job.deadline_at):
            updated = replace(job, state=JobState.QUEUED, last_error=error)
        else:
            updated = replace(job, state=JobState.FAILED, last_error=error)
        self.jobs[job_id] = updated
        self.storage.write_record("job_events", self._event(updated, "failed"))
        return updated

    def blocked_jobs(self, now: datetime | None = None) -> list[JobDefinition]:
        now = now or datetime.now(UTC)
        return [job for job in self.jobs.values() if not self._is_ready(job, now) and job.state in {JobState.QUEUED, JobState.WAITING}]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job.state.value] = counts.get(job.state.value, 0) + 1
        return counts

    def _is_ready(self, job: JobDefinition, now: datetime) -> bool:
        if job.state not in {JobState.QUEUED, JobState.WAITING}:
            return False
        if job.deadline_at and job.deadline_at < now and job.dependencies:
            return False
        for dep in job.dependencies:
            dep_job = self.jobs.get(dep)
            if dep_job is None or dep_job.state not in {JobState.SUCCEEDED, JobState.PUBLISHED_STALE}:
                return False
        return True

    @staticmethod
    def _job_record(job: JobDefinition) -> dict:
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "priority": int(job.priority),
            "state": job.state.value,
            "deadline_at": job.deadline_at.isoformat() if job.deadline_at else None,
            "dependencies": list(job.dependencies),
            "max_retries": job.max_retries,
            "fallback_job_types": list(job.fallback_job_types),
            "allow_partial_publish": job.allow_partial_publish,
        }

    @staticmethod
    def _event(job: JobDefinition, event: str) -> dict:
        return {"job_id": job.job_id, "event": event, "state": job.state.value, "at": datetime.now(UTC).isoformat()}
