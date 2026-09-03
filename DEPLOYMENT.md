# Cloud Deployment

This repository is deployment-ready, but it intentionally has no provider-specific deployment target or credentials committed. The container serves the dashboard, `/health`, and read-only control-plane endpoints. It starts with `ENABLE_PIPELINE_SCHEDULER=false` so deploying it cannot trigger the legacy in-process scheduler.

## Before the first push

1. Create an empty GitHub repository and add it as `origin`.
2. Review the current uncommitted worktree, then make the first commit. GitHub Actions runs tests and builds the container on every push to `main` and pull request.
3. Add secrets only in the deployment environment or GitHub repository settings. Never place them in `.env.example`, source files, logs, or commits.

## Cloud configuration

Set these runtime values in the selected cloud service:

```text
ENABLE_PIPELINE_SCHEDULER=false
PAID_FALLBACK_ENABLED=false
SEC_EDGAR_USER_AGENT=Your Name your-email@example.com
```

Add provider credentials only after the corresponding real adapter is implemented:

```text
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
KAGGLE_USERNAME=
KAGGLE_KEY=
HUGGINGFACE_TOKEN=
S3_ENDPOINT_URL=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
METADATA_DATABASE_URL=
```

## Verification after deployment

1. Request `GET /health`; it must return `{"status":"ok","service":"quant-platform"}`.
2. Request `GET /api/control-plane/status` and `GET /api/control-plane/gpu-quota`.
3. Run `python scripts/cloudctl.py predict` in the deployed worker environment after real GPU adapters are connected.
4. Confirm an exhausted quota queues or stops work. It must not select a paid provider.

## Required implementation before automated production jobs

The current API only reports the in-memory control-plane plan and configured stub quotas. Before enabling automated collection, training, or inference, connect:

- a durable metadata store implementing `StructuredStorage`;
- an object/Parquet store implementing `ObjectStorage`;
- authenticated Modal, Kaggle, and Hugging Face adapters; and
- a cloud scheduler or event source that invokes the control plane.

Keep `paid_fallback_enabled: false` in `config/settings.yaml` and validate each provider's current recurring-free terms before enabling it. Free-tier availability and limits are account- and provider-specific and can change.
