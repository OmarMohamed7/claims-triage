import csv
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi_swagger_ui_theme import setup_swagger_ui_theme
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import config
from src.graph import build_pipeline
from src.schemas import ClaimRequest, ClaimSubmission, PipelineState

logging.basicConfig(level=logging.INFO)

USAGE_LOG = "usage.csv"
CSV_HEADER = [
    "Request ID", "Datetime", "Endpoint Triggered", "Client IP Address",
    "Response Time", "Status Code", "Successful",
]


limiter = Limiter(key_func=get_remote_address)


def check_artifacts() -> None:
    missing = []
    if not Path(config.models.fraud_model_path).is_file():
        missing.append("fraud model (run: uv run python -m scripts.train_fraud_model)")
    if not Path(config.paths.qdrant_path).is_dir():
        missing.append("policy index (run: uv run python -m scripts.build_policy_index)")
    if missing:
        raise RuntimeError("Missing artifacts: " + "; ".join(missing))


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    check_artifacts()
    app.state.pipeline = build_pipeline()
    print(f"Hello from Claims Triage \nApp Version: {config.VERSION}\n")
    yield
    del app.state.pipeline


app = FastAPI(docs_url=None, lifespan=app_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

setup_swagger_ui_theme(app=app, docs_path="/docs")


@app.middleware("http")
async def monitor_service(
    req: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    req_id = uuid4().hex
    request_datetime = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    response = await call_next(req)
    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = req_id
    with open(USAGE_LOG, "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(CSV_HEADER)
        writer.writerow(
            [
                req_id,
                request_datetime,
                req.url,
                req.client.host if req.client else "",
                response_time,
                response.status_code,
                response.status_code < 400,
            ]
        )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "version": config.VERSION}


@app.get("/models")
def models():
    return {
        "default": config.models.default_llm,
        "choices": config.models.llm_choices,
    }


@app.post("/claims", response_model=PipelineState)
@limiter.limit(config.api.rate_limit)
def submit_claim(request: Request, body: ClaimRequest):
    if (
        body.custom_llm is None
        and body.model is not None
        and body.model not in config.models.llm_choices
    ):
        raise HTTPException(status_code=400, detail=f"Model not allowed: {body.model}")
    submission = ClaimSubmission(
        submission_id=uuid4().hex,
        raw_text=body.raw_text,
        policy_number=body.policy_number,
    )
    result = request.app.state.pipeline.invoke(
        PipelineState(
            submission=submission, model=body.model, custom_llm=body.custom_llm
        )
    )
    return PipelineState(**result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
