"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from portrait981_serve.config import ServeConfig
from portrait981_serve.routes import init_routes, router

logger = logging.getLogger(__name__)


def create_app(config: ServeConfig | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    if config is None:
        config = ServeConfig.from_env()

    app = FastAPI(
        title="portrait981-serve",
        description="REST API for portrait981 pipeline",
        version="0.1.0",
    )

    init_routes(config)
    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main() -> None:
    """CLI entry point: p981-serve."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
