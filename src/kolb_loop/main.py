from __future__ import annotations

import uvicorn

from kolb_loop.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "kolb_loop.ingress.app:create_app",
        factory=True,
        host=settings.sidecar.ingress.host,
        port=settings.sidecar.ingress.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
