from __future__ import annotations

from typing import Sequence

from voxcpm_runtime.config import RuntimeConfig


def main(argv: Sequence[str] | None = None) -> int:
    if argv not in (None, (), []):
        raise SystemExit("voxcpm-serve does not accept command-line options; use VOXCPM_* environment variables")
    config = RuntimeConfig.from_env()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("The API extra is required: pip install 'openbmb-voxcpm2-inference[api]'") from None

    from voxcpm_runtime.api_app import create_app

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        log_level=config.log_level.value.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
