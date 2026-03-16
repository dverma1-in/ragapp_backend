import logging
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api_logger")


async def exception_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        error_trace = traceback.format_exc()
        logger.error(f"Unhandled error at {request.url.path}:\n{error_trace}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "type": type(exc).__name__,
            },
        )
