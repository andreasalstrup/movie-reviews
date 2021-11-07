from fastapi import Request, Response
from fastapi.applications import FastAPI
from loguru import logger

from utilities.logging.config import initialize_logging


async def _http_request_logging_middleware(request: Request, call_next) -> Response:
    """
    Intercepts all HTTP requests and responses to log rudimentary information.
    The request and response objects are assigned to the record.extra dict.
    """
    logger.info(f'HTTP {request.method} for {request.url}', request=request)
    response: Response = await call_next(request)
    logger.info(f'HTTP {response.status_code} for {request.url}',
                request=request, response=response)
    return response


def setup(api: FastAPI):
    initialize_logging()
    api.middleware("http")(_http_request_logging_middleware)
