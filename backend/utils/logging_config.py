import logging
import sys
import uuid
from contextvars import ContextVar

# ContextVar to store request-id in thread-safe and async-safe manner
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    """
    Python logging filter that injects request_id into the log record.
    """
    def filter(self, record):
        record.request_id = request_id_var.get() or "N/A"
        return True

def setup_logging():
    """
    Configures structured logging for the application.
    """
    logger = logging.getLogger("stt_service")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Console Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Custom Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [ReqID: %(request_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Add Filter and Handler
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

# Initialize logger
logger = logging.getLogger("stt_service")
setup_logging()

def get_logger():
    return logger

def set_request_id(req_id: str = None) -> str:
    """
    Set request ID in the context variable. Creates a new one if not provided.
    """
    if not req_id:
        req_id = str(uuid.uuid4())
    request_id_var.set(req_id)
    return req_id
