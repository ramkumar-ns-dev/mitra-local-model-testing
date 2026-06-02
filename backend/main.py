import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api.routes import router
from backend.services.model_loader import model_loader
from backend.utils.logging_config import get_logger, set_request_id

logger = get_logger()

# Async lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Initializing application startup...")
    startup_start = time.time()
    
    # Load ASR and translation models
    try:
        model_loader.load_models()
        logger.info(f"Models loaded successfully during startup in {time.time() - startup_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")
        
    yield
    
    # Shutdown tasks (cleanup if any)
    logger.info("Shutting down application...")

# Initialize FastAPI App
app = FastAPI(
    title=settings.API_TITLE,
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for external/frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging and Request ID Propagation Middleware
@app.middleware("http")
async def log_and_track_request(request: Request, call_next):
    # 1. Establish Request ID (Check header, or generate new UUID)
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_request_id(req_id)
    
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    # 2. Process Request
    response = None
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(f"Unhandled exception during request processing: {exc}")
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."}
        )
    finally:
        process_time_ms = int((time.time() - start_time) * 1000)
        # Apply Request-ID back to response headers
        if response:
            response.headers["X-Request-ID"] = req_id
            logger.info(f"Finished request: {request.method} {request.url.path} -> Status {response.status_code} in {process_time_ms}ms")
        else:
            logger.info(f"Finished request with error in {process_time_ms}ms")
            
    return response

# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Global exception caught: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Include API Router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Start the server locally
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False  # Disabled reload to prevent model reloading in dev loops
    )
