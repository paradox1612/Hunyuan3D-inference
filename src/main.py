from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Hunyuan3D Serverless API",
        description="A pure serverless, high-performance API for Tencent's Hunyuan3D models",
        version="0.1.0",
        docs_url=None,  # Disable default docs
        redoc_url=None,  # Disable default redoc
        openapi_url="/openapi.json"  # Keep OpenAPI JSON
    )

    # Mount static files for docs
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    from .api.endpoints import router as api_router
    app.include_router(api_router, prefix="/v1")

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Hunyuan3D Serverless API",
            version="0.1.0",
            description="A pure serverless, high-performance API for Tencent's Hunyuan3D models",
            routes=app.routes,
        )
        
        # Add server information
        openapi_schema["servers"] = [
            {
                "url": "/",
                "description": "Current server"
            },
            {
                "url": "https://api.example.com",
                "description": "Production server"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi

    # Root endpoint redirects to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    # Health check endpoint
    @app.get(
        "/health", 
        status_code=status.HTTP_200_OK,
        summary="Health Check",
        description="Check if the API is running and healthy"
    )
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "version": "0.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # Custom docs endpoint
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Hunyuan3D API - Swagger UI",
            swagger_favicon_url="https://hunyuan.tencent.com/favicon.ico",
            swagger_ui_parameters={
                "defaultModelsExpandDepth": -1,
                "filter": "",
                "displayRequestDuration": True,
                "tryItOutEnabled": True,
            }
        )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler"""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": type(exc).__name__,
                    "details": str(exc)
                }
            },
        )

    return app

# Create the FastAPI app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
