# docs/api_documentation.py
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import json
from datetime import datetime
import importlib.util
import inspect
import os

class APIDocumentation:
    """Generate and serve API documentation using OpenAPI/Swagger."""
    
    def __init__(self, title: str = "N5StyleEA API", version: str = "1.0.0"):
        self.app = FastAPI(
            title=title,
            version=version,
            description="API documentation for N5StyleEA trading system",
            docs_url=None,  # Disable default docs
            redoc_url=None  # Disable default redoc
        )
        
        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self.app.get("/docs", include_in_schema=False)(self.swagger_ui)
        self.app.get("/openapi.json", include_in_schema=False)(self.openapi_json)
        
        # Store API metadata
        self.title = title
        self.version = version
        self.tags_metadata = []
        self.paths = {}
        
    def add_endpoint(self, path: str, methods: list, **kwargs):
        """Add an API endpoint to the documentation."""
        if path not in self.paths:
            self.paths[path] = {}
            
        for method in methods:
            self.paths[path][method.lower()] = {
                "summary": kwargs.get("summary", ""),
                "description": kwargs.get("description", ""),
                "parameters": kwargs.get("parameters", []),
                "responses": kwargs.get("responses", {
                    "200": {"description": "Successful Response"},
                    "422": {"description": "Validation Error"}
                }),
                "tags": kwargs.get("tags", [])
            }
            
    def add_tag(self, name: str, description: str = ""):
        """Add a tag for API grouping."""
        self.tags_metadata.append({
            "name": name,
            "description": description
        })
        
    def generate_from_module(self, module_path: str):
        """Generate API documentation from a module using docstrings."""
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find all route handlers
        for name, obj in inspect.getmembers(module):
            if hasattr(obj, "route"):
                route = obj.route
                self.add_endpoint(
                    path=route["path"],
                    methods=route["methods"],
                    summary=inspect.getdoc(obj) or "",
                    tags=route.get("tags", [])
                )
                
    def generate_openapi_schema(self) -> Dict[str, Any]:
        """Generate the OpenAPI schema."""
        return {
            "openapi": "3.0.2",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": "N5StyleEA Trading System API",
                "contact": {
                    "name": "API Support",
                    "email": "support@n5styleea.com"
                }
            },
            "tags": self.tags_metadata,
            "paths": self.paths
        }
        
    async def openapi_json(self):
        """Serve the OpenAPI JSON schema."""
        return self.generate_openapi_schema()
        
    async def swagger_ui(self):
        """Serve the Swagger UI."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{self.title} - Swagger UI"
        )
        
    def save_to_file(self, output_dir: Path = Path("docs/api")):
        """Save API documentation to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save OpenAPI JSON
        schema = self.generate_openapi_schema()
        with open(output_dir / "openapi.json", "w") as f:
            json.dump(schema, f, indent=2)
            
        # Save OpenAPI YAML
        with open(output_dir / "openapi.yaml", "w") as f:
            yaml.dump(schema, f, sort_keys=False)
            
        # Generate HTML documentation
        self._generate_html_docs(output_dir)
        
    def _generate_html_docs(self, output_dir: Path):
        """Generate HTML documentation using Redoc."""
        template = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>{self.title} - API Documentation</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 0; }}
            </style>
        </head>
        <body>
            <redoc spec-url='openapi.json'></redoc>
            <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
        </body>
        </html>"""
        
        (output_dir / "index.html").write_text(template)

# Example usage:
if __name__ == "__main__":
    # Initialize documentation
    docs = APIDocumentation(
        title="N5StyleEA Trading API",
        version="1.0.0"
    )
    
    # Add tags
    docs.add_tag("authentication", "User authentication and authorization")
    docs.add_tag("trading", "Trading operations")
    docs.add_tag("market_data", "Market data endpoints")
    
    # Add example endpoint
    docs.add_endpoint(
        "/api/v1/orders",
        methods=["GET", "POST"],
        summary="List or create trading orders",
        description="Get a list of all orders or create a new order",
        tags=["trading"],
        parameters=[
            {
                "name": "symbol",
                "in": "query",
                "description": "Trading pair symbol",
                "required": True,
                "schema": {"type": "string"}
            }
        ]
    )
    
    # Save documentation
    docs.save_to_file()
    
    print("API documentation generated in docs/api/")