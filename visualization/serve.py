#!/usr/bin/env python3
"""
Simple HTTP server for trait-interp visualization.
Serves the visualization with proper CORS headers and API endpoints.
"""

import http.server
import socketserver
import os
import sys
import json
from pathlib import Path

PORT = 8000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support and API endpoints."""

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Handle GET requests, including API endpoints."""
        # API endpoint: list experiments
        if self.path == '/api/experiments':
            self.send_api_response(self.list_experiments())
            return

        # API endpoint: list traits for an experiment
        if self.path.startswith('/api/experiments/') and '/traits' in self.path:
            exp_name = self.path.split('/')[3]
            self.send_api_response(self.list_traits(exp_name))
            return

        # Default: serve files
        super().do_GET()

    def send_api_response(self, data):
        """Send JSON API response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def list_experiments(self):
        """List all experiments in experiments/ directory."""
        experiments_dir = Path('experiments')
        if not experiments_dir.exists():
            return {'experiments': []}

        experiments = []
        for item in experiments_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has at least one trait subdirectory
                has_traits = any(
                    (subdir / 'extraction').exists()
                    for subdir in item.iterdir()
                    if subdir.is_dir()
                )
                if has_traits:
                    experiments.append(item.name)

        return {'experiments': sorted(experiments)}

    def list_traits(self, experiment_name):
        """List all traits for an experiment."""
        exp_dir = Path('experiments') / experiment_name
        if not exp_dir.exists():
            return {'traits': []}

        traits = []
        for item in exp_dir.iterdir():
            if item.is_dir() and (item / 'extraction').exists():
                # Check for responses to confirm it's a real trait
                responses_dir = item / 'extraction' / 'responses'
                has_responses = (
                    responses_dir.exists() and (
                        (responses_dir / 'pos.csv').exists() or
                        (responses_dir / 'pos.json').exists()
                    )
                )
                if has_responses:
                    traits.append(item.name)

        return {'traits': sorted(traits)}

def main():
    # Change to the project root directory (parent of visualization/)
    os.chdir(Path(__file__).parent.parent)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Trait Interpretation Visualization Server          ║
╚══════════════════════════════════════════════════════════════╝

Starting server on http://localhost:{PORT}

Available visualizations:
  • Main: http://localhost:{PORT}/visualization/
  • Legacy: http://localhost:{PORT}/visualization/legacy.html

Press Ctrl+C to stop the server.
""")

    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)

if __name__ == "__main__":
    main()
