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

        # API endpoint: cross-distribution data index
        if self.path == '/api/cross-distribution/index':
            self.send_cross_dist_index()
            return

        # API endpoint: cross-distribution results for a trait
        if self.path.startswith('/api/cross-distribution/results/'):
            trait_name = self.path.split('/')[-1]
            self.send_cross_dist_results(trait_name)
            return

        # Serve index.html at root
        if self.path == '/':
            self.path = '/visualization/index.html'

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

    def send_cross_dist_index(self):
        """Send cross-distribution data index."""
        index_path = Path('results/cross_distribution_analysis/data_index.json')
        if not index_path.exists():
            self.send_api_response({'error': 'Index not found. Run analysis/cross_distribution_scanner.py'})
            return

        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
            self.send_api_response(data)
        except Exception as e:
            self.send_api_response({'error': str(e)})

    def send_cross_dist_results(self, trait_name):
        """Send cross-distribution results for a specific trait."""
        # Try to find the results file for this trait
        results_dir = Path('results/cross_distribution_analysis')
        result_files = [
            results_dir / f'{trait_name}_full_4x4_results.json',
            results_dir / f'{trait_name}_cross_dist_results.json',
        ]

        for result_file in result_files:
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    self.send_api_response(data)
                    return
                except Exception as e:
                    self.send_api_response({'error': str(e)})
                    return

        self.send_api_response({'error': f'No results found for {trait_name}'})

def main():
    # Change to the project root directory (parent of visualization/)
    os.chdir(Path(__file__).parent.parent)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Trait Interpretation Visualization Server          ║
╚══════════════════════════════════════════════════════════════╝

Starting server on http://localhost:{PORT}

Available at:
  • http://localhost:{PORT}/

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
