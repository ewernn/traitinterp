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

    # Add explicit MIME types for JavaScript and CSS
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.css': 'text/css',
    }

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

        # API endpoint: list inference prompt sets
        if self.path.startswith('/api/experiments/') and '/inference/prompt-sets' in self.path:
            exp_name = self.path.split('/')[3]
            self.send_api_response(self.list_prompt_sets(exp_name))
            return

        # API endpoint: list prompts in a prompt set
        if self.path.startswith('/api/experiments/') and '/inference/prompts/' in self.path:
            parts = self.path.split('/')
            exp_name = parts[3]
            prompt_set = parts[6]
            self.send_api_response(self.list_prompts_in_set(exp_name, prompt_set))
            return

        # API endpoint: get inference projection data
        if self.path.startswith('/api/experiments/') and '/inference/projections/' in self.path:
            # Path: /api/experiments/{exp}/inference/projections/{category}/{trait}/{set}/{prompt_id}
            parts = self.path.split('/')
            if len(parts) >= 10:
                exp_name = parts[3]
                category = parts[6]
                trait = parts[7]
                prompt_set = parts[8]
                prompt_id = parts[9]
                self.send_inference_projection(exp_name, category, trait, prompt_set, prompt_id)
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
        category_names = {'behavioral', 'cognitive', 'stylistic', 'alignment'}

        for item in experiments_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                has_traits = False

                # Check for categorized structure
                for subdir in item.iterdir():
                    if not subdir.is_dir():
                        continue

                    # If it's a category directory, check inside for traits
                    if subdir.name in category_names:
                        has_traits = any(
                            (trait_dir / 'extraction').exists()
                            for trait_dir in subdir.iterdir()
                            if trait_dir.is_dir()
                        )
                        if has_traits:
                            break

                if has_traits:
                    experiments.append(item.name)

        return {'experiments': sorted(experiments)}

    def list_traits(self, experiment_name):
        """List all traits for an experiment."""
        exp_dir = Path('experiments') / experiment_name
        if not exp_dir.exists():
            return {'traits': []}

        traits = []

        # Check for categorized structure (behavioral, cognitive, stylistic, alignment)
        category_names = {'behavioral', 'cognitive', 'stylistic', 'alignment'}

        for item in exp_dir.iterdir():
            if not item.is_dir():
                continue

            # If this is a category directory, look inside it
            if item.name in category_names:
                for trait_item in item.iterdir():
                    if trait_item.is_dir() and (trait_item / 'extraction').exists():
                        # Check for responses OR vectors to confirm it's a real trait
                        responses_dir = trait_item / 'extraction' / 'responses'
                        vectors_dir = trait_item / 'extraction' / 'vectors'
                        has_responses = (
                            responses_dir.exists() and (
                                (responses_dir / 'pos.csv').exists() or
                                (responses_dir / 'pos.json').exists()
                            )
                        )
                        has_vectors = vectors_dir.exists() and len(list(vectors_dir.glob('*.pt'))) > 0
                        if has_responses or has_vectors:
                            # Use category/trait format
                            traits.append(f"{item.name}/{trait_item.name}")

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

    def list_prompt_sets(self, experiment_name):
        """List all prompt sets for an experiment's inference directory."""
        prompts_dir = Path('experiments') / experiment_name / 'inference' / 'prompts'
        if not prompts_dir.exists():
            return {'prompt_sets': []}

        prompt_sets = []
        for prompt_file in prompts_dir.glob('*.txt'):
            # Count prompts in file
            try:
                with open(prompt_file) as f:
                    num_prompts = sum(1 for line in f if line.strip())
                prompt_sets.append({
                    'name': prompt_file.stem,
                    'num_prompts': num_prompts
                })
            except Exception:
                continue

        return {'prompt_sets': sorted(prompt_sets, key=lambda x: x['name'])}

    def list_prompts_in_set(self, experiment_name, prompt_set):
        """List all prompts in a specific prompt set."""
        prompt_file = Path('experiments') / experiment_name / 'inference' / 'prompts' / f'{prompt_set}.txt'
        if not prompt_file.exists():
            return {'error': f'Prompt set not found: {prompt_set}'}

        try:
            with open(prompt_file) as f:
                prompts = [line.strip() for line in f if line.strip()]
            return {
                'prompt_set': prompt_set,
                'prompts': prompts
            }
        except Exception as e:
            return {'error': str(e)}

    def send_inference_projection(self, experiment_name, category, trait, prompt_set, prompt_id):
        """Send inference projection data for a specific trait and prompt."""
        projection_file = (
            Path('experiments') / experiment_name / 'inference' / 'projections' /
            category / trait / prompt_set / f'{prompt_id}.json'
        )

        if not projection_file.exists():
            self.send_api_response({
                'error': f'Projection not found: {category}/{trait}/{prompt_set}/{prompt_id}'
            })
            return

        try:
            with open(projection_file, 'r') as f:
                data = json.load(f)
            self.send_api_response(data)
        except Exception as e:
            self.send_api_response({'error': str(e)})

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
