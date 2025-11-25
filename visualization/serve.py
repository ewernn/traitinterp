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
import subprocess
from pathlib import Path

PORT = int(os.environ.get('PORT', 8000))

# Cache for integrity data (populated on startup)
integrity_cache = {}

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support and API endpoints."""

    # Add explicit MIME types for JavaScript and CSS
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.css': 'text/css',
    }

    def log_message(self, format, *args):
        """Override to suppress noisy 404 logs and improve error messages."""
        # Suppress 404s for HEAD requests (file existence checks)
        if len(args) >= 2 and args[1] == '404' and self.command == 'HEAD':
            return

        # Suppress 304 (Not Modified) for less noise
        if len(args) >= 2 and args[1] == '304':
            return

        # Enhanced 404 logging - show what file was requested
        if len(args) >= 2 and args[1] == '404':
            # Extract path from request string "GET /path HTTP/1.1"
            request_parts = args[0].split(' ')
            path = request_parts[1] if len(request_parts) > 1 else args[0]

            # Clarify common issues
            if '/visualization/experiments/' in path:
                self.log_error("❌ 404: %s (Should not have /visualization/ prefix - bug in frontend path construction)", path)
            elif '/experiments/' in path:
                self.log_error("❌ 404: %s (File missing or not synced from R2)", path)
            else:
                self.log_error("❌ 404: %s", path)
            return

        # Suppress generic "File not found" messages (redundant with above)
        if 'File not found' in format or 'code 404' in format:
            return

        # Log everything else normally
        super().log_message(format, *args)

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
        try:
            # API endpoint: list experiments
            if self.path == '/api/experiments':
                self.send_api_response(self.list_experiments())
                return

            # API endpoint: get integrity data for an experiment
            if self.path.startswith('/api/integrity/') and self.path.endswith('.json'):
                exp_name = self.path.split('/')[3].replace('.json', '')
                if exp_name in integrity_cache:
                    self.send_api_response(integrity_cache[exp_name])
                else:
                    self.send_api_response({'error': f'No integrity data for experiment: {exp_name}'})
                return

            # API endpoint: list traits for an experiment
            if self.path.startswith('/api/experiments/') and '/traits' in self.path:
                exp_name = self.path.split('/')[3]
                self.send_api_response(self.list_traits(exp_name))
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

            # Serve SPA at root
            if self.path == '/' or self.path == '/index.html':
                self.path = '/visualization/index.html'

            # Backwards compatibility redirects
            elif self.path == '/overview':
                self.send_response(301)
                self.send_header('Location', '/?tab=overview')
                self.end_headers()
                return

            elif self.path == '/visualization/' or self.path == '/visualization/index.html':
                self.send_response(301)
                self.send_header('Location', '/?tab=data-explorer')
                self.end_headers()
                return

            # Serve design playground
            if self.path == '/design':
                self.path = '/visualization/design.html'

            # Default: serve files
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            self.log_message("Connection broken by client")

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
                has_traits = False

                # Check for extraction/{category}/ structure
                extraction_dir = item / 'extraction'
                if extraction_dir.exists():
                    for category_dir in extraction_dir.iterdir():
                        if not category_dir.is_dir():
                            continue

                        # Check if any subdirectory has trait data (responses or vectors)
                        for trait_dir in category_dir.iterdir():
                            if not trait_dir.is_dir():
                                continue

                            responses_dir = trait_dir / 'responses'
                            vectors_dir = trait_dir / 'vectors'
                            has_responses = (
                                responses_dir.exists() and
                                (responses_dir / 'pos.json').exists()
                            )
                            has_vectors = vectors_dir.exists() and len(list(vectors_dir.glob('*.pt'))) > 0

                            if has_responses or has_vectors:
                                has_traits = True
                                break

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

        # Check for extraction/{category}/ structure
        extraction_dir = exp_dir / 'extraction'
        if not extraction_dir.exists():
            return {'traits': []}

        for category_dir in extraction_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Check all subdirectories as potential trait directories
            for trait_item in category_dir.iterdir():
                if trait_item.is_dir():
                    # Check for responses OR vectors to confirm it's a real trait
                    responses_dir = trait_item / 'responses'
                    vectors_dir = trait_item / 'vectors'
                    has_responses = (
                        responses_dir.exists() and
                        (responses_dir / 'pos.json').exists()
                    )
                    has_vectors = vectors_dir.exists() and len(list(vectors_dir.glob('*.pt'))) > 0
                    if has_responses or has_vectors:
                        # Use category/trait format
                        traits.append(f"{category_dir.name}/{trait_item.name}")

        return {'traits': sorted(traits)}

    def list_prompt_sets(self, experiment_name):
        """List all prompt sets with available prompt IDs for an experiment.

        Discovers available prompts by scanning raw/residual/{prompt_set}/ directories.
        Also loads prompt definitions from inference/prompts/{set}.json files.
        """
        exp_dir = Path('experiments') / experiment_name
        prompts_def_dir = exp_dir / 'inference' / 'prompts'
        raw_residual_dir = exp_dir / 'inference' / 'raw' / 'residual'

        prompt_sets = []

        # Get prompt set definitions from JSON files
        if prompts_def_dir.exists():
            for prompt_file in prompts_def_dir.glob('*.json'):
                set_name = prompt_file.stem

                # Load prompt definitions
                try:
                    with open(prompt_file) as f:
                        definition = json.load(f)
                except Exception:
                    definition = {'prompts': []}

                # Discover available IDs from raw/residual/{set}/
                available_ids = []
                set_raw_dir = raw_residual_dir / set_name
                if set_raw_dir.exists():
                    for pt_file in set_raw_dir.glob('*.pt'):
                        # Parse ID from filename: "1.pt" -> 1
                        try:
                            prompt_id = int(pt_file.stem)
                            available_ids.append(prompt_id)
                        except ValueError:
                            continue

                prompt_sets.append({
                    'name': set_name,
                    'description': definition.get('description', ''),
                    'prompts': definition.get('prompts', []),
                    'available_ids': sorted(available_ids)
                })

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

def cache_integrity_data():
    """Run check_available_data.py for each experiment and cache results."""
    experiments_dir = Path('experiments')
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue

        # Check if experiment has extraction data
        extraction_dir = exp_dir / 'extraction'
        if not extraction_dir.exists():
            continue

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    'analysis/check_available_data.py',
                    '--experiment', exp_dir.name,
                    '--json_output'
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                integrity_cache[exp_dir.name] = json.loads(result.stdout)
                print(f"  ✓ Cached integrity for {exp_dir.name}")
            else:
                print(f"  ✗ Failed to get integrity for {exp_dir.name}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout getting integrity for {exp_dir.name}")
        except json.JSONDecodeError as e:
            print(f"  ✗ Invalid JSON from integrity check for {exp_dir.name}: {e}")
        except Exception as e:
            print(f"  ✗ Error getting integrity for {exp_dir.name}: {e}")


def main():
    # Change to the project root directory (parent of visualization/)
    os.chdir(Path(__file__).parent.parent)

    # Use a ThreadingTCPServer to handle multiple concurrent requests from the browser,
    # preventing BrokenPipeErrors when the frontend makes many simultaneous fetch calls.
    class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        pass

    Handler = CORSHTTPRequestHandler

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Trait Interpretation Visualization Server          ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Cache integrity data for all experiments on startup
    print("Caching integrity data...")
    cache_integrity_data()
    print(f"Cached {len(integrity_cache)} experiment(s)\n")

    print(f"""Starting server on http://localhost:{PORT}

Available at:
  • http://localhost:{PORT}/              (Overview - landing page)
  • http://localhost:{PORT}/visualization/index.html  (Dashboard)

Press Ctrl+C to stop the server.
""")

    try:
        with ThreadingTCPServer(("", PORT), Handler) as httpd_server:
            httpd_server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)

if __name__ == "__main__":
    main()
