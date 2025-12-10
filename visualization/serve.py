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
import yaml
import subprocess
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import get as get_path

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
        """Override to suppress noisy logs."""
        # Suppress 404s for HEAD requests (file existence checks)
        if len(args) >= 2 and args[1] == '404' and self.command == 'HEAD':
            return

        # Suppress 304 (Not Modified)
        if len(args) >= 2 and args[1] == '304':
            return

        # Suppress all 404s (browser requests for optional files)
        if len(args) >= 2 and args[1] == '404':
            return

        # Suppress base class "code 404" messages
        if 'code 404' in format or 'File not found' in format:
            return

        # Log successful requests and errors
        super().log_message(format, *args)

    def log_error(self, format, *args):
        """Override to suppress error logs for expected 404s."""
        # Suppress all 404 error logs
        if '404' in format:
            return
        super().log_error(format, *args)

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Handle GET requests, including API endpoints."""
        try:
            # API endpoint: get data schema
            if self.path == '/api/schema':
                self.send_api_response(self.get_schema())
                return

            # API endpoint: list experiments
            if self.path == '/api/experiments':
                self.send_api_response(self.list_experiments())
                return

            # API endpoint: get experiment config
            if self.path.startswith('/api/experiments/') and self.path.endswith('/config'):
                exp_name = self.path.split('/')[3]
                self.send_api_response(self.get_experiment_config(exp_name))
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

            # Serve SPA at root (including with query params like /?tab=...)
            if self.path == '/' or self.path == '/index.html' or self.path.startswith('/?'):
                self.path = '/visualization/index.html'

            # Serve design playground
            if self.path == '/design':
                self.path = '/visualization/design.html'

            # Default: serve files
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            # Browser cancelled request - expected when loading many files
            pass

    def do_POST(self):
        """Handle POST requests for chat API."""
        try:
            if self.path == '/api/chat':
                self.handle_chat_stream()
                return

            # Unknown POST endpoint
            self.send_error(404, "Not Found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_chat_stream(self):
        """Stream chat response with trait scores via SSE."""
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        prompt = data.get('prompt', '')
        experiment = data.get('experiment', 'gemma-2-2b-it')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        history = data.get('history', [])  # Multi-turn conversation history

        if not prompt:
            self.send_error(400, "Missing prompt")
            return

        # Send SSE headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        try:
            # Import here to avoid loading model on server start
            from visualization.chat_inference import get_chat_instance

            chat = get_chat_instance(experiment)

            for event in chat.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, history=history):
                sse_data = f"data: {json.dumps(event)}\n\n"
                self.wfile.write(sse_data.encode())
                self.wfile.flush()

        except Exception as e:
            error_event = f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
            self.wfile.write(error_event.encode())
            self.wfile.flush()

    def send_api_response(self, data):
        """Send JSON API response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_schema(self):
        """Get data schema from paths.yaml."""
        config_path = Path(__file__).parent.parent / 'config' / 'paths.yaml'
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get('schema', {})
        except Exception as e:
            return {'error': str(e)}

    def get_experiment_config(self, experiment: str):
        """Get experiment config.json (extraction_model, application_model)."""
        config_path = get_path('experiments.config', experiment=experiment)
        if not config_path.exists():
            return {}
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            return {'error': str(e)}

    def list_experiments(self):
        """List all experiments in experiments/ directory."""
        experiments_dir = get_path('experiments.list')
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

        # Sort alphabetically, but prioritize gemma-2-2b-it as default
        def sort_key(name):
            if name == 'gemma-2-2b-it':
                return (0, name)
            return (1, name)
        return {'experiments': sorted(experiments, key=sort_key)}

    def list_traits(self, experiment_name):
        """List all traits for an experiment."""
        exp_dir = get_path('experiments.base', experiment=experiment_name)
        if not exp_dir.exists():
            return {'traits': []}

        traits = []

        # Check for extraction/{category}/ structure
        extraction_dir = get_path('extraction.base', experiment=experiment_name)
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

        Discovers available prompts by scanning:
        1. Per-trait projection JSONs: inference/{trait}/residual_stream/{prompt_set}/*.json
        2. Fallback to raw .pt files: raw/residual/{prompt_set}/*.pt

        Prompt definitions loaded from datasets/inference/{set}.json.
        """
        prompts_def_dir = get_path('datasets.inference')
        inference_dir = get_path('inference.base', experiment=experiment_name)
        raw_residual_dir = get_path('inference.raw', experiment=experiment_name) / 'residual'

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

                # Discover available IDs from projection JSONs across all traits
                available_ids = set()

                # Check per-trait projection directories
                if inference_dir.exists():
                    # Scan all trait directories for residual_stream/{set}/*.json
                    for trait_dir in inference_dir.iterdir():
                        if not trait_dir.is_dir() or trait_dir.name == 'raw' or trait_dir.name == 'prompts':
                            continue
                        # Handle nested category/trait structure
                        for subdir in trait_dir.rglob('residual_stream'):
                            set_proj_dir = subdir / set_name
                            if set_proj_dir.exists():
                                for json_file in set_proj_dir.glob('*.json'):
                                    try:
                                        prompt_id = int(json_file.stem)
                                        available_ids.add(prompt_id)
                                    except ValueError:
                                        continue

                # Fallback: check raw .pt files if no projections found
                if not available_ids:
                    set_raw_dir = raw_residual_dir / set_name
                    if set_raw_dir.exists():
                        for pt_file in set_raw_dir.glob('*.pt'):
                            try:
                                prompt_id = int(pt_file.stem)
                                available_ids.add(prompt_id)
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
        prompt_file = get_path('datasets.inference_prompt_set', prompt_set=prompt_set)
        if not prompt_file.exists():
            return {'error': f'Prompt set not found: {prompt_set}'}

        try:
            with open(prompt_file) as f:
                data = json.load(f)
            return {
                'prompt_set': prompt_set,
                'prompts': data.get('prompts', [])
            }
        except Exception as e:
            return {'error': str(e)}

    def send_inference_projection(self, experiment_name, category, trait, prompt_set, prompt_id):
        """Send inference projection data for a specific trait and prompt."""
        trait_path = f"{category}/{trait}"
        projection_dir = get_path('inference.residual_stream', experiment=experiment_name, trait=trait_path, prompt_set=prompt_set)
        filename = get_path('patterns.residual_stream_json', prompt_id=prompt_id)
        projection_file = projection_dir / filename

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
    """Run data_checker.py for each experiment and cache results."""
    experiments_dir = get_path('experiments.list')
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue

        # Check if experiment has extraction data
        extraction_dir = get_path('extraction.base', experiment=exp_dir.name)
        if not extraction_dir.exists():
            continue

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    'analysis/data_checker.py',
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
  • http://localhost:{PORT}/                    (Dashboard - defaults to Overview tab)
  • http://localhost:{PORT}/?tab=data-explorer  (Data Explorer)
  • http://localhost:{PORT}/?tab=overview       (Overview documentation)

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
