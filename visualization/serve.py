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
        """Override to suppress noisy logs. Only show errors and POST requests."""
        # Extract status code
        status = args[1] if len(args) >= 2 else None

        # Always log errors (4xx, 5xx) except expected 404s
        is_error = status and str(status).startswith(('4', '5'))
        is_expected_404 = status == '404' or status == 404

        if is_error and not is_expected_404:
            super().log_message(format, *args)
            return

        # Always log POST requests (chat, etc.)
        if self.command == 'POST':
            super().log_message(format, *args)
            return

        # Suppress everything else (200 OK for static assets, API calls, etc.)

    def log_error(self, format, *args):
        """Override to suppress error logs for expected 404s."""
        # Suppress all 404 error logs
        # Format is "code %d, message %s" with code as first arg
        if args and args[0] == 404:
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

            # API endpoint: app config (mode, features)
            if self.path == '/api/config':
                self.send_api_response(self.get_app_config())
                return

            # API endpoint: Modal warmup (GET for simplicity - triggers container warm-up)
            if self.path == '/api/modal/warmup':
                self.send_api_response(self.warmup_modal())
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

            # API endpoint: list available model variants
            if self.path.startswith('/api/experiments/') and self.path.endswith('/model-variants'):
                exp_name = self.path.split('/')[3]
                self.send_api_response(self.list_model_variants(exp_name))
                return

            # API endpoint: list steering entries
            if self.path.startswith('/api/experiments/') and self.path.endswith('/steering'):
                exp_name = self.path.split('/')[3]
                self.send_api_response(self.list_steering_entries(exp_name))
                return

            # API endpoint: get steering results (loads JSONL, returns JSON)
            # Path: /api/experiments/{exp}/steering-results/{trait}/{model_variant}/{position}/{prompt_set...}
            # Note: prompt_set can be nested (e.g., rm_syco/train_100)
            if self.path.startswith('/api/experiments/') and '/steering-results/' in self.path:
                parts = self.path.split('/')
                if len(parts) >= 9:
                    exp_name = parts[3]
                    trait = f"{parts[5]}/{parts[6]}"
                    model_variant = parts[7]
                    position = parts[8]
                    prompt_set = '/'.join(parts[9:]) if len(parts) > 9 else 'steering'
                    self.send_api_response(self.get_steering_results(exp_name, trait, model_variant, position, prompt_set))
                return

            # API endpoint: list steering response files
            # Path: /api/experiments/{exp}/steering-responses/{trait}/{model_variant}/{position}/{prompt_set...}
            # Note: prompt_set can be nested (e.g., rm_syco/train_100)
            if self.path.startswith('/api/experiments/') and '/steering-responses/' in self.path:
                parts = self.path.split('/')
                if len(parts) >= 9:
                    exp_name = parts[3]
                    trait = f"{parts[5]}/{parts[6]}"
                    model_variant = parts[7]
                    position = parts[8]
                    prompt_set = '/'.join(parts[9:]) if len(parts) > 9 else 'steering'
                    self.send_api_response(self.list_steering_responses(exp_name, trait, model_variant, position, prompt_set))
                return

            # API endpoint: get inference projection data
            # Path: /api/experiments/{exp}/inference/{model_variant}/projections/{category}/{trait}/{set}/{prompt_id}
            if self.path.startswith('/api/experiments/') and '/inference/' in self.path and '/projections/' in self.path:
                parts = self.path.split('/')
                if len(parts) >= 11:
                    exp_name = parts[3]
                    model_variant = parts[5]
                    category = parts[7]
                    trait = parts[8]
                    prompt_set = parts[9]
                    prompt_id = parts[10]
                    self.send_inference_projection(exp_name, model_variant, category, trait, prompt_set, prompt_id)
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
        # Default to first discovered experiment if not specified
        if 'experiment' not in data:
            experiments = self.list_experiments()['experiments']
            experiment = experiments[0] if experiments else None
            if not experiment:
                self.send_error(400, "No experiments found")
                return
        else:
            experiment = data['experiment']
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.0)  # Default to greedy for live-chat
        history = data.get('history', [])  # Multi-turn conversation history
        previous_context_length = data.get('previous_context_length', 0)  # Tokens already captured
        model_type = data.get('model_type', 'application')  # Which model to use: extraction or application
        inference_mode = data.get('inference_mode')  # 'local' or 'modal' - per-request override
        steering_configs = data.get('steering_configs', [])  # [{trait, coefficient}, ...]

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

            chat = get_chat_instance(experiment, backend=inference_mode, model_type=model_type)

            for event in chat.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, history=history, previous_context_length=previous_context_length, steering_configs=steering_configs):
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

    def get_app_config(self):
        """Get app-wide config for frontend (mode, features)."""
        mode = os.environ.get('MODE', 'development')

        # Mode determines available features
        is_dev = mode == 'development'

        return {
            'mode': mode,
            'features': {
                'model_picker': is_dev,        # Show model dropdown in dev
                'experiment_picker': is_dev,   # Show experiment picker in dev
                'inference_toggle': is_dev,    # Show local/modal toggle in dev
                'debug_info': is_dev,          # Show debug info in dev
                'steering': True,              # Always show steering (it's the point)
            },
            'defaults': {
                'inference_backend': 'local' if is_dev else 'modal',
                'experiment': 'live-chat',
                'model': 'google/gemma-2-2b-it',  # Default for production
            }
        }

    def get_experiment_config(self, experiment: str):
        """Get experiment config.json with model metadata (max_context_length, etc.)."""
        from utils.model_registry import get_model_config

        config_path = get_path('experiments.config', experiment=experiment)
        if not config_path.exists():
            return {}
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Enhance with model metadata - get from default application variant
            app_variant = config.get('defaults', {}).get('application', 'instruct')
            model_variants = config.get('model_variants', {})
            app_model = model_variants.get(app_variant, {}).get('model', 'google/gemma-2-2b-it')
            try:
                model_config = get_model_config(app_model)
                config['max_context_length'] = model_config.get('max_context_length', 8192)
            except Exception:
                config['max_context_length'] = 8192  # Default fallback

            return config
        except Exception as e:
            return {'error': str(e)}

    def warmup_modal(self):
        """Warm up Modal GPU container by loading the model."""
        print("[Warmup] Starting Modal warmup...")
        try:
            # Get model from live-chat experiment config
            from utils.paths import get_model_variant
            variant_info = get_model_variant('live-chat', mode='application')
            model_name = variant_info['model']
            print(f"[Warmup] Model from config: {model_name}")

            import sys
            inference_path = Path(__file__).parent.parent / "inference"
            if str(inference_path) not in sys.path:
                sys.path.insert(0, str(inference_path))

            print("[Warmup] Importing modal_inference...")
            import modal_inference

            print(f"[Warmup] Calling warmup.remote({model_name})...")
            with modal_inference.app.run():
                result = modal_inference.warmup.remote(model_name=model_name)

            print(f"[Warmup] Success: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"[Warmup] Error: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e)
            }

    def list_experiments(self):
        """List all experiments in experiments/ directory."""
        experiments_dir = get_path('experiments.list')
        if not experiments_dir.exists():
            return {'experiments': []}

        experiments = []

        for item in experiments_dir.iterdir():
            # Skip hidden dirs and 'live-chat' (internal experiment for public demo)
            if item.is_dir() and not item.name.startswith('.') and item.name != 'live-chat':
                has_traits = False

                # Check for extraction/{category}/{trait}/{model_variant}/ structure
                extraction_dir = item / 'extraction'
                if extraction_dir.exists():
                    for category_dir in extraction_dir.iterdir():
                        if not category_dir.is_dir():
                            continue

                        # Check if any subdirectory has trait data
                        for trait_dir in category_dir.iterdir():
                            if not trait_dir.is_dir():
                                continue

                            # Look for model_variant subdirs (e.g., base, instruct)
                            for variant_dir in trait_dir.iterdir():
                                if not variant_dir.is_dir():
                                    continue

                                responses_dir = variant_dir / 'responses'
                                vectors_dir = variant_dir / 'vectors'
                                has_responses = (
                                    responses_dir.exists() and
                                    (responses_dir / 'pos.json').exists()
                                )
                                # Vectors are in {position}/{component}/{method}/ subdirs
                                has_vectors = vectors_dir.exists() and len(list(vectors_dir.rglob('layer*.pt'))) > 0

                                if has_responses or has_vectors:
                                    has_traits = True
                                    break

                            if has_traits:
                                break

                        if has_traits:
                            break

                if has_traits:
                    experiments.append(item.name)

        # Sort alphabetically
        return {'experiments': sorted(experiments)}

    def list_traits(self, experiment_name):
        """List all traits for an experiment."""
        exp_dir = get_path('experiments.base', experiment=experiment_name)
        if not exp_dir.exists():
            return {'traits': []}

        traits = []

        # Check for extraction/{category}/{trait}/{model_variant}/ structure
        extraction_dir = get_path('extraction.base', experiment=experiment_name)
        if not extraction_dir.exists():
            return {'traits': []}

        for category_dir in extraction_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Check all subdirectories as potential trait directories
            for trait_item in category_dir.iterdir():
                if trait_item.is_dir():
                    # Look for model_variant subdirs with responses or vectors
                    for variant_dir in trait_item.iterdir():
                        if not variant_dir.is_dir():
                            continue
                        responses_dir = variant_dir / 'responses'
                        vectors_dir = variant_dir / 'vectors'
                        has_responses = (
                            responses_dir.exists() and
                            (responses_dir / 'pos.json').exists()
                        )
                        # Vectors are in {position}/{component}/{method}/ subdirs
                        has_vectors = vectors_dir.exists() and len(list(vectors_dir.rglob('layer*.pt'))) > 0
                        if has_responses or has_vectors:
                            # Use category/trait format
                            traits.append(f"{category_dir.name}/{trait_item.name}")
                            break  # Found valid variant, no need to check others

        return {'traits': sorted(traits)}

    def list_prompt_sets(self, experiment_name):
        """List all prompt sets with available prompt IDs for an experiment.

        Discovers prompt sets by scanning projection directories:
        inference/{model_variant}/projections/{trait}/{prompt_set}/*.json
        """
        inference_dir = get_path('inference.base', experiment=experiment_name)

        # Discover prompt sets from projection directories
        discovered_sets = {}  # set_name -> set of available IDs

        if inference_dir.exists():
            # Look in inference/{model_variant}/projections/...
            for variant_dir in inference_dir.iterdir():
                if not variant_dir.is_dir():
                    continue
                projections_dir = variant_dir / 'projections'
                if not projections_dir.exists():
                    continue
                # Scan projections/{category}/{trait}/{prompt_set}/
                for category_dir in projections_dir.iterdir():
                    if not category_dir.is_dir():
                        continue
                    for trait_dir in category_dir.iterdir():
                        if not trait_dir.is_dir():
                            continue
                        for set_dir in trait_dir.iterdir():
                            if not set_dir.is_dir():
                                continue
                            set_name = set_dir.name
                            if set_name not in discovered_sets:
                                discovered_sets[set_name] = set()
                            for json_file in set_dir.glob('*.json'):
                                try:
                                    discovered_sets[set_name].add(int(json_file.stem))
                                except ValueError:
                                    continue

        # Build response with prompt definitions included
        prompt_sets = []
        for name, ids in sorted(discovered_sets.items()):
            if not ids:
                continue
            # Load prompt definitions from dataset file
            prompt_file = get_path('datasets.inference_prompt_set', prompt_set=name)
            prompts = []
            if prompt_file.exists():
                try:
                    with open(prompt_file) as f:
                        data = json.load(f)
                    prompts = data.get('prompts', [])
                except Exception:
                    pass
            prompt_sets.append({
                'name': name,
                'available_ids': sorted(ids),
                'prompts': prompts
            })

        return {'prompt_sets': prompt_sets}

    def list_model_variants(self, experiment_name):
        """List available model variants for an experiment."""
        config = self.get_experiment_config(experiment_name)
        if 'error' in config:
            return config
        model_variants = config.get('model_variants', {})
        defaults = config.get('defaults', {})
        return {
            'variants': list(model_variants.keys()),
            'defaults': defaults,
            'model_variants': model_variants
        }

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

    def list_steering_entries(self, experiment_name):
        """List all steering entries for an experiment."""
        from utils.paths import discover_steering_entries
        return {'entries': discover_steering_entries(experiment_name)}

    def list_steering_responses(self, experiment_name, trait, model_variant, position, prompt_set):
        """List all response files for a steering entry."""
        from utils.paths import get as get_path
        responses_dir = get_path('steering.responses', experiment=experiment_name, trait=trait,
                                  model_variant=model_variant, position=position, prompt_set=prompt_set)

        if not responses_dir.exists():
            return {'files': [], 'baseline': None}

        files = []
        baseline = None

        # Find baseline.json
        baseline_file = responses_dir / 'baseline.json'
        if baseline_file.exists():
            baseline = 'baseline.json'

        # Find all response files: {component}/{method}/L{layer}_c{coef}_{timestamp}.json
        for component_dir in responses_dir.iterdir():
            if not component_dir.is_dir():
                continue
            component = component_dir.name
            for method_dir in component_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                for resp_file in method_dir.glob('*.json'):
                    # Parse filename: L{layer}_c{coef}_{timestamp}.json
                    name = resp_file.stem
                    parts = name.split('_')
                    if len(parts) >= 2 and parts[0].startswith('L'):
                        layer = int(parts[0][1:])
                        coef = float(parts[1][1:]) if parts[1].startswith('c') else 0
                        files.append({
                            'path': f'{component}/{method}/{resp_file.name}',
                            'component': component,
                            'method': method,
                            'layer': layer,
                            'coef': coef,
                            'filename': resp_file.name
                        })

        # Sort by layer, then coef
        files.sort(key=lambda x: (x['layer'], x['coef']))

        return {'files': files, 'baseline': baseline}

    def get_steering_results(self, experiment_name, trait, model_variant, position, prompt_set):
        """Load steering results from JSONL file and return as JSON."""
        from analysis.steering.results import load_results

        try:
            # sanitize_position is idempotent, so already-sanitized positions work fine
            return load_results(experiment_name, trait, model_variant, position, prompt_set)
        except FileNotFoundError:
            return {'error': f'Results not found for {trait}/{model_variant}/{position}/{prompt_set}'}
        except Exception as e:
            return {'error': str(e)}

    def send_inference_projection(self, experiment_name, model_variant, category, trait, prompt_set, prompt_id):
        """Send inference projection data for a specific trait and prompt.

        Supports both formats:
        - Multi-vector: {id}.json with multi_vector=True
        - Individual files: {id}_{method}_L{layer}.json (combined on-the-fly)
        """
        trait_path = f"{category}/{trait}"
        projection_dir = get_path('inference.projections', experiment=experiment_name, model_variant=model_variant, trait=trait_path, prompt_set=prompt_set)
        filename = f"{prompt_id}.json"
        projection_file = projection_dir / filename

        try:
            # Try to load main file
            main_data = None
            if projection_file.exists():
                with open(projection_file, 'r') as f:
                    main_data = json.load(f)

                # If already multi-vector format, return as-is
                if main_data.get('metadata', {}).get('multi_vector'):
                    self.send_api_response(main_data)
                    return

            # Look for individual vector files: {id}_{method}_L{layer}.json
            import re
            pattern = re.compile(rf'^{re.escape(prompt_id)}_(\w+)_L(\d+)\.json$')
            individual_files = []

            if projection_dir.exists():
                for f in projection_dir.iterdir():
                    match = pattern.match(f.name)
                    if match:
                        individual_files.append((f, match.group(1), int(match.group(2))))

            # If we have individual files, combine them into multi-vector format
            if individual_files:
                all_projections = []
                base_metadata = None
                activation_norms = None
                token_norms = None

                for file_path, method, layer in sorted(individual_files, key=lambda x: (x[1], x[2])):
                    with open(file_path, 'r') as f:
                        vec_data = json.load(f)

                    # Use first file for base metadata
                    if base_metadata is None:
                        base_metadata = vec_data.get('metadata', {})
                        activation_norms = vec_data.get('activation_norms')
                        token_norms = vec_data.get('token_norms')

                    vs = vec_data.get('metadata', {}).get('vector_source', {})
                    proj = vec_data.get('projections', {})

                    all_projections.append({
                        'method': method,
                        'layer': layer,
                        'selection_source': vs.get('selection_source', 'unknown'),
                        'baseline': vs.get('baseline', 0),
                        'prompt': proj.get('prompt', []),
                        'response': proj.get('response', [])
                    })

                # Build combined multi-vector response
                combined = {
                    'metadata': {
                        'prompt_id': prompt_id,
                        'prompt_set': prompt_set,
                        'n_prompt_tokens': base_metadata.get('n_prompt_tokens', 0),
                        'n_response_tokens': base_metadata.get('n_response_tokens', 0),
                        'multi_vector': True,
                        'n_vectors': len(all_projections),
                        'combined_from_files': True
                    },
                    'projections': all_projections
                }
                if activation_norms:
                    combined['activation_norms'] = activation_norms
                if token_norms:
                    combined['token_norms'] = token_norms

                self.send_api_response(combined)
                return

            # Fall back to single-vector main file if it exists
            if main_data:
                self.send_api_response(main_data)
                return

            # No data found
            self.send_api_response({
                'error': f'Projection not found: {category}/{trait}/{prompt_set}/{prompt_id}'
            })

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
