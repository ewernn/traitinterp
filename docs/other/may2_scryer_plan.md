# scryer — v0 plan

**Date written:** 2026-05-02
**Status:** Design locked; implementation not started
**Repo target:** `~/code/scryer/` (separate from traitinterp)
**PyPI name target:** `scryer`
**Audience for this doc:** Claude Code reading it in a future session to implement scryer; or future-Eric returning to the project

---

## 0. How to read this doc

This doc captures decisions made during a long design conversation. It is NOT:
- A schema spec (full SQL is regenerable from the noun set + design choices here)
- An API reference (endpoint enumeration is its own session)
- A CLI design (command structure is its own session)
- A repo layout (module structure is its own session)
- An implementation plan with task ordering (lightweight staging is in §17)

It IS:
- Every architectural decision locked
- The first-class noun set with reasoning per noun
- Naming conventions and rejected alternatives
- Things deferred to v1 with the conditions under which they'd be added

When reading: section §10 (noun set) and §11 (schema overview) are the densest; everything else can be skimmed in any order.

---

## 1. What scryer is

scryer is a project-agnostic LLM evaluation harness with a control-plane-shaped resource model. The category is "eval framework" — peer to Inspect (UK AISI), lm-evaluation-harness (EleutherAI), Braintrust, Phoenix (Arize), LangSmith, OpenAI evals.

Honest framing: scryer is **NOT** an agent control plane in the Guild sense (Guild manages production agents; scryer manages eval execution). But scryer DOES have all 6 control-plane primitives applied to the eval vertical:

| Primitive | scryer's version |
|---|---|
| Identity + Auth | User, ServiceAccount, ApiKey, JWT |
| Resource model | Workspace > Project > resources |
| Scheduling/Execution | Trigger fires Tasks; subprocess executor |
| State + Persistence | Postgres + R2 |
| Observability | Run, Result, Trace, AuditEvent |
| Policy + Governance | per-resource sharing, soft-delete, Budget |

The phrase "eval control plane" is honest because it carries the control-plane primitives; the data plane it manages is eval execution rather than running production agents.

Positioning vs alternatives:

- **Inspect (UK AISI)** — Local-first research tool. No multi-user, no API, eval logs are files. scryer is multi-user with a real backend.
- **lm-evaluation-harness** — Academic benchmark framework (HuggingFace integration, YAML tasks). scryer is code-first (Python `@task` decorator), not YAML-first.
- **Braintrust** — Commercial SaaS eval platform with multi-user from the start. scryer matches that shape but is OSS-friendly and self-hostable.
- **Phoenix** — Observability-first; eval bolted on. scryer is eval-first.
- **LangSmith** — Trace-centric (production traffic + offline eval). scryer defers the production-trace ingestion to v1.
- **Guild** — Production agent platform; manages running agents. scryer evaluates agents but doesn't run them in production.

The closest reference is **Inspect** — same code-first ergonomics — augmented with multi-user infrastructure (workspaces, auth, audit) borrowed from Braintrust/Phoenix.

---

## 2. Goals & non-goals

### v0 must do

- Support 8 use cases (see §3) — primarily judge prompt iteration as the first user, with agent eval secondary
- Multi-user with real auth (invite-only beta)
- Server-side execution of user-supplied Scorer/Agent code in a sandbox
- Versioned, content-hashed resources with reproducibility
- Audit log capturing every state mutation
- Cron-triggered scheduled evals
- Outbound webhooks for integration
- A working dashboard for browsing Runs/Results/Datasets

### v0 won't do (deferred to v1; see §16)

- Online eval on production traces (Trace ingestion + async scoring)
- Human-rater pipeline (multi-rater Annotation)
- RL preference data primitives (formal Comparison noun)
- Approval gating workflow (use ad-hoc patterns until needed)
- Plan noun for multi-step agent work
- Real Docker-grade sandbox (use Tier 1+ subprocess for invite-only)
- Hash chain on AuditEvent (use Postgres WAL + offsite backup)
- Tag claim/lock manager
- Public unauthenticated share links
- GitHub-source-linked user code
- Bulk endpoints, SSE streaming, agent_feed unified inbox
- Curated MCP server (ship `/mcp/full` only initially)

### Never in scope

- Becoming a generic agent control plane (that's Guild's vertical)
- Hosting production traffic for agents (out of scope)
- Replacing TRL / axolotl for training (scryer is eval, not training)

### Constraints driving v0 scope

- **Solo dev, invite-only beta** — primary user is you (Eric); collaborators are 1-3 trusted humans + your own Claude Code sessions
- **Cost-conscious** — target ~$20-40/month all-in
- **"Build it right"** — chose Architecture 3 (full backend with multi-user, dashboard, MCP server) over a local-tool-only shape
- **Production-codebase intent** — scryer is for ongoing real use, not just an interview artifact

---

## 3. Use cases in scope

The 12 generic eval use cases were enumerated; v0 supports 8.

### v0 use cases

1. **Static rubric eval** — score pre-existing model outputs against a labeled rubric (judge prompt iteration; the first user via traitinterp's `experiments/judge_optimization/`)
2. **Outcome agent eval** — run a multi-step agent against tasks, score the final answer (e.g., bitbucket-pr-splitter)
3. **Trajectory agent eval** — run a multi-step agent, score intermediate steps (tool calls, reasoning)
4. **A/B comparison** — two Runs sharing a Dataset, diff via computed `POST /runs/compare?a=X&b=Y` endpoint
5. **Regression / CI** — Trigger fires nightly; diff against baseline; fail if metric drops past threshold
6. **Cross-judge agreement** — multiple Runs of same Dataset with different judge models; report Spearman/agreement
7. **Adversarial eval** — Datasets with adversarial inputs; standard scoring
8. **Cost/latency/safety as scored dimensions** — operational metrics captured per-Run alongside semantic scores

### v1 use cases

9. **Online / production trace eval** — continuous trace ingestion + async scoring
10. **Human-rater / labeling pipeline** — multi-rater Annotation workflow
11. **RL / preference data collection** — pairwise preferences, reward modeling data export
12. **Side-effect / sandbox eval** — agents that touch real systems (mocked)

### First user

The judge_optimization rebuild in `traitinterp/experiments/judge_optimization/` is the first concrete user. Workflow:
- Hand-label 80-120 responses on 4 sub-dimensions (answers_question, repetition, hallucination, on_topic) — done by Claude Code
- Register Dataset
- Iterate Prompt versions for the coherence judge
- Run Tasks against the golden Dataset; compare per-Spearman + sub-dimension breakdown
- Replace existing AI-written `docs/viz_findings/llm-judge-optimization.md` with rigorous, harness-backed content

---

## 4. Stack & hosting

### Locked technology choices

- **Language: Python 3.12+ only** (no TypeScript backend in v0). Reasons: numerical/statistical core (Spearman, Cohen's d, agreement metrics) lives well in scipy; ecosystem alignment with Inspect/Phoenix/lm-eval; solo ownership means single-language is much faster.
- **Web framework: FastAPI** (uvicorn behind it)
- **ORM: SQLAlchemy 2.x async** with Alembic migrations
- **Validation: Pydantic 2.x** (single source of truth for API request/response models)
- **Database: PostgreSQL** on Neon (free tier initially)
- **Blob storage: Cloudflare R2** (already in use for traitinterp; reuse account)
- **Hosting: Railway Pro Plan** ($20/mo + usage credits; 24 vCPU / 24GB RAM available per replica)
- **Dashboard: server-rendered with Jinja templates + htmx + Tailwind CSS + Alpine.js** (NO Next.js in v0; can swap later via API-first design)
- **MCP server: `fastapi-mcp` mounted alongside FastAPI** at `/mcp` (auto-generated; curation deferred)

### API-first separation (architectural commitment)

This is a load-bearing decision. **Jinja routes call the service layer directly (Python function calls), NOT internal HTTP roundtrips.**

```
scryer/server/
  api/                  # JSON endpoints: GET /api/v1/runs, POST /api/v1/datasets, etc.
    runs.py
    datasets.py
    ...
  templates/            # Jinja templates rendering data from service layer
  routes_web.py         # Web routes that call service functions, hand results to templates
  services/             # Pure Python service functions called by both api/ and routes_web.py
```

This means: the same JSON API serves the CLI, Python SDK, htmx dashboard (via routes_web.py rendering HTML), and any future Next.js dashboard. Swapping to Next.js later = delete `templates/` + `routes_web.py`, deploy Next.js calling `/api/v1/*`. Backend untouched.

### Cost & sleep

- Railway Pro: ~$20-40/month with normal use
- Neon Postgres: free tier likely covers v0 (500MB / 0.5GB compute)
- R2: ~$0.015/GB/month for storage (negligible at v0 scale)

Railway Serverless mode + keep-alive pattern:
- FastAPI configured for Serverless (sleeps after 10 min idle)
- `Workspace.active_run_count` column tracks in-flight Runs per Workspace
- Railway Cron pings `/internal/keepalive` every 5 min IF `SUM(active_run_count) > 0`
- When idle, FastAPI sleeps; cron-triggered Run wakes it on demand

### Hosting structure

- One FastAPI service on Railway (handles HTTP + executes Scorers via subprocess)
- Postgres on Neon (separate)
- R2 for Trace blobs and large artifacts
- No separate worker service in v0 (background tasks fold into FastAPI process; revisit when scale demands)

---

## 5. Multi-user & sharing model

### Hierarchy

Two-level: **Workspace > Project**.

- **Workspace** = billing/team boundary. Has slug (URL-safe globally unique), name, owner.
- **Project** = work-unit, scoped under a Workspace. Has slug (unique per Workspace), name, visibility setting.
- All other resources are scoped under a Project (transitively under Workspace).

### Auto-creation on signup

When a new user accepts an Invitation, the system auto-creates:
- A `personal` Workspace owned by the new user
- A `default` Project in that Workspace

User can rename either later.

### Slug history

Slugs are mutable; old slugs permanently redirect (GitHub model). Implementation: separate `workspace_slugs` and `project_slugs` tables that map old_slug → resource_id forever within their namespace. New Workspaces/Projects can never claim a previously-used slug in the same namespace.

### Project visibility (Option C — hybrid)

Each Project has `visibility: 'private' | 'workspace'`:
- `workspace` (default): all Workspace members see this Project per their Workspace role
- `private`: only members explicitly added to the Project via `project_members` table

### Per-resource share grants

Orthogonal to Project visibility. A `share_grants` table allows individual resources (Run, Dataset, Trace, Comment, Collection) to be shared with specific users outside Project membership:
- `(resource_type, resource_id, grantee_user_id, permission)`
- Permission is `read` or `write`
- Useful for "share this one Run with my coworker" without inviting them to the Project

### No public unauthenticated share links in v0

Every resource read requires authentication. Public share-token URLs deferred to v1 if needed.

### Roles

Three roles at both Workspace and Project level: `viewer`, `member`, `owner`.

- `viewer` — read-only access
- `member` — can create/modify resources, run Tasks, write Comments
- `owner` — viewer + member powers + can invite users, change visibility, archive resources

### Multi-user shape

**Invite-only signup.** No public signup endpoint in v0. Existing users (with `admin` scope on their ApiKey OR Workspace `owner` role) can generate Invitations. Invitations are first-class (see §10).

---

## 6. Auth model

### Two principal types

- **User** — humans with email + password (argon2id hash) + interactive login flow
- **ServiceAccount** — bot identities (no email, no interactive login). Has `name`, `owner_user_id` (nullable: null = autonomous, set = "acting on behalf of"), `is_active`, `requires_approval`.

### ApiKey — the universal credential

Both Users and ServiceAccounts authenticate programmatically via ApiKey:

- `principal_type: 'user' | 'service_account'`
- `principal_id: UUID` (FK to User or ServiceAccount per principal_type)
- `key_prefix` (display, e.g., `scrk_live_a8f3e2`)
- `key_hash` (sha256 hex of full key — input is 256-bit-entropy random, so a fast hash is sufficient; bcrypt/argon2 here adds ~50ms per request for no security gain. Matches Stripe/GitHub/crates.io. Same goes for `Invitation.token_hash`.)
- `scopes: TEXT[]` — the 4-scope vocabulary (see below)
- `expires_at` (nullable; null = never expires)
- `allowed_ips` (nullable INET array; null = any IP)
- `last_used_at`, `revoked_at`

### Scopes (4 total)

Reduced from an earlier 8-scope GitHub-PAT-style vocabulary; lean v0 stance.

- `read` — all GET endpoints
- `write` — POST/PUT/PATCH on resources the principal can access
- `admin` — manage Workspace settings, invite users, modify Credentials, archive resources
- `impersonate` — explicitly carries `on_behalf_of_user_id` semantics

For ServiceAccounts: by default, never granted `admin`. The `approvals:grant` capability (which would mark Approval rows as approved) is deferred to v1 with the Approval noun.

### JWT for human session auth

Web dashboard login → bcrypt password verify → issue short-lived JWT (1 hour) + refresh token. JWT goes in `Authorization: Bearer <jwt>` header.

For programmatic clients, ApiKey goes in `Authorization: Bearer scrk_live_<...>`. Server distinguishes by prefix (JWTs are typically dot-separated; ApiKeys start with `scrk_live_`).

### Credential — outbound LLM keys

Distinct from ApiKey. Stores the user's external API keys for LLM providers (OpenAI, Anthropic, OpenRouter, custom OpenAI-compatible endpoints):

- `provider: 'openai' | 'anthropic' | 'openrouter' | 'custom'`
- `encrypted_value: TEXT` (AES-GCM encrypted; encryption key in server env, versioned for rotation)
- `metadata_json: JSONB` (endpoint URL, org_id, etc.)
- `restricted_to_user_ids: UUID[]` (nullable; null = anyone in workspace can use)

The encryption key for Credentials lives in the server's env vars, NOT in any user-accessible place. Subprocess running user code receives the decrypted Credential as an env var (e.g., `OPENAI_API_KEY=...`); the encryption key is never exposed to subprocess.

### Invitation — signup tokens

First-class noun. Existing user with appropriate scope generates an Invitation:
- `token_hash` (bcrypt of the actual token, which is shown ONCE to inviter to share)
- `email` (optional pre-fill)
- `workspace_id` + `workspace_role` + optional `project_grants` JSONB
- `expires_at`, `used_at`, `used_by_user_id`, `revoked_at`
- `invited_by` (FK to User)

Lifecycle: created → used | expired | revoked. Audit trail preserved forever.

### `on_behalf_of_user_id` semantics

When a ServiceAccount has `owner_user_id` set, it can carry "acting on behalf of" semantics. Every action the ServiceAccount takes records:
- `actor_id` = ServiceAccount.id (what mechanically did the thing)
- `actor_type` = 'service_account'
- `on_behalf_of_user_id` = the human who's accountable (set automatically from ServiceAccount.owner_user_id if present)

This is the IETF RFC 8693 (token exchange) pattern. Audit can answer both "what bot did this" and "which human authorized it."

### Validation: `on_behalf_of` cannot be caller-supplied (v1 if needed)

In v0, the `on_behalf_of_user_id` is set by the server from `ServiceAccount.owner_user_id` only — never accepted from the request. If/when v1 adds explicit delegation grants (User X grants ServiceAccount Y permission to act on their behalf), an explicit `service_account_delegations` table will validate.

---

## 7. Execution & sandbox model

### Server-side execution from day 1

User-supplied Scorer/Agent/Tool code runs in a Python subprocess on the server. Source code is stored in Postgres TEXT fields (locked in §6 of the noun model). Server reads source from DB, writes to a temp file, executes via subprocess.

### Sandbox: Tier 1+ (subprocess + stripped env + ulimits + timeout)

This is what's possible on Railway. Railway blocks user namespaces (confirmed via empirical test on the user's `trait-interp-viz` Railway service: `unshare: Permission denied`). So nsjail/bubblewrap don't work. Real Docker-per-run is too slow on Railway (DinD adds 1-3s per spawn).

What v0 sandbox does:
- Subprocess spawned with `env={}` plus only the user-authorized credentials (LLM API keys decrypted from Credentials at run time)
- `cwd=/tmp/run_<uuid>/` (not server's working directory)
- `resource.setrlimit(RLIMIT_AS, max_memory)` — memory cap
- `resource.setrlimit(RLIMIT_CPU, max_cpu_seconds)` — CPU cap
- `asyncio.wait_for(..., timeout=300)` — 5-min hard wall (Scorers should be < 1 min typically)
- Subprocess inputs piped via stdin as JSON; outputs returned via stdout as JSON
- Subprocess process killed via SIGTERM → wait → SIGKILL after grace period

### What sandbox does NOT protect against

- Network egress to arbitrary URLs (subprocess can `urllib.urlopen()` to attacker.com)
- Reading host system files (`/etc/passwd` etc. — though no secrets there)
- Side-channel attacks
- Kernel exploits

These are accepted risks for invite-only beta. When the system opens to public signup, migrate execution to Modal or E2B (real isolation) — see §16 for the deferred sandbox upgrade.

### Defenses layered on top

- **Mandatory credential-pattern redaction** on all write paths (Comment body, Trace step content, AuditEvent metadata, error responses): regex for `sk-*`, `Bearer *`, hex-40, plus high-entropy detection for any token-like string `[A-Za-z0-9_-]{32,}`. Match → redact; log a security AuditEvent on redaction firing.
- **Hook URL SSRF denylist**: validate all outbound webhook URLs against RFC 5735 private ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x cloud metadata, ::1, fc00::/7) at creation time AND DNS re-resolution at fire time (DNS rebinding defense).
- **Trigger rate caps**: max 20 active Triggers per Workspace; minimum 5-minute cron interval; `triggers:write` scope only granted to humans by default.
- **Per-ServiceAccount spend governor with circuit breaker**: see §6 of cheap fixes (§15) and Budget noun (§10).
- **Pre-baked Docker image** with ~25 standard ML/LLM packages (numpy, scipy, pandas, scikit-learn, openai, anthropic, openrouter, httpx, pydantic, tiktoken, etc.). Users can NOT add custom dependencies in v0 — known limitation; mitigated by the package list being broad. Custom-deps-per-Scorer deferred to v1.

### Pre-baked image package list (v0 minimum)

- Standard data: `numpy`, `scipy`, `pandas`, `polars`, `scikit-learn`, `statsmodels`
- LLM clients: `openai`, `anthropic`, `httpx`, `aiohttp`, `tiktoken`
- Validation: `pydantic`, `pydantic-settings`, `python-dotenv`, `pyyaml`
- Utility: `tenacity` (retries), `structlog`, `rich`
- Image total: ~800MB-1.2GB

Tradeoff: agents can use these libraries; specialized libraries (custom HuggingFace models, internal company SDKs) are deferred to v1's per-Scorer dependency story.

### Heartbeat + stale-run reaper

Critical for resilience under Railway Serverless (FastAPI sleeps after 10 min idle; in-flight tasks die):

- `Run.last_heartbeat_at` updated every 30 seconds by the executing background task
- `Run.resume_cursor` records last successfully completed `record_id`
- Railway Cron hits `/internal/reap-stale-runs` every 5 min: any Run in `running` state with `last_heartbeat_at > 2 * heartbeat_interval` transitions to `failed` with `failure_reason="heartbeat_timeout"`
- `scryer run resume <run-id>` re-runs Agent only on the unfinished records (using `resume_cursor`)

### Active-run keep-alive

- `Workspace.active_run_count INTEGER NOT NULL DEFAULT 0` — incremented on Run start, decremented on Run end
- Railway Cron pings `/internal/keepalive` every 5 min IF `SUM(active_run_count) > 0` across Workspaces
- When all Workspaces idle, FastAPI is allowed to sleep (cost saving)
- Periodic reconciliation job compares the counter against `SELECT COUNT(*) FROM runs WHERE status='running'` to fix counter drift from crashes

---

## 8. Audit model

### Three layers

scryer captures historical data at three layers, each optimized for a different purpose:

**Layer 1: Per-resource version history** (already in the noun model)
- Datasets, Scorers, Agents, Prompts, Tasks, Tools have `version` + `content_hash` + `parent_id`
- Comments have `comment_versions` table for edit history
- Run, Result, Trace are immutable artifacts

**Layer 2: Global audit log** (the AuditEvent noun)
- Append-only `audit_events` table
- Captures every state mutation system-wide
- Indexed for forensic queries (by actor, by resource, by request, over time)

**Layer 3: Auto-Comments** (kind='system' on Comment)
- High-signal narrative subset of audit events surfaced as user-readable Comments
- A subset of audit events also write a Comment with `kind='system'`
- For: `run.completed` (with summary), `run.failed`, `suite.regression_detected`, `trigger.fired`
- NOT for: every `resource.created` / `resource.updated` (would be too noisy)
- Per-Project disable flag: `projects.auto_comments BOOLEAN DEFAULT true`

### AuditEvent schema (key fields)

- `workspace_id`, `project_id` (nullable for cross-scope events)
- `actor_user_id` OR `actor_service_account_id` (XOR; CHECK constraint)
- `actor_type: 'user' | 'service_account' | 'cron' | 'system'`
- `on_behalf_of_user_id` (nullable; delegation chain)
- `action: TEXT` (e.g., `'dataset.create'`, `'permission.grant'`, `'login'`, `'verification'`)
- `resource_type`, `resource_id` (nullable for non-resource events)
- `before_json`, `after_json` (full state snapshots)
- `reason: TEXT` (commit-message-flavored, optional in v0; required for destructive ops)
- `ip_address`, `user_agent`
- `request_id: UUID` (correlates events from same API call)
- `parent_event_id: BIGINT` (when one event causes another)
- `timestamp: TIMESTAMPTZ`
- `metadata: JSONB`
- BIGSERIAL primary key (cheap monotonic; no UUID overhead for log table)

### Capture mechanism

Middleware-based (Option B from earlier discussion):
- Decorator wraps mutating endpoints; captures actor + request_id + before/after diffs automatically
- Manual instrumentation for the most sensitive 5 actions (deletes, permission grants, credential rotations) for explicit reason capture

Database triggers (Option C) are deferred to v1 for tamper-resistance.

### Reason field

Required for destructive actions (delete, archive, permission revoke, credential rotate). Optional otherwise.

CLI: `-m` flag (commit-message-style)
- `scryer scorer push coh_judge.py -m "added explicit instruction for repetition handling"`
- `scryer dataset archive old-golden -m "superseded by golden-v4 with sub-dim labels"`

API: `X-Reason` header or `reason` field in JSON body

UI: prompts for reason on destructive actions

### Soft delete only

- Every resource has `archived_at TIMESTAMPTZ` (nullable)
- `archived_at IS NULL` = active; non-null = archived
- All read queries default to `WHERE archived_at IS NULL`
- Helper functions enforce this; lint rule should catch raw queries that bypass
- Hard delete is a separate admin-only action with explicit AuditEvent (deferred to v1 with RetentionPolicy)

### PII / secret redaction

Per-resource-type redaction filter strips sensitive fields from `before_json` / `after_json` before persisting:
- Credential.encrypted_value: redacted
- Any field matching credential patterns: redacted
- ApiKey.key_hash: redacted (keep prefix)

### Hash chain — DEFERRED to v1

Originally locked, then reconsidered. In Postgres without external anchor (Sigstore, blockchain, transparency log), hash chain is mostly theater — anyone with DB write access can re-chain. v0 relies on Postgres WAL + Neon's continuous backups. v1 adds hash chain when there's a compliance need + external anchoring story.

### Activity feed widget

Dashboard ships a "Recent Activity" widget on the home page rendering the last 50 AuditEvents in the user's accessible scope. Replaces the deferred `agent_feed` unified inbox endpoint for v0 humans.

---

## 9. Versioning model

### Content-hashed everywhere

Every versioned resource (Dataset, Scorer, Agent, Prompt, Tool, Task) has:
- `version: INTEGER` — auto-incremented per `(project_id, slug)`
- `content_hash: TEXT` — sha256 of canonicalized content
- `parent_id: UUID` — FK to previous version (nullable for v1)

Two versions of the same logical resource are two separate rows in the table, linked via `parent_id`.

### Immutable resource versions

A version is immutable once written. Editing creates a new version with new `version`, new `content_hash`, and `parent_id` pointing at the previous.

### Adding rows to a Dataset = new version

Datasets are immutable; adding rows creates Dataset version N+1 with parent pointer to version N. Old version still exists; old Runs reference it by hash.

### Schema evolution (Datasets)

Schema fields declare `required: bool` and `nullable: bool` independently:
- Adding an `optional` field to schema → no back-fill needed; existing Records stay as-is
- Adding a `required` field → all existing Records must be back-filled before bumping version

### Tasks reference specific versions

`Task.dataset_id` + the resolved `dataset.version` are pinned at Task definition time. A Run captures both. Re-running a Task uses the pinned version.

### Lineage queries

- "Show all versions of Dataset X" → recursive CTE on `parent_id`
- "What changed between Dataset@v2 and v3" → diff `before_json` / `after_json` from AuditEvents OR fetch both versions and diff `records`
- No standalone Version noun in v0 (deferred — `parent_id` + `version` columns are enough)

### DatasetVersion.source_results[]

When an agent curates a Dataset by adding rows from failed Results, the `source_results: UUID[]` field links new rows back to motivating Results. Provenance chain: "this Record was added because Result #abc failed."

---

## 10. First-class noun set

25 first-class nouns + several sub-resources. Categorized below for learnability.

### Org & Auth (8)

| # | Noun | Purpose |
|---|---|---|
| 1 | **Workspace** | Billing/team boundary; root of resource hierarchy |
| 2 | **Project** | Work-unit; scoped under Workspace; has visibility setting |
| 3 | **User** | Human identity (email, password) |
| 4 | **ServiceAccount** | Bot identity (no email; can have owner_user_id for delegation) |
| 5 | **Credential** | Encrypted external API keys (LLM providers); workspace-scoped |
| 6 | **ApiKey** | Programmatic auth credential; supports both User and ServiceAccount principals |
| 7 | **Invitation** | Signup token; lifecycle pending → used \| expired \| revoked |
| 8 | **Budget** | Spend limit per principal per period; checked by circuit breaker before LLM calls |

### Eval Inputs (5)

| # | Noun | Purpose |
|---|---|---|
| 9 | **Dataset** | Polymorphic Records (id, inputs, expected, metadata); immutable; versioned; optional declared schema |
| 10 | **Scorer** | Per-record scoring function; source code stored in Postgres; versioned |
| 11 | **Prompt** | Versioned prompt template; reusable across Scorers/Agents |
| 12 | **Suite** | Named collection of Tasks with aggregate metrics |
| 13 | **Tag** | Workspace-scoped, orthogonal organization; cross-Project within Workspace |

### Agent Setup (2)

| # | Noun | Purpose |
|---|---|---|
| 14 | **Agent** | The thing being evaluated (multi-step or single-shot LLM pipeline); optional on Task |
| 15 | **Tool** | Registered tool definitions agents can call; cross-Agent reuse |

### Execution & Output (4)

| # | Noun | Purpose |
|---|---|---|
| 16 | **Task** | Named binding of (Dataset, Scorer, Agent\|None, Prompt\|None); content-hashed; versioned |
| 17 | **Run** | One execution of a Task version; status enum (queued/running/done/failed/cancelled/superseded) |
| 18 | **Result** | Per-record score within a Run; references Trace; has invalidated_at "void" pattern |
| 19 | **Trace** | Agent's execution capture per Record; enables rescore/resume; sub-resource: TraceStep |

### Automation (2)

| # | Noun | Purpose |
|---|---|---|
| 20 | **Trigger** | Schedule (cron) OR webhook event firing Tasks/Suites; missed-fire policy enum |
| 21 | **Webhook** | Outbound webhook fired on scryer events; renamed from "Hook"; sub-resource: WebhookDelivery |

### Annotation & Audit (3)

| # | Noun | Purpose |
|---|---|---|
| 22 | **Comment** | Flat (no threading except `reply_to_comment_id`), timestamped, project-scoped; with author + structured{tldr, confidence} + references[] + mentions[] + kind enum |
| 23 | **Collection** | Curated bundle of pointers to scryer resources; has `purpose` enum + members[].group/note |
| 24 | **AuditEvent** | Append-only audit log; renamed from "Action"; bigserial PK |

### Operations (1)

| # | Noun | Purpose |
|---|---|---|
| 25 | **UsageRecord** | Per-LLM-call entries; tokens, cost, latency, FK to Credential |

### Sub-resources (not first-class)

- **DatasetRecord** — rows inside a Dataset (composite PK with dataset_id)
- **TraceStep** — steps inside a Trace (normalized for `tool_id` queries)
- **SuiteRun** — execution of a Suite (groups multiple Run rows via junction)
- **WebhookDelivery** — retry attempts for outbound webhook fires
- **CommentVersion** — edit history for Comments
- **ResourceTag** — many-to-many junction for tag application
- **WorkspaceMember**, **ProjectMember** — membership tables
- **WorkspaceSlugs**, **ProjectSlugs** — slug history for permanent redirect
- **AgentTool** — junction for Agent-Tool composition
- **SuiteTask** — junction for Suite-Task composition
- **SuiteRunRuns** — junction for SuiteRun-Run aggregation
- **ShareGrant** — per-resource sharing grants
- **ApiKeyUsage** — lightweight forensics on key usage (separate from AuditEvent)

### Why these specific 25

Validated against three concrete use cases (judge_optimization, multi-step agent eval, autonomous agent loop) by spawned investigation agents. Each noun was checked for: USED HEAVILY / USED LIGHTLY / UNUSED in each use case. Survived the cut.

Critically reviewed by spawned critic agents. Several proposed defers were accepted (see §16); several were rejected with reasoning (e.g., ServiceAccount kept despite critic's defer recommendation because the autonomous-loop use case explicitly justified it).

---

## 11. Schema overview (cluster-level, not full SQL)

### Cluster 1: Identity + Org + Auth

- `users` — humans with email/password
- `workspaces` — billing/team boundary; has `active_run_count` for keep-alive
- `workspace_slugs`, `project_slugs` — slug history tables
- `projects` — scoped under workspace; `visibility` enum
- `workspace_members`, `project_members` — membership with role
- `service_accounts` — bot identities; `owner_user_id` nullable; `requires_approval` flag
- `api_keys` — `principal_type` + `principal_id` polymorphic; bcrypt key_hash
- `api_key_usage` — lightweight per-request forensics; bigserial PK
- `invitations` — token-based signup; bcrypt token_hash
- `credentials` — AES-GCM encrypted; versioned encryption key
- `share_grants` — per-resource cross-Project sharing
- `budgets` — spend limits per (principal, period)

Polymorphic FK pattern: two nullable columns (e.g., `actor_user_id` + `actor_service_account_id`) + CHECK constraint that exactly one is set. Used for any case where the FK target could be either User or ServiceAccount.

### Cluster 2: Evaluation Core

- `datasets` — versioned; content-hashed; `parent_id` lineage; optional `schema_json`; `source_results[]`
- `dataset_records` — composite PK (dataset_id, record_id); polymorphic JSONB inputs/expected/metadata
- `scorers` — source_text in Postgres; content-hashed; versioned; `server_executable BOOL`
- `agents` — source_text in Postgres; content-hashed; versioned; `config_json`
- `tools` — source_text in Postgres; content-hashed; versioned; `schema_json`; `sandbox_required BOOL`
- `agent_tools` — junction (agent_id, tool_id)
- `prompts` — template + `template_format` enum
- `tasks` — bind dataset_id + scorer_id + agent_id (nullable) + prompt_id (nullable); `estimated_cost_usd`
- `runs` — task_id + task_version pinned at queue time; status enum; `last_heartbeat_at`; `resume_cursor`; `invalidated_at`
- `results` — composite (run_id, record_id); denormalized `score_value` + structured `score_json`
- `traces` — per-Record execution capture; `storage_uri` to R2 for large blobs; `storage_sha256` integrity check
- `trace_steps` — normalized table with `tool_id FK` (enables cross-Trace tool-usage queries)

Versioning pattern: every versioned resource uses `(project_id, slug, version)` UNIQUE + `parent_id` for lineage chain.

### Cluster 3: Activity + Audit + Governance + Automation

- `audit_events` — append-only; bigserial PK; multi-index (workspace+time, resource, actor, request_id)
- `suites` — named collection of Tasks
- `suite_tasks` — junction with `position` and `weight` for aggregation
- `suite_runs` — group multiple Run rows
- `suite_run_runs` — junction
- `triggers` — `trigger_type: 'schedule' | 'webhook'`; `target_type: 'task' | 'suite'`; `missed_fire_policy` enum
- `webhooks` — outbound; `event_types[]`; `secret` for HMAC
- `webhook_deliveries` — retry table with exponential backoff + dead-letter
- `tags` — workspace-scoped
- `resource_tags` — many-to-many
- `usage_records` — per-LLM-call; bigserial PK; FKs to run/result/trace/trace_step/credential

### Cluster 4: Collaboration

- `comments` — flat; `references JSONB`; `kind` enum; `idempotency_key` REMOVED (use middleware); `author` jsonb {type, id, agent_run_id}
- `comment_versions` — full edit history
- `collections` — curated bundles
- `collection_members` — junction (collection_id, member_type, member_id) with `position` + `note` + `group`

### Indexes worth flagging

- `runs (status, queued_at) WHERE status IN ('queued','running')` — for job scheduling
- `runs (last_heartbeat_at) WHERE status = 'running'` — for stale-run reaper
- `audit_events (workspace_id, timestamp DESC)` — for activity feed
- `audit_events (resource_type, resource_id, timestamp DESC)` — for "history of this resource"
- `usage_records (workspace_id, timestamp DESC)` — for spend dashboards
- Per-resource: `(project_id, slug, version)` UNIQUE for all versioned resources

### Migrations

- Use Alembic, additive-only changes for v0
- Maintenance mode flag (`maintenance_mode BOOLEAN` in a `system_state` table) — when true, new Run queueing is rejected (returns 503); existing Runs continue to completion
- For `ALTER TABLE ADD COLUMN`: use `DEFAULT NULL` (no lock on PG 11+); back-fill in batches if needed

---

## 12. API conventions

### URL structure

Hybrid: nested URLs for list endpoints; flat URLs for direct resource addressing.

- **Nested (list endpoints)**: `GET /api/v1/workspaces/{ws}/projects/{p}/runs`
- **Flat (resource endpoints)**: `GET /api/v1/runs/{run_id}` — Run IDs encode workspace/project membership server-side

This matches GitHub's pattern. Cleaner CLI/SDK ergonomics; auth still enforced at the resource level.

### Pagination — cursor-based universally

```
GET /runs?limit=100&after=run_abc123
→ { "data": [...], "has_more": true, "next_cursor": "<opaque>" }
```

Opaque base64-encoded cursors (NOT exposing primary keys). No offset pagination anywhere. Stable under concurrent writes. Matches OpenAI Assistants API + GitHub.

### Response envelope

- **List endpoints**: `{ data: [...], has_more: bool, next_cursor: str }`
- **Single-resource endpoints**: naked object `{ id: ..., status: ..., ... }`

GitHub/Stripe pattern. Consistency is the rule — every list looks the same; every single-resource looks the same.

### Errors — RFC 9457 with `retryable` extension

```json
{
  "type": "https://scryer.io/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "scorer_id references a Scorer that does not exist",
  "instance": "/api/v1/runs/abc123",
  "retryable": false,
  "retry_after": null,
  "problem_field": "scorer_id",
  "suggestion": "Call GET /scorers to list valid scorer IDs"
}
```

Drop-in `fastapi-problem-details` library. The `retryable: bool` field is critical for agent clients (Anthropic SDK has known issues from missing it).

Each error `type` URI resolves to a JSON document explaining recovery (also serves an HTML version for humans via content negotiation).

### Idempotency keys (Stripe pattern)

Middleware on all POST/DELETE endpoints. Header: `Idempotency-Key: <UUID>`.

- Cache key scoped to `(workspace_id, principal_id, key)` — prevents poisoning
- Cache stored in Postgres `idempotency_cache` table with 24h TTL
- Server auto-generates key if not provided; returned in response header
- Replay returns cached response only if request body fingerprint matches; else 422

### OpenAPI metadata discipline

- Every endpoint has explicit `operation_id`, `summary`, `description`
- Every Pydantic field has `description=` and `example=`
- This is what powers MCP tool generation and agent comprehension
- Lint rule enforces (CI fails on missing description)

### Versioning headers

Every response includes:
```
Scryer-Version: 2026-05-02
Sunset: 2027-05-02
Link: <https://scryer.io/docs/migrations/2026-05-02>; rel="deprecation"
```

Dated API versions; `Sunset` (RFC 8594) machine-readable for agent clients.

### MCP server

- `fastapi-mcp` mounted at `/mcp` (auto-generated from OpenAPI)
- v0 ships `/mcp` only (full surface)
- Curated `/mcp/curated` with hand-written 8-tool subset DEFERRED — observe usage first, curate later
- Tool descriptions hardcoded in source (NOT DB-configurable; prevents injection vector)

### Discovery: `GET /api/v1/me`

Every agent's first call. Returns:
```json
{
  "identity": { "type": "service_account", "id": "sa_...", "name": "..." },
  "workspace": { "id": "...", "slug": "..." },
  "permissions": ["read", "write"],
  "top_resources": {
    "projects": [...top 5 by recency...],
    "datasets": [...top 5 by recency...],
    "scorers": [...top 5 by recency...]
  },
  "api_version": "2026-05-02",
  "docs": "https://scryer.io/docs/agent-guide"
}
```

Eliminates bootstrap roundtrips for agents. Cap each `top_resources` array at 5.

### Polling vs webhook vs streaming

- **Polling**: `GET /runs/{id}` always works; recommended exponential backoff (1s, 2s, 4s, 8s, cap 30s)
- **Webhooks (outbound)**: Webhook noun fires on events
- **SSE streaming**: `GET /runs/{id}/events` DEFERRED to v1

---

## 13. Naming

### Project name: `scryer`

Locked. PyPI: `scryer`. GitHub: `github.com/ewernn/scryer`. CLI: `scryer`. Python package: `import scryer`.

Note collision with Scryer Prolog (active project). Scryer is second-most-prominent thing called scryer. People who care about Prolog know the difference; people who care about ours find it via your work.

### Renames from earlier locks

| Old name | New name | Reason |
|---|---|---|
| Hook | **Webhook** | Universal convention; "hook" doesn't telegraph outbound |
| Action | **AuditEvent** | "Action" overloaded in API context (could mean "thing user can do") |
| `scryer log` | **`scryer audit list`** | Consistency with noun-first CLI pattern |

### CLI command structure

Noun-first, kebab-case, singular noun:
- `scryer dataset push`
- `scryer scorer push`
- `scryer task run`
- `scryer run start`
- `scryer audit list`
- `scryer credential add`
- `scryer service-account create`

Matches `kubectl pod get`, `gh repo clone`, `aws s3 ls`, `gcloud compute instances list`.

### Pluralization

- API endpoints: plural (`/datasets`, `/runs`)
- CLI commands: singular (`scryer dataset`, `scryer run`)

Standard practice across kubectl/gh/aws/gcloud. Not an inconsistency.

### Naming non-changes (decided to keep)

- **Trace + TraceStep** kept (TraceStep ≈ OTel Span). Don't rename Trace → Span; the existing semantics of Trace = "execution capture per Record" is fine.
- **Result** kept as first-class (sub-resource of Run, but addressable by ID).
- **Tool** kept (industry standard; OpenAI/Anthropic SDKs use it). Distinguish from "MCP tools (scryer's API surface)" in docs only.
- **Prompt** kept (capital-P versioned template vs lowercase prompt-text; readers handle context).
- **Suite** kept (mild overload with "test suite" but acceptable).
- **Agent + ServiceAccount** both kept despite semantic adjacency. Different domains: Agent = thing under test; ServiceAccount = bot identity calling scryer's API. Docs put them in different sections (Eval Setup vs Auth).

### Doc categorization

Use the §10 categories: Org+Auth / Eval Inputs / Agent Setup / Execution+Output / Automation / Annotation+Audit / Operations. Both dashboard navigation and docs site should organize by these categories.

---

## 14. Documentation strategy for agent users

### V0 minimum viable

1. **`GET /api/v1/me`** returning identity + permissions + top 5 resources per type
2. **OpenAPI spec** with non-empty `description` on every field; `example` value where possible
3. **MCP `server_info.instructions`** — 200-token onboarding block at connect time:
   ```
   Scryer evaluates LLM outputs and agents. Core workflow: register a Dataset, write a 
   Scorer, bind them in a Task, execute a Run. Get results via get_results.
   Workspace: <user's workspace>. Available resources via list_datasets / list_scorers.
   For multi-step agent eval, register an Agent; for prompt iteration, version Prompts.
   See https://scryer.io/docs/agent-guide for cookbook.
   ```
4. **MCP tool descriptions** in `[verb] [noun] [constraint] [anti-pattern]` template:
   ```
   create_run — Start an evaluation run against a dataset.
     dataset_id: use list_datasets to find valid IDs. scorer_id: must exist in this workspace.
     Returns run_id immediately; run is async. Poll get_run for status.
     Do not call twice for the same (dataset, scorer) — use list_runs first.
   ```
5. **RFC 9457 errors** with `type` URIs that resolve to structured JSON recovery docs
6. **`Scryer-Version` + `Sunset` headers** on all responses

### V1 documentation additions

- MCP cookbook resources at `scryer://guides/<name>` (e.g., `iterate-judge-prompt`, `regression-detection-setup`)
- `list_guides` MCP tool returning available recipes with one-line summaries
- Per-noun fixture files referenced from OpenAPI spec via `x-scryer-fixture`
- Versioned doc URLs with migration guides

### Inline OpenAPI vs separate docs site

**Rule**: everything an agent needs at call-time lives in the OpenAPI spec. Everything a human needs to understand the system philosophically lives in the docs site.

- `dataset_id` field description in OpenAPI: format + valid values + example value + how to obtain
- Docs site explains what a Dataset is conceptually + how to design one for evaluation
- Agents read OpenAPI; humans read docs site

### Tier 3 nouns (admin warnings for agents)

OpenAPI descriptions for Credential, ApiKey, Invitation, ServiceAccount, UsageRecord lead with: **"Admin operation. Confirm with workspace owner before calling."** Discourages agents from accidentally rotating credentials without human approval.

---

## 15. Cheap fixes & adjustments accepted

These ~22 items were surfaced by critic/security/failure-mode agents and accepted with minor adjustments. Applied to the design.

### Concurrency & data integrity

1. **Postgres transactions on multi-table state transitions** — always wrap related writes in single transaction
2. **CAS atomic operations** — Idempotency via `INSERT ON CONFLICT DO NOTHING RETURNING`; claim acquisition via single atomic `UPDATE ... WHERE` with conflict check (when claims land in v1)
3. **Heartbeat + stale-run reaper** — `Run.last_heartbeat_at` updated every 30s; reaper Cron transitions stale Runs to `failed`
4. **Run.resume_cursor** — last completed `record_id` for resumability after partial failure
5. **Active-run keep-alive** — `Workspace.active_run_count`; Cron pings FastAPI when > 0
6. **Periodic counter reconciliation** — compare `active_run_count` against actual COUNT to fix drift

### Security

7. **Credential-pattern redaction on all write paths** — regex for known patterns + high-entropy detection for `[A-Za-z0-9_-]{32,}`. Logs a security AuditEvent on redaction firing.
8. **Idempotency keys scoped to `(workspace_id, principal_id, key)`** — prevents cross-user cache poisoning
9. **Hook URL SSRF denylist + DNS re-resolution** — RFC 5735 private ranges blocked; DNS resolved at fire time (rebinding defense)
10. **Trigger rate caps** — max 20 active per Workspace, 5-min minimum cron interval, `triggers:write` scope only granted to humans by default
11. **Missed-fire policy on Trigger** — `skip_to_latest` default
12. **MCP tool descriptions hardcoded in source** — never DB-configurable (prevents injection vector)
13. **Per-ServiceAccount spend governor with circuit breaker** — pre-LLM-call check against Budget; terminate Run with `failure_reason='spend_limit_exceeded'`
14. **`approvals:grant` capability** locked away from default ServiceAccount keys (relevant when Approval lands in v1)

### Reliability

15. **Webhook delivery retry table with exponential backoff** — attempts at 1s, 10s, 100s, 1000s; dead-letter status; manual retry endpoint
16. **Trace SHA-256 integrity check on read** — `traces.storage_sha256` verified when fetching from R2
17. **R2 write-before-commit ordering** — write Trace blob to R2 first; Postgres Result row only commits if R2 write succeeded; orphan blob handling via cleanup job
18. **Soft-delete with `archived_at`** on every mutable resource; helper functions enforce `WHERE archived_at IS NULL` filter
19. **Maintenance mode flag** for online migrations; pauses new Run queueing during ALTER TABLE
20. **Postgres disk monitoring** — alert at 80%; hard-stop new writes at 95% (audit log archival to R2 deferred to v1).

    **2026-05-02 deferral lock**: cold-tier archive scoped (Parquet manifests, Cron promotion, dual-tier read path). Cost analysis at 3 scales: Beta saves $0/yr, Growing saves $185/yr (5.7yr ROI), Scale-out saves $1,882/yr (20mo ROI). 9/10 STRONG critic arguments to defer. Build trigger (ANY one): (a) PG > 50 GB cumulative, (b) Neon storage line-item > $20/mo, (c) user requests > 2yr audit retention. Until then, do nothing. Side effect: surfaced web pagination + index-swap bugs as separate fixes.

### Operational

21. **`Run.invalidated_at` + `invalidation_reason`** — "void" pattern from accounting; never delete Runs, mark invalidated
22. **Comment author principal type rendered as structured metadata** — UI distinguishes `kind='user'` vs `kind='system'` vs `kind='agent'` clearly; structured rendering passed to agent context (not raw text concatenation)

### Architecture

23. **Jinja routes call service layer directly** (not internal HTTP) — saves latency and complexity; reduces complexity
24. **API key usage forensics in dedicated `api_key_usage` table** — bigserial PK, per-request log, lightweight (separate from full AuditEvent table)

---

## 16. Deferred to v1 (with reasoning)

Each item below was considered for v0 and explicitly deferred. The "trigger" column states the condition under which it'd be added.

| Item | Reason deferred | Trigger to add |
|---|---|---|
| **Online trace eval** | Use case 7; requires streaming ingestion + async scoring + alerting subsystem | When you want to monitor production agents continuously |
| **Plan noun** | Use case requires multi-agent autonomous workflows you don't have yet; idempotency_key pattern in metadata covers v0 dedup needs | When two parallel agents independently probe the same regression — that's the trigger |
| **Approval noun** | Most natural use case (autonomous loop) is itself v1; for invite-only-trusted v0 the gating pattern is "agent writes Comment requesting approval; human responds in dashboard" | When you have non-owners running agents that touch production systems |
| **Annotation (multi-rater)** | Human-rater pipeline (use case 8) deferred; v0 ingests pre-labeled JSONL | When you want multiple humans to label the same items with inter-rater agreement |
| **Comparison noun** | Computed-on-the-fly via `POST /runs/compare?a=X&b=Y` is enough for v0; first-class Comparison is over-modeling | When you want to save/share/comment on specific comparisons |
| **Sandbox upgrade (Modal/E2B)** | Tier 1+ subprocess is acceptable for invite-only-trusted; Modal/E2B adds ~$30-50/mo + integration work | When you open public signup OR have evidence of subprocess sandbox abuse |
| **ReviewQueue** | Use case 8 (human-rater) deferred | With Annotation |
| **Thread/Conversation** | For multi-Run agent conversation grouping; Tags + metadata cover v0 stopgap | With online trace eval |
| **Alert** | Online eval concern | With online trace eval |
| **Rule/Automation (LangSmith pattern)** | Online eval concern | With online trace eval |
| **Provider** | Endpoint config separated from Credential; v0 stuffs it in Credential.metadata_json | When users have multiple endpoints per provider (custom OpenAI-compatible) |
| **Hash chain on AuditEvent** | Without external anchor, it's theater; Postgres WAL + Neon backups are the actual integrity story | Compliance demands cryptographic tamper evidence |
| **`claimed_by`/`claimed_until` lock manager** | For invite-only with 3 users, contention isn't real | Multiple agents concurrently editing same resource |
| **Run.baseline_run_id** | Baseline is a comparison parameter, not a Run field; Suite-level baseline is cleaner when Suite gets one | If diff queries get slow without precomputed baselines |
| **`agent_feed` unified inbox endpoint** | Solving for hypothetical agent loops not yet in use | When you write a Claude Code session that needs it |
| **MCP curation** | Ship `/mcp/full` first; observe which 8 tools agents actually use; then curate | After 2 weeks of agent use shows hot-tool patterns |
| **Comment.claims[]** | Duplicates AuditEvent of `kind='verification'`; let claims be ad-hoc until pattern emerges | When verifiable assertions become a real workflow |
| **Comment.idempotency_key + Run.idempotency_key columns** | Stripe-style middleware idempotency is enough; per-noun keys are layered nonsense | Never (middleware handles it) |
| **Custom-deps-per-Scorer** | Pre-baked image with 25 packages covers ~90% of v0 needs | When a real user needs a niche package not in image |
| **GitHub source linking** | User code lives in scryer's DB via push; GitHub-backed source is a v1 ergonomic win | When you want IDE-driven editing of Scorers + git workflow |
| **`scryer export --git`** | Cute feature; not v0 critical | When you want offline audit history analysis |
| **Bulk endpoints** (POST /runs/query, /datasets/.../records/bulk) | Single-resource endpoints sufficient for v0 scale | When iterating over 50+ resources at once becomes common |
| **SSE on `/runs/{id}/events`** | Polling with backoff sufficient for solo+invite-only | When agent count grows enough that polling spam is real |
| **Public unauthenticated share links** | All resources require auth in v0 | When you want to share a Run with someone outside your invite circle |
| **Audit log archival to R2** | Wait until disk fills; just monitor for now | Postgres free-tier disk hits 80% |
| **Hard delete + RetentionPolicy** | Soft delete only in v0 | Compliance requires data deletion |
| **2FA, OAuth, SSO** | Email+password sufficient for invite-only beta | Paid tier or compliance demand |
| **Settings/Environment noun** for Workspace-scoped defaults | Per-resource explicit config is simpler | When repeated per-resource settings become tedious |
| **Org tier above Workspace** | Workspace covers team scoping for v0 | When a single company wants separate Workspaces per team with shared billing |

---

## 17. Implementation staging (high-level)

Not a task list — a sketch of build order. Actual implementation is its own session(s); that's what generates schema SQL, API endpoint code, CLI commands, etc. from this plan.

### Phase 0 — Repo bootstrap (~2-4 hours)

1. Create `~/code/scryer/` repo (separate from traitinterp)
2. Set up Python 3.12 project: `pyproject.toml`, `uv` or `pip-tools` for deps
3. Set up Alembic for migrations
4. Set up FastAPI skeleton with basic `/api/v1/healthz`
5. Set up Neon Postgres + Railway service + R2 bucket
6. Set up Sentry or similar for error tracking
7. Set up GitHub Actions CI (lint + type check + basic tests)

### Phase 1 — Foundation cluster (Identity + Org + Auth) (~1-2 days)

1. Generate Cluster 1 schema from §11 (write Alembic migration)
2. Build core service layer for User CRUD, Workspace creation, Project creation
3. Build auth middleware: JWT verification + ApiKey verification + scope checking
4. Build invitation flow: token generation + signup endpoint
5. Build Credential CRUD with AES-GCM encryption
6. Build basic CLI: `scryer auth login`, `scryer workspace list`, `scryer project list`
7. Tests: invitation lifecycle, multi-user isolation, API key revocation

### Phase 2 — Evaluation core (~2-3 days)

1. Generate Cluster 2 schema; Alembic migration
2. Build Dataset CRUD with content hashing + versioning + parent_id lineage
3. Build Scorer / Agent / Tool / Prompt CRUD (similar pattern)
4. Build Task creation (validates Dataset + Scorer + optional Agent + Prompt; computes content hash)
5. Build Run executor: subprocess with stripped env + ulimits + timeout + heartbeat
6. Build Result + Trace + TraceStep persistence
7. Build basic CLI commands for these resources
8. Tests: end-to-end Run of a simple Scorer; resume after simulated crash; rescore via different Scorer version

### Phase 3 — Activity + Audit + Governance + Automation (~1-2 days)

1. Generate Cluster 3 schema
2. Build AuditEvent middleware (decorator pattern; before/after capture)
3. Build Suite + SuiteRun
4. Build Trigger + dispatcher endpoint hit by Railway Cron
5. Build Webhook + WebhookDelivery + retry worker
6. Build Tag + ResourceTag
7. Build UsageRecord per-LLM-call instrumentation
8. Build Budget circuit-breaker check before LLM calls
9. Tests: Trigger fires; AuditEvent captured; Webhook delivery + retry; budget enforcement

### Phase 4 — Collaboration (~1 day)

1. Generate Cluster 4 schema
2. Build Comment CRUD with author object + structured fields + references + mentions parser
3. Build CommentVersion edit history
4. Build Collection CRUD with members[].group/note + purpose enum
5. Build auto-Comment writers for `run.completed`, `run.failed`, `suite.regression_detected`, `trigger.fired`

### Phase 5 — Dashboard (~2-3 days)

1. Set up Jinja + htmx + Tailwind + Alpine
2. Build login/signup pages
3. Build Workspace switcher + Project switcher
4. Build Runs list page with filtering
5. Build Run detail page (scorecard + per-record table + Trace inline)
6. Build Diff page (`POST /runs/compare` rendered)
7. Build Datasets list + detail + version history
8. Build Activity feed widget
9. Build Comments rendering on resource detail pages

### Phase 6 — Agent UX (~1-2 days)

1. Build `GET /api/v1/me` endpoint
2. Mount `fastapi-mcp` at `/mcp`
3. Add OpenAPI metadata discipline (lint rule + back-fill descriptions)
4. Add `Scryer-Version` + `Sunset` headers
5. Add RFC 9457 error responses with type URIs

### Phase 7 — Wire-in to traitinterp (~1 day)

1. Install scryer Python SDK in traitinterp via `pip install -e ~/code/scryer`
2. Migrate `experiments/judge_optimization/` workflow:
   - Re-label golden set into a Dataset with sub-dimension labels (Claude Code does this)
   - Push current judge prompt as Scorer source
   - Define a Task; run it; iterate
3. Replace `docs/viz_findings/llm-judge-optimization.md` with rigorous, harness-backed write-up

### Phase 8 — Production polish (~1-2 days)

1. Set up Railway Cron for: heartbeat reaper, keep-alive, Trigger dispatcher, Webhook retry
2. Set up monitoring (Postgres disk, R2 usage, Workspace spend)
3. Write README + agent-guide docs
4. Beta-invite first collaborator (or self via separate ServiceAccount)

### Total estimated time

**~10-15 days of focused engineering** with Claude Code. Could be done in 2-3 weeks of part-time work or 1-2 weeks of full-time. The schema design (~50% of architectural difficulty) is already done in this plan.

---

## 18. Open questions

After 100+ design decisions, none should be left unsettled. If anything resurfaces during implementation, treat it as a real design question requiring its own session — not a quick fix.

The honest list of things that may need re-decision during build:

- **Polymorphic FK pattern**: I locked "two nullable columns + CHECK constraint" but if SQLAlchemy 2.x async session ergonomics fight this, may need to use a discriminator pattern. Worth verifying in Phase 1.
- **Trace storage threshold**: locked "if n_steps < 100, store in trace_steps; else spill to R2." May need tuning based on actual size distribution.
- **Encryption key rotation for Credentials**: schema supports `encryption_key_version` but the rotation procedure isn't specified. v0 ships with a single static key; rotation is its own operational runbook later.
- **Polling interval defaults**: locked "exponential backoff 1s, 2s, 4s, 8s, cap 30s" but the SDK should make this configurable.

---

## 19. What's NOT in this plan

The plan is the durable artifact of design decisions. The following are explicitly NOT here and should be derived in their own sessions:

- **Full SQL CREATE TABLE statements** — generate from §10 noun set + §11 schema overview. Cluster-by-cluster.
- **API endpoint enumeration** — generate from §10 + standard CRUD per resource + §12 conventions
- **CLI command list** — generate from API endpoints + §13 naming conventions (noun-first, kebab-case, singular)
- **SDK method enumeration** — generate from API endpoints
- **Repo / module structure** — generate from §4 architecture + standard FastAPI project layout
- **Dashboard page wireframes** — generate from §5 sharing + §10 noun set + §17 Phase 5
- **Pydantic models** — generate from schema clusters
- **MCP tool descriptions (8 curated)** — generate when writing the curated MCP server in v1
- **Test plan** — generate per Phase in §17
- **Deployment runbook** — generate when Phase 0 ships
- **Lint rules** — for: required OpenAPI descriptions, required `archived_at` filter on read queries, required `reason` on destructive operations, etc.

Each of these is its own session of work. Claude Code reading THIS plan generates them at build time.

---

## Appendix A: Glossary (alphabetized)

- **Action** — old name for AuditEvent (renamed)
- **Agent** — first-class noun; the multi-step LLM pipeline being evaluated
- **ApiKey** — first-class noun; programmatic auth credential
- **Approval** — DEFERRED to v1; pause-for-human-Y/N gating
- **Audit** — colloquial for AuditEvent + auto-Comments combined
- **AuditEvent** — first-class noun; append-only audit log row
- **Budget** — first-class noun; spend limit per (principal, period)
- **Collection** — first-class noun; curated bundle of pointers to scryer resources
- **Comment** — first-class noun; flat timestamped annotation with optional resource references
- **Credential** — first-class noun; encrypted external API key for outbound LLM calls
- **Dataset** — first-class noun; collection of polymorphic Records to evaluate against
- **Diff** — computed-on-the-fly comparison between Runs; not a noun
- **Hook** — old name for Webhook (renamed)
- **Invitation** — first-class noun; signup token with lifecycle
- **MCP** — Model Context Protocol; server scryer ships at `/mcp`
- **Plan** — DEFERRED to v1; multi-step agent work record
- **Project** — first-class noun; work-unit scoped under Workspace
- **Prompt** — first-class noun; versioned prompt template
- **Result** — first-class noun; per-record score within a Run
- **Run** — first-class noun; one execution of a Task version
- **Scorer** — first-class noun; per-record scoring function (source code stored)
- **ServiceAccount** — first-class noun; bot identity (no email)
- **Suite** — first-class noun; named collection of Tasks with aggregation
- **Tag** — first-class noun; Workspace-scoped orthogonal label
- **Task** — first-class noun; binding of (Dataset, Scorer, Agent|None, Prompt|None)
- **Tool** — first-class noun; registered tool definition Agents can call
- **Trace** — first-class noun; per-Record execution capture
- **TraceStep** — sub-resource of Trace; normalized for tool_id queries
- **Trigger** — first-class noun; schedule or webhook event firing Tasks/Suites
- **UsageRecord** — first-class noun; per-LLM-call entry with tokens/cost/latency
- **User** — first-class noun; human identity
- **Webhook** — first-class noun; outbound webhook fired on scryer events
- **WebhookDelivery** — sub-resource of Webhook; retry attempts
- **Workspace** — first-class noun; billing/team boundary

---

## Appendix B: Conversation context

This plan distills decisions made during a multi-hour design conversation on 2026-05-01 / 2026-05-02. Key context:

### Origin

scryer was conceived as the user's Killian-grade artifact for an interview at Guild.ai (Agents+Evaluation role). The framing evolved: not building "for Killian" specifically, but informed by the JD's emphasis on research-engineering style + benchmark/eval artifacts that push the broader foundation model ecosystem forward. The first concrete user is the user's traitinterp judge_optimization rebuild, which is currently AI-written and bad — scryer-backed evaluation is the path to fixing it properly.

### Major decision pivots during the conversation

- Originally considered TS server + Python SDK (hybrid); pivoted to Python-only after considering boundary tax + numerical core fit
- Originally considered Tier 2 sandbox (nsjail/bubblewrap); empirically tested Railway via `unshare -U` — confirmed blocked. Settled for Tier 1+ subprocess.
- Originally considered server-side execution as v1 deferral; user pushed for v0 ("yes/no/yes" — server runs code, no separate per-Scorer venv, source stored). Locked v0.
- Originally locked Approval as first-class; critic agent flagged premature for invite-only beta; deferred to v1.
- Originally locked Plan; smeared into Comment.structured fields; reconsidered; Plan stays deferred but Comment.structured slimmed to just `tldr` + `confidence`.
- Originally locked hash chain on Action (now AuditEvent); critic flagged it as theater without external anchor; deferred to v1.
- Originally proposed 8-scope ApiKey vocabulary; critic flagged GitHub-PAT cargo-cult; reduced to 4 scopes (read, write, admin, impersonate).

### Use case validation

Three concrete use cases were walked through by spawned agents:
- **judge_optimization** — found Tool, ServiceAccount, Approval all UNUSED; validated Dataset/Scorer/Run/Result/Comment/Collection. Flagged a real gap: no first-class Comparison noun (resolved as computed-on-the-fly endpoint).
- **multi-step agent eval (bitbucket-pr-splitter)** — validated Tool first-class IF cross-Agent reuse is real (it is); critical structural requirement: TraceStep stores `tool_id` as FK, not string.
- **autonomous regression-detection loop** — strong case for ServiceAccount + Approval + Plan; deferred Approval and Plan to v1 but kept ServiceAccount.

### Critic-pass results

A critic agent stress-tested every locked decision. Key accepted recommendations:
- Drop hash chain (theater without anchor)
- Strip Comment.structured to `{tldr, confidence}` only (Plan-by-stealth)
- Drop `Comment.claims[]` (duplicates AuditEvent)
- Pick idempotency middleware only (drop per-noun keys)
- Collapse spend governors into single Budget noun
- Defer `claimed_by`/`claimed_until` (no contention at 3-user scale)
- Defer `Run.baseline_run_id` (overlaps Suite baseline)
- Defer MCP curation (ship `/mcp/full` first, observe usage)
- Defer `agent_feed` endpoint (premature)
- Drop `Comment.references[].selector` union (use opaque `fragment` string instead)

Rejected recommendations:
- "Drop ServiceAccount, use User.kind enum" — autonomous-loop use case justified it
- "Keep Approval in v0" — deferred to v1 since autonomous-loop use case is itself v1
- "Reduce scopes to 2" — kept at 4 (read/write/admin/impersonate)

### Things almost-locked-but-explicitly-rejected

- **TypeScript server** — rejected for Python-only stack
- **GraphQL API** — rejected; REST + RFC 9457 is enough
- **WebSockets** — rejected; SSE is simpler when added in v1
- **Public signup** — rejected; invite-only beta
- **Email-based auth (passwordless / magic links)** — rejected; email+password for v0
- **Public unauthenticated share links** — rejected; auth required for everything in v0
- **Per-Scorer custom dependencies** — rejected for v0; pre-baked image only
- **Annotation/labeling UI in v0** — rejected; pre-labeled JSONL ingested
- **Plan as first-class noun in v0** — rejected; Comments + metadata.idempotency_key cover dedup needs
- **Real Docker sandbox** — rejected; subprocess + stripped env for invite-only

### Why the design ended up where it did

Two competing pressures:
1. **"Build it right from the start"** — pushed toward Architecture 3 (real backend, multi-user, dashboard, MCP server) and rich noun set
2. **"Solo dev, invite-only beta, cost-conscious"** — pushed toward minimal viable shape

Where they conflicted, the user repeatedly chose "build it right" with critic-driven corrections to avoid over-engineering. The final design is a real production-shape system (multi-user from day 1, real auth, audit log, sandbox) but defers anything speculative (Plan, Approval, Annotation, Hash chain, Curated MCP) until evidence demands it.

The 25 first-class nouns are at the upper end of what feels learnable; a tighter system would have ~15-18. The choice to keep 25 is honest: each was use-case-validated and several would require painful migration if added later.

### What this plan replaces

This plan is the durable artifact of the design phase. Future sessions:
- Generate schema SQL from §10 + §11
- Generate API endpoints from §10 + §12
- Generate CLI commands from §13
- Generate repo structure from §4
- Generate test plans from §17

Without this plan, those future sessions would re-litigate locked decisions. With it, they execute against settled answers.

### Files referenced

- `/Users/ewern/Desktop/code/resume-automation/ongoing/learning/control_planes_primer.md` — the 6-primitive frame + eval-vs-test + reproducibility primitives. Required reading for understanding scryer's architectural framing.
- `/Users/ewern/Desktop/code/trait-stuff/traitinterp/experiments/judge_optimization/` — first user; the workflow scryer rebuilds.
- `/Users/ewern/Desktop/code/trait-stuff/traitinterp/utils/judge_backends.py` — battle-tested judge code that scryer's first Scorers will adapt (NOT bundled into scryer SDK; lives in user code).
- `/Users/ewern/code/cc-plugins/r/commands/plan-experiment.md` — inspired Comment + Collection structure for decision-history use cases.

---

**End of plan. ~7500 words.**
