# Lens 2: User Pain Points Discovery Report
**Date**: 2026-04-13  
**Data Sources**: `.hermes_history` (193 lines), `gateway.log` (8487 lines), `agent.log` (11.8K lines), `errors.log.*` (~22K lines)

---

## Search 1: Error / Failed / Exception Patterns

### Key Recurring Patterns

| Pattern | Count | Severity | Fixable? |
|---------|-------|----------|----------|
| **Session summarization failed** | 119+ occurrences in gateway, 30+ in errors | HIGH | ✅ Yes |
| **Firecrawl init failure** (missing config/auth) | ~6 occurrences | MEDIUM | ✅ Yes |
| **Vision tool errors** (invalid source, too large, blocked, download fail) | ~15 unique error paths | MEDIUM | ✅ Yes |
| **Docker backend fallback** (`/usr/bin/docker` not found → falls back to bash) | ~6 occurrences | LOW | ✅ Yes |
| **MCP server connection failures** ("cannot reach server") | ~5 occurrences | MEDIUM | ✅ Yes |
| **Approval callback TypeError** (`unexpected keyword argument 'allow_permanent'`) | Recurring across test runs | MEDIUM | ✅ Yes |
| **Terminal env misconfig** (unknown backend, missing SSH creds, Modal creds) | ~6 unique error messages | LOW | ✅ Yes |

### Top Error Producers (by module frequency in errors.log.1)
1. `gateway.platforms.matrix` — 995 entries (decryption failures)
2. `gateway.run` — 530 entries (startup conflicts, platform failures)
3. `gateway.platforms.discord` — 465 entries
4. `root` (agent core) — 433 entries
5. `tools.url_safety` — 278 entries (blocked URLs)
6. `gateway.platforms.telegram` — 258 entries
7. `tools.image_generation_tool` — 257 entries
8. `agent.prompt_builder` — 254 entries

---

## Search 2: User Frustration ("not working / broken / doesn't work")

### Direct User Quotes from Session History
| Quote | Context |
|-------|---------|
| "the api is work here, why is it not working for the subagents?" | Subagent credential/API routing issue |
| "why is the self-improve cron not happening" | Cron silently failing, no user feedback |
| "I am not seeing any new logs from fork-self-improve cron" | Cron visibility gap |
| "the original-project-factory so far has made useless project" | Skill quality not meeting expectations |
| "That still seems like it will make useless things" | Persistent quality issue |
| "did you stop?" | Agent silently stopped mid-task |
| "why are you doing nothing" | Agent stall with no feedback |
| "First is there a way we can see what is working or active" | **No visibility into cron/status** — asked TWICE |

### Recurring Frustration Themes
1. **No visibility into running crons/jobs** — User asked twice, never resolved
2. **Silent failures** — Crons stop running, no notification
3. **Subagent API routing** — Works for main agent, fails for subagents
4. **Agent stalls mid-task** — No progress feedback, user has to check manually
5. **Quality of generated projects** — Original Project Factory making useless things

---

## Search 3: Timeout / Slow / Retry Performance Issues

### Session Summarization — #1 Performance Problem
- **119 failures** in gateway.log alone (spanning Apr 11–12)
- Two failure modes:
  - **`Request timed out`** (~60% of failures) — Summarization API call timing out
  - **`Error code: 429 Rate limit`** (~40% of failures) — Rate limited after 3 attempts
- **Impact**: Every session transition loses its summary, degrading context continuity
- **Root cause**: Summarization retries 3 times then gives up; no backoff or fallback

### Cron Job Timeouts
- `fork-sync` job timed out at 601s (limit 600s) — stuck on `delegate_task`
- **Impact**: Self-improvement pipeline stalls completely

### API Timeout Chain
- `openai.APITimeoutError` → `httpx.ReadTimeout` → `httpcore.ReadTimeout`
- Multiple layers of timeout wrapping with no adaptive timeout strategy

### Rate Limiting
- 429 errors from auxiliary model provider (code 1302 — "Rate limit reached")
- Summarization uses same rate-limited endpoint with no queueing or throttle

---

## Search 4: "How do I" / Confusion / UX Gaps

### User Confusion Patterns from History
| User Message | UX Gap |
|-------------|--------|
| "what is different from `self-improve` and `fork-self-improve`?" | Naming confusion between two similar crons |
| "How does the new one discover where improvements can be made?" | Discovery process not documented/visible |
| "First is there a way we can see what is working or active?" | **No status dashboard or visibility** |
| "/model codex" then "/model codex " (with trailing space) | CLI parsing inconsistency |
| "run self improve skill again" | User doesn't know how to trigger manually |
| "make sure Original Project Factory cron is setup correctly" | No way to validate cron health |
| "I want you to fix it" (about broken cron) | No self-healing capability |

### UX Gaps Identified
1. **No `/status` or `/health` command** — Users can't see what crons are active/healthy
2. **No cron failure notifications** — Jobs fail silently
3. **No `/cron test` or `/cron validate`** — Can't verify cron config before saving
4. **Discovery process is opaque** — User asked how self-improve discovers issues
5. **No progress feedback during long tasks** — User thinks agent stopped

---

## Top 5 Actionable Pain Points (Ranked by Impact × Fixability)

### 1. 🔴 Session Summarization Reliability (119+ failures)
- **Problem**: Every session summarization fails silently (timeout or rate limit)
- **Impact**: HIGH — Context is lost between sessions, degrading agent quality over time
- **Fix**: Add exponential backoff with jitter, implement a local fallback summarizer, queue failed summaries for retry

### 2. 🔴 No Visibility into Cron/Job Status
- **Problem**: User asked twice "is there a way to see what is working?" — no solution exists
- **Impact**: HIGH — Core UX gap, users can't monitor autonomous operations
- **Fix**: Add `/status` command showing active crons, recent run results, health status

### 3. 🟠 Silent Cron Failures
- **Problem**: Cron jobs fail (rate limit, timeout, errors) with no user notification
- **Impact**: MEDIUM — Self-improve pipeline stalls for hours without anyone knowing
- **Fix**: Add failure notifications via gateway, implement cron health checks with auto-restart

### 4. 🟠 Agent Stalls Without Feedback
- **Problem**: Agent stops mid-task, user asks "did you stop?" and "why are you doing nothing?"
- **Impact**: MEDIUM — User trust erosion, manual babysitting required
- **Fix**: Add heartbeat/progress logging, notify user when agent is idle >5 minutes

### 5. 🟡 Subagent API Routing Inconsistency
- **Problem**: "the api is working here, why is it not working for the subagents?"
- **Impact**: MEDIUM — Delegation fails when it should work
- **Fix**: Ensure credential pool propagates to subagents consistently; add subagent API diagnostic

---

## Files Analyzed
- `/home/sonny/.hermes/.hermes_history` — User conversation history
- `/home/sonny/.hermes/logs/gateway.log` — 8487 lines of gateway runtime logs
- `/home/sonny/.hermes/logs/agent.log` — 11857 lines of agent runtime logs  
- `/home/sonny/.hermes/logs/errors.log` — Current error log
- `/home/sonny/.hermes/logs/errors.log.1` — 20352 lines of historical errors
- `/home/sonny/.hermes/logs/errors.log.2` — 1486 lines of older errors
- `/home/sonny/.hermes/logs/self-improve/backlog.md` — Existing improvement backlog
