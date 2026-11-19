#!/bin/bash
# Container entrypoint: fresh HOME, hook settings, one headless Claude Code run on /testbed.
# Inputs (env, set by runner.py; the API key arrives only through docker --env-file):
#   MODEL MAX_TURNS WALL_SECONDS PAUSE_AT PAUSE_SECONDS CC_MAX_BUDGET_USD (per-run safety cap, the
#   CLI's own accounting); prompt in /out/prompt.txt. ENABLE_PROMPT_CACHING_1H, if set, makes an
#   API-key session write one-hour entries (Claude Code 2.1.326 defaults API keys to five minutes).
# Outputs under /out: result.json (CLI JSON), stderr.txt, patch.diff, exit.json, pause/*.jsonl;
# transcripts stay in $HOME/.claude/projects for the runner to docker cp out.
set -u
export HOME=/agent-home
rm -rf "$HOME" && mkdir -p "$HOME/.claude" /out/pause
# The SWE-bench image activates the "testbed" conda env from /root/.bashrc; keep that for the agent's shell.
[ -f /root/.bashrc ] && cp /root/.bashrc "$HOME/.bashrc"
export PATH="/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH"
hook_timeout=$(( ${PAUSE_SECONDS:-0} + 120 ))
cat > "$HOME/.claude/settings.json" <<JSON
{
  "hooks": {"PostToolUse": [{"matcher": "*", "hooks": [
    {"type": "command", "command": "/opt/pause/pause_hook.sh", "timeout": ${hook_timeout}}]}]}
}
JSON
export PAUSE_STATE_DIR=/out/pause
cd /testbed
git config --global --add safe.directory /testbed >/dev/null 2>&1
start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
nice -n 10 timeout --kill-after=30 "${WALL_SECONDS}" claude -p "$(cat /out/prompt.txt)" \
  --model "${MODEL}" --max-turns "${MAX_TURNS}" --output-format json \
  --max-budget-usd "${CC_MAX_BUDGET_USD:-8}" --dangerously-skip-permissions > /out/result.json 2> /out/stderr.txt
rc=$?
end=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
git -C /testbed add -A >/dev/null 2>&1
git -C /testbed diff --cached > /out/patch.diff 2>/dev/null
printf '{"rc":%d,"start":"%s","end":"%s","claude_code":"%s"}\n' "$rc" "$start" "$end" \
  "$(cat /usr/local/lib/claude-code-version.txt 2>/dev/null | tr -d '\n')" > /out/exit.json
exit 0
