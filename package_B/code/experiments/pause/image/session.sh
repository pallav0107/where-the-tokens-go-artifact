#!/bin/bash
# Chained session driver (PREREG_pause.md, Amendment 2). One container = one continuing Claude Code
# conversation over several tasks of the same repository and version.
#   session.sh hold                 fresh HOME and settings, then wait (the runner docker-execs tasks)
#   session.sh task K BASE_COMMIT   reset /testbed to BASE_COMMIT and run task K: `claude -p` for K=1,
#                                   `claude -p --continue` afterwards (same conversation, same cwd)
# Env (runner): MODEL MAX_TURNS WALL_SECONDS CC_MAX_BUDGET_USD [ENABLE_PROMPT_CACHING_1H]; the API key
# arrives only through docker --env-file. Prompt for task K in /out/task_K/prompt.txt.
# Outputs per task in /out/task_K/: result.json, stderr.txt, patch.diff, exit.json. The PostToolUse
# hook only logs tool calls here (PAUSE_AT empty): pauses happen in the runner, between tasks.
set -u
export HOME=/agent-home
export PATH="/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH"
export PAUSE_STATE_DIR=/out/pause PAUSE_AT="" PAUSE_SECONDS=0
case "${1:-}" in
  hold)
    rm -rf "$HOME" && mkdir -p "$HOME/.claude" /out/pause
    [ -f /root/.bashrc ] && cp /root/.bashrc "$HOME/.bashrc"
    cat > "$HOME/.claude/settings.json" <<'JSON'
{
  "hooks": {"PostToolUse": [{"matcher": "*", "hooks": [
    {"type": "command", "command": "/opt/pause/pause_hook.sh", "timeout": 60}]}]}
}
JSON
    git config --global --add safe.directory /testbed >/dev/null 2>&1
    exec sleep infinity
    ;;
  task)
    k="$2"; base="$3"; o="/out/task_$k"
    mkdir -p "$o"
    cd /testbed
    # keep ignored files (editable-install metadata); drop the previous task's edits and new files
    git reset -q --hard && git clean -fdq && git checkout -qf "$base" || { echo "checkout failed" > "$o/stderr.txt"; exit 3; }
    cont=(); [ "$k" -gt 1 ] && cont=(--continue)
    # Amendment 3 (injected invalidators): optional per-task model override and extra CLI arguments, one per
    # line; without these files the command is exactly that of Amendment 2
    model="${MODEL}"; [ -s "$o/model" ] && model="$(cat "$o/model")"
    extra=(); [ -s "$o/extra_args" ] && mapfile -t extra < "$o/extra_args"
    start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    nice -n 10 timeout --kill-after=30 "${WALL_SECONDS}" claude -p "$(cat "$o/prompt.txt")" "${cont[@]}" "${extra[@]}" \
      --model "${model}" --max-turns "${MAX_TURNS}" --output-format json \
      --max-budget-usd "${CC_MAX_BUDGET_USD}" --dangerously-skip-permissions > "$o/result.json" 2> "$o/stderr.txt"
    rc=$?
    end=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    git add -A >/dev/null 2>&1
    git diff --cached > "$o/patch.diff" 2>/dev/null
    printf '{"task":%d,"rc":%d,"start":"%s","end":"%s","base_commit":"%s","continue":%s,"model":"%s","extra_args":%d}\n' \
      "$k" "$rc" "$start" "$end" "$base" "$([ "$k" -gt 1 ] && echo true || echo false)" "$model" "${#extra[@]}" \
      > "$o/exit.json"
    exit 0
    ;;
  *) echo "usage: session.sh hold | task K BASE_COMMIT" >&2; exit 2 ;;
esac
