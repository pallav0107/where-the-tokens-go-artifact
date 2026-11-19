#!/bin/bash
# PostToolUse hook (both conditions, identical): counts tool calls across the whole run (main loop
# and subagents) and, at the tool-call indices in PAUSE_AT, records the planned point and sleeps
# PAUSE_SECONDS (condition A: 0, so the point is logged and nothing sleeps). The hook runs after the
# tool and before the next API request, so a pause lands between two requests.
#   PAUSE_AT="5,15,25"  PAUSE_SECONDS=360  PAUSE_STATE_DIR=/out/pause
# Log: $PAUSE_STATE_DIR/calls.jsonl (one line per tool call) and pauses.jsonl (one line per planned
# point, with UTC start/end of the sleep). Always exits 0 and prints nothing, so it never changes
# what the agent sees.
set -u
dir="${PAUSE_STATE_DIR:-/out/pause}"
mkdir -p "$dir"
cat > /dev/null
now() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
exec 9>>"$dir/.lock"
flock 9
n=$(( $(cat "$dir/count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$dir/count"
printf '{"call":%d,"ts":"%s"}\n' "$n" "$(now)" >> "$dir/calls.jsonl"
flock -u 9
secs="${PAUSE_SECONDS:-0}"
for k in ${PAUSE_AT//,/ }; do
  if [ "$k" = "$n" ]; then
    s="$(now)"
    if [ "$secs" -gt 0 ]; then sleep "$secs"; fi
    printf '{"call":%d,"start":"%s","end":"%s","seconds":%d}\n' "$n" "$s" "$(now)" "$secs" >> "$dir/pauses.jsonl"
  fi
done
exit 0
