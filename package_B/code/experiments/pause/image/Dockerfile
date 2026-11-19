# Agent image: the official prebuilt SWE-bench instance image (repo at /testbed, conda env
# "testbed" ready) plus Node, a pinned Claude Code and the pause hook. Built by runner.py:
#   docker build --build-arg BASE=<swebench image> --build-arg CC=<cc layer image> -t <tag> .
ARG BASE
ARG CC=cachepause-cc:2.1.326
FROM ${CC} AS cc
FROM ${BASE}
COPY --from=cc /usr/local/bin/node /usr/local/bin/node
COPY --from=cc /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=cc /usr/local/lib/claude-code-version.txt /usr/local/lib/claude-code-bin.txt /usr/local/lib/
RUN ln -sf "$(cat /usr/local/lib/claude-code-bin.txt)" /usr/local/bin/claude && mkdir -p /opt/pause /out
COPY pause_hook.sh entry.sh session.sh /opt/pause/
RUN chmod 0755 /opt/pause/pause_hook.sh /opt/pause/entry.sh /opt/pause/session.sh && claude --version
ENTRYPOINT ["/opt/pause/entry.sh"]
