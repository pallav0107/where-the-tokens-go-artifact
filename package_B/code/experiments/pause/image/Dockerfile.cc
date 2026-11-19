# Claude Code layer, built once and copied into every SWE-bench instance image (Dockerfile).
# Pinned: Node image by tag, Claude Code 2.1.326 from the npm registry tarballs (the package and its
# linux-x64 native binary), downloaded into pkgs/ beforehand and checked against the registry's
# sha1 shasums, so the slow-link build is resumable and the bytes are verified:
#   curl -C - -o pkgs/claude-code-2.1.326.tgz \
#     https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-2.1.326.tgz
#   curl -C - -o pkgs/claude-code-linux-x64-2.1.326.tgz \
#     https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.326.tgz
#   docker build --network host -f Dockerfile.cc -t cachepause-cc:2.1.326 .
FROM node:22.22.0-bookworm-slim
COPY pkgs/claude-code-2.1.326.tgz pkgs/claude-code-linux-x64-2.1.326.tgz /tmp/pkgs/
RUN cd /tmp/pkgs \
 && echo "a74c405d24800c82e5e157001e984175982f1721  claude-code-2.1.326.tgz" | sha1sum -c - \
 && echo "0bddf2a2dec256e33db9509da3ce47412df68682  claude-code-linux-x64-2.1.326.tgz" | sha1sum -c - \
 && npm install -g --no-fund --no-audit --prefer-offline \
      ./claude-code-linux-x64-2.1.326.tgz ./claude-code-2.1.326.tgz \
 && cd / && rm -rf /tmp/pkgs && npm cache clean --force \
 && claude --version > /usr/local/lib/claude-code-version.txt \
 && readlink -f "$(command -v claude)" > /usr/local/lib/claude-code-bin.txt
