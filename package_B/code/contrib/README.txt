tokens_contrib.py
=================

WHAT IT IS
One Python script that reads the Claude Code logs on your own computer and counts where the
tokens went: how much of your input was read from the prompt cache, how much was written to it,
how often a cached prompt had to be written again and why (for example after an idle gap of over
an hour), and how list-price cost splits across token kinds. It is for the AGENT 2027 workshop
paper "Where the Tokens Go: Auditing Cache Reads and Re-Writes in a Multi-Agent Coding Harness",
and applies exactly the paper's counting rules.

It sends nothing anywhere. You read the result and decide whether to send it.


HOW TO RUN IT
1. Save tokens_contrib.py in any folder and open a terminal in that folder.
2. Run this one line:

       python tokens_contrib.py

   (On Windows "py tokens_contrib.py" also works; on macOS use "python3 tokens_contrib.py".)
   It needs Python 3.9 or newer and nothing else: no installs, standard library only.
   It usually takes under a minute or two.
3. It prints the result and writes tokens_contrib_result.json in the same folder.

Optional:
   python tokens_contrib.py --self-test       check the script on a built-in synthetic log first
   python tokens_contrib.py --root FOLDER     if your logs are not in ~/.claude/projects
                                              (or $CLAUDE_CONFIG_DIR/projects if you set that)
   --since YYYY-MM-DD / --until YYYY-MM-DD    only requests from / before 00:00 UTC on that date


WHAT IS SHARED (everything in tokens_contrib_result.json)
- Counts: API requests, sessions, subagent runs, the number of months covered and the months
  themselves (YYYY-MM).
- Token totals by kind: uncached input, cache read, cache write (5-minute and 1-hour), output.
- Percentages computed from them: cache-read share of input, cost shares at Anthropic list
  prices, the share of cache writes that re-write an already-cached prompt, and those re-writes
  split by cause and by idle gap; log lines per request.
- Claude model names, such as claude-opus-5-5. Anything that is not a standard Claude model id
  (for example a gateway or deployment name) is written as "other".
- Claude Code version numbers, such as 2.1.300. Anything else is written as "other".
- A few data-quality counters (log files read, lines skipped) and the script's own sha256.


WHAT IS NEVER SHARED
Your prompts, Claude's replies, tool names or tool inputs, file contents, file or folder paths,
project names, session, message or request ids, exact dates or times (only the month), your user
name, email addresses or host names. Before writing, the script checks every piece of text in the
result and refuses to write the file if any of it is not a month, a Claude model name, a version
number, its own hash or the result's fixed label.


HOW TO CHECK THE OUTPUT BEFORE SENDING
- Open tokens_contrib_result.json in any text editor. It is exactly the JSON the script printed.
  Every value is a number, true/false/null, a month, a model name, a version number, the hash or
  the fixed label "tokens_contrib_result".
- Search it for anything you consider private (a project, client or user name). You should find
  nothing. If you do find something, please do not send the file; tell the sender instead.
- The script is plain Python you can read. It opens no network connections.
- Please run it unmodified: the paper only accepts results whose script_sha256 matches the
  released script, whose sha256 (LF line endings) is
      b08c5e1538d0746df4353be144cb5e77c1e73f2ad4d1cd4485a759822597bd40


HOW TO SEND IT BACK
Reply to the person who gave you the script and attach only tokens_contrib_result.json. Do not
send your logs or anything else. In the paper you are anonymous, counted as one of "N other
developers", unless you say you would like to be acknowledged by name.

If anything goes wrong, the script prints a one-line message and writes nothing. Send that line
instead.
