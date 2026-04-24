"""Context-files bundle injected into the system prompt when the adapter
declares non-empty `context_files`. Produces a short list of files the LLM
may read via the READ-FILE protocol, plus a one-paragraph explanation of
the protocol semantics (2-state: read or emit M1b; cap at 3 reads per
candidate).
"""


def render_context_files_section(*, context_files: list[str]) -> str:
    """Return a string to splice into the system prompt, or "" when the list
    is empty. The rendered section is markdown-flavored plain text.
    """
    if not context_files:
        return ""

    lines = [
        "",
        "## Files you may read (read-only)",
        "",
        "The following files are available for inspection before you emit "
        "your M1b manifest. They contain types, helpers, or documentation "
        "that may be useful context for the optimization:",
        "",
    ]
    for rel in context_files:
        lines.append(f"- {rel}")

    lines.extend([
        "",
        "### READ-FILE protocol",
        "",
        "To read one of the files above, emit EXACTLY this as your entire "
        "response (no other text, no backticks, no commentary):",
        "",
        "```",
        "READ-FILE: <path>",
        "```",
        "",
        "The file content will be returned as the next user message and "
        "you may then emit your M1b manifest. You may read at most 3 "
        "files per candidate. Otherwise, emit your M1b manifest directly "
        "in the `# EDIT-BLOCK-START / # EDIT-BLOCK-END` envelope.",
        "",
    ])
    return "\n".join(lines)
