"""Path-safe file reader for the READ-FILE protocol.

Resolves a relative path against a list of allowed roots; returns content
iff the resolved absolute path is genuinely under one of those roots (no
symlink/parent-escape tricks). Enforces a max-size cap so a benchmark can't
blow up the prompt.
"""
from pathlib import Path


DEFAULT_MAX_BYTES = 128 * 1024  # 128 KB


class ReadFileError(Exception):
    pass


def read_file_safe(
    *,
    rel_path: str,
    allowed_roots: list[Path],
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowed_relpaths: list[str] | None = None,
) -> str:
    """Return `rel_path`'s content from the first allowed root that contains it.

    Safety: `Path.resolve()` the candidate and compare against each root's
    resolved form — guarantees no `../..` or absolute-path escape even
    through symlinks.

    Spec §9 allow-list: when `allowed_relpaths` is provided (non-None), the
    requested path must match one of those entries verbatim. This implements
    the `allowed_read_paths = context_files ∪ evolvable_files ∪
    datalayout_api_headers` constraint and prevents the model from reading
    arbitrary files that merely happen to live under benchmark_root. When
    `allowed_relpaths` is None, the legacy root-only check applies (kept for
    Fork-1 / direct callers that haven't been migrated yet).
    """
    if allowed_relpaths is not None and rel_path not in set(allowed_relpaths):
        raise ReadFileError(
            f"{rel_path}: not in declared READ-FILE allow-list "
            f"(context_files ∪ evolvable_files ∪ datalayout_api_headers)"
        )
    for root in allowed_roots:
        root_resolved = Path(root).resolve()
        candidate = (Path(root) / rel_path).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            # candidate escapes root — try next
            continue
        if candidate.is_file():
            size = candidate.stat().st_size
            if size > max_bytes:
                raise ReadFileError(
                    f"{rel_path}: too large ({size} bytes > {max_bytes})"
                )
            return candidate.read_text()

    # Exhausted roots. Determine why.
    for root in allowed_roots:
        root_resolved = Path(root).resolve()
        candidate = (Path(root) / rel_path).resolve()
        try:
            candidate.relative_to(root_resolved)
            # In range but missing file
            raise ReadFileError(
                f"{rel_path}: not found under any allowed root"
            )
        except ValueError:
            continue
    raise ReadFileError(
        f"{rel_path}: outside allowed roots "
        f"{[str(r) for r in allowed_roots]}"
    )
