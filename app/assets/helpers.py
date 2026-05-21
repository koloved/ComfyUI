import os
from datetime import datetime, timezone
from typing import Sequence


def select_best_live_path(states: Sequence) -> str:
    """
    Return the best on-disk path among cache states:
      1) Prefer a path that exists with needs_verify == False (already verified).
      2) Otherwise, pick the first path that exists.
      3) Otherwise return empty string.
    """
    alive = [
        s
        for s in states
        if getattr(s, "file_path", None) and os.path.isfile(s.file_path)
    ]
    if not alive:
        return ""
    for s in alive:
        if not getattr(s, "needs_verify", False):
            return s.file_path
    return alive[0].file_path


def escape_sql_like_string(s: str, escape: str = "!") -> tuple[str, str]:
    """Escapes %, _ and the escape char in a LIKE prefix.

    Returns (escaped_prefix, escape_char).
    """
    s = s.replace(escape, escape + escape)  # escape the escape char first
    s = s.replace("%", escape + "%").replace("_", escape + "_")  # escape LIKE wildcards
    return s, escape


def get_utc_now() -> datetime:
    """Naive UTC timestamp (no tzinfo). We always treat DB datetimes as UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_tags(tags: list[str] | None) -> list[str]:
    """
    Normalize a list of tags by:
      - Stripping whitespace and converting to lowercase.
      - Removing duplicates.
    """
    return list(dict.fromkeys(t.strip().lower() for t in (tags or []) if (t or "").strip()))


def _known_bucket_prefixes() -> set[str]:
    """Lowercased model-category names eligible for standalone-prefix
    expansion. Tags whose first slash segment matches one of these get
    the bucket inserted as a separate token, so FE filters like
    ``include_tags=models,checkpoints`` keep matching even when the
    asset lives in a nested subfolder (`models/checkpoints/flux/foo`).

    Bare user labels with slashes whose first segment is not a registered
    bucket (e.g. ``my-org/team-a``) pass through unchanged.
    """
    try:
        import folder_paths

        return {
            name.lower()
            for name in folder_paths.folder_names_and_paths.keys()
            if name != "custom_nodes"
        }
    except Exception:
        return set()


def expand_bucket_prefixes(tags: list[str]) -> list[str]:
    """Insert standalone bucket tokens after any slash-joined tag whose
    first segment is a registered model category. Preserves caller order
    and is idempotent (existing bucket tokens are not duplicated).
    """
    if not tags:
        return list(tags)
    buckets = _known_bucket_prefixes()
    if not buckets:
        return list(tags)
    seen = set(tags)
    result: list[str] = []
    for t in tags:
        result.append(t)
        if "/" in t:
            prefix = t.split("/", 1)[0]
            if prefix.lower() in buckets and prefix not in seen:
                result.append(prefix)
                seen.add(prefix)
    return result


def validate_blake3_hash(s: str) -> str:
    """Validate and normalize a blake3 hash string.

    Returns canonical 'blake3:<hex>' or raises ValueError.
    """
    s = s.strip().lower()
    if not s or ":" not in s:
        raise ValueError("hash must be 'blake3:<hex>'")
    algo, digest = s.split(":", 1)
    if (
        algo != "blake3"
        or len(digest) != 64
        or any(c for c in digest if c not in "0123456789abcdef")
    ):
        raise ValueError("hash must be 'blake3:<hex>'")
    return f"{algo}:{digest}"
