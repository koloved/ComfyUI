"""Unit tests for app.assets.helpers."""

from app.assets.helpers import expand_bucket_prefixes


class TestExpandBucketPrefixes:
    def test_flat_category_unchanged(self):
        # `checkpoints` is already a standalone token, no expansion needed.
        assert expand_bucket_prefixes(["models", "checkpoints"]) == [
            "models",
            "checkpoints",
        ]

    def test_nested_category_inserts_bucket(self):
        # Path-derived shape for `models/checkpoints/flux/foo.safetensors` —
        # the standalone bucket has to be present so the FE set-membership
        # filter (`include_tags=models,checkpoints`) matches the asset.
        assert expand_bucket_prefixes(["models", "checkpoints/flux"]) == [
            "models",
            "checkpoints/flux",
            "checkpoints",
        ]

    def test_deeply_nested_only_first_segment_expands(self):
        # Only the FIRST slash segment ever gets emitted as a standalone —
        # intermediate path segments don't have routing significance.
        assert expand_bucket_prefixes(
            ["models", "diffusers/kolors/text_encoder"]
        ) == ["models", "diffusers/kolors/text_encoder", "diffusers"]

    def test_unknown_prefix_does_not_expand(self):
        # Free-form user labels with slashes whose first segment is not a
        # registered bucket pass through opaquely.
        assert expand_bucket_prefixes(["models", "my-org/team-a"]) == [
            "models",
            "my-org/team-a",
        ]

    def test_idempotent(self):
        # Re-applying the helper is a no-op once the bucket is in the set.
        expanded = expand_bucket_prefixes(["models", "checkpoints/flux"])
        assert expand_bucket_prefixes(expanded) == expanded

    def test_does_not_duplicate_existing_bucket(self):
        # If the caller already supplied the standalone bucket, don't add a
        # second copy.
        assert expand_bucket_prefixes(
            ["models", "checkpoints/flux", "checkpoints"]
        ) == ["models", "checkpoints/flux", "checkpoints"]

    def test_preserves_caller_order(self):
        # User tags after path tags must stay after; the inserted bucket
        # token slots in immediately after its slash-joined parent so the
        # microsecond stagger lands it at path-tier before user-tier.
        assert expand_bucket_prefixes(
            ["models", "loras/style", "favorite", "v2"]
        ) == ["models", "loras/style", "loras", "favorite", "v2"]

    def test_empty_input(self):
        assert expand_bucket_prefixes([]) == []

    def test_input_root_with_subpath_no_expansion(self):
        # `portraits` isn't a registered model category, so the input
        # subpath stays opaque (FE filter doesn't have a checkpoint-loader
        # analogue for input subfolders).
        assert expand_bucket_prefixes(["input", "portraits/2026"]) == [
            "input",
            "portraits/2026",
        ]
