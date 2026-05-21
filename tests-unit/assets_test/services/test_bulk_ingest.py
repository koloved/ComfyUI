"""Tests for bulk ingest services."""

from pathlib import Path

from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference, AssetReferenceTag
from app.assets.services.bulk_ingest import SeedAssetSpec, batch_insert_seed_assets


class TestBatchInsertSeedAssets:
    def test_populates_mime_type_for_model_files(self, session: Session, temp_dir: Path):
        """Verify mime_type is stored in the Asset table for model files."""
        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"fake safetensors content")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 24,
                "mtime_ns": 1234567890000000000,
                "info_name": "Test Model",
                "tags": ["models"],
                "fname": "model.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            }
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1

        # Verify Asset has mime_type populated
        assets = session.query(Asset).all()
        assert len(assets) == 1
        assert assets[0].mime_type == "application/safetensors"

    def test_mime_type_none_when_not_provided(self, session: Session, temp_dir: Path):
        """Verify mime_type is None when not provided in spec."""
        file_path = temp_dir / "unknown.bin"
        file_path.write_bytes(b"binary data")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 11,
                "mtime_ns": 1234567890000000000,
                "info_name": "Unknown File",
                "tags": [],
                "fname": "unknown.bin",
                "metadata": None,
                "hash": None,
                "mime_type": None,
            }
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1

        assets = session.query(Asset).all()
        assert len(assets) == 1
        assert assets[0].mime_type is None

    def test_various_model_mime_types(self, session: Session, temp_dir: Path):
        """Verify various model file types get correct mime_type."""
        test_cases = [
            ("model.safetensors", "application/safetensors"),
            ("model.pt", "application/pytorch"),
            ("model.ckpt", "application/pickle"),
            ("model.gguf", "application/gguf"),
        ]

        specs: list[SeedAssetSpec] = []
        for filename, mime_type in test_cases:
            file_path = temp_dir / filename
            file_path.write_bytes(b"content")
            specs.append(
                {
                    "abs_path": str(file_path),
                    "size_bytes": 7,
                    "mtime_ns": 1234567890000000000,
                    "info_name": filename,
                    "tags": [],
                    "fname": filename,
                    "metadata": None,
                    "hash": None,
                    "mime_type": mime_type,
                }
            )

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == len(test_cases)

        for filename, expected_mime in test_cases:
            ref = session.query(AssetReference).filter_by(name=filename).first()
            assert ref is not None
            asset = session.query(Asset).filter_by(id=ref.asset_id).first()
            assert asset.mime_type == expected_mime, f"Expected {expected_mime} for {filename}, got {asset.mime_type}"


class TestBucketPrefixExpansionOnIngest:
    """Path-scanning ingest must persist the standalone bucket token for
    nested category paths so the FE set-membership filter
    (`include_tags=models,checkpoints`) matches assets organized into
    subfolders (`models/checkpoints/flux/foo.safetensors`).
    """

    def test_nested_path_inserts_standalone_bucket(
        self, session: Session, temp_dir: Path
    ):
        file_path = temp_dir / "flux.safetensors"
        file_path.write_bytes(b"content")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 7,
                "mtime_ns": 1234567890000000000,
                "info_name": "flux",
                # Shape emitted by get_name_and_tags_from_asset_path for a
                # nested model path.
                "tags": ["models", "checkpoints/flux"],
                "fname": "flux.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            }
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1
        ref = session.query(AssetReference).filter_by(name="flux").one()
        stored = [
            row.tag_name
            for row in session.query(AssetReferenceTag)
            .filter_by(asset_reference_id=ref.id)
            .order_by(AssetReferenceTag.added_at.asc())
            .all()
        ]
        assert stored == ["models", "checkpoints/flux", "checkpoints"]

    def test_flat_path_remains_two_tags(
        self, session: Session, temp_dir: Path
    ):
        file_path = temp_dir / "vanilla.safetensors"
        file_path.write_bytes(b"content")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 7,
                "mtime_ns": 1234567890000000000,
                "info_name": "vanilla",
                "tags": ["models", "checkpoints"],
                "fname": "vanilla.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            }
        ]

        batch_insert_seed_assets(session, specs=specs, owner_id="")

        ref = session.query(AssetReference).filter_by(name="vanilla").one()
        stored = {
            row.tag_name
            for row in session.query(AssetReferenceTag)
            .filter_by(asset_reference_id=ref.id)
            .all()
        }
        # Dedupe means flat layouts don't pick up a redundant `checkpoints`
        # row — tag[1] already serves both positional and set-membership.
        assert stored == {"models", "checkpoints"}


class TestMetadataExtraction:
    def test_extracts_mime_type_for_model_files(self, temp_dir: Path):
        """Verify metadata extraction returns correct mime_type for model files."""
        from app.assets.services.metadata_extract import extract_file_metadata

        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"fake safetensors content")

        meta = extract_file_metadata(str(file_path))

        assert meta.content_type == "application/safetensors"

    def test_mime_type_for_various_model_formats(self, temp_dir: Path):
        """Verify various model file types get correct mime_type from metadata."""
        from app.assets.services.metadata_extract import extract_file_metadata

        test_cases = [
            ("model.safetensors", "application/safetensors"),
            ("model.sft", "application/safetensors"),
            ("model.pt", "application/pytorch"),
            ("model.pth", "application/pytorch"),
            ("model.ckpt", "application/pickle"),
            ("model.pkl", "application/pickle"),
            ("model.gguf", "application/gguf"),
        ]

        for filename, expected_mime in test_cases:
            file_path = temp_dir / filename
            file_path.write_bytes(b"content")

            meta = extract_file_metadata(str(file_path))

            assert meta.content_type == expected_mime, f"Expected {expected_mime} for {filename}, got {meta.content_type}"
