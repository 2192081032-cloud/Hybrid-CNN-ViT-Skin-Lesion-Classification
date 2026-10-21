"""Read existing splits and validate their integrity without rewriting them."""
from pathlib import Path
import pandas as pd

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DEFAULT_DATA_ROOT = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
DEFAULT_SPLIT_ROOT = "/kaggle/working/splits"


def _require_strings(frame, column, source):
    if frame[column].isna().any() or frame[column].str.strip().eq("").any():
        raise ValueError(f"{source}: {column} contains missing or empty values")


def validate_splits(data_root=DEFAULT_DATA_ROOT, split_root=DEFAULT_SPLIT_ROOT):
    """Return validated frames and counts, preserving CSV row order and membership.

    CSVs require image_id; dx (or label) and lesion_id may instead be joined
    from HAM10000_metadata.csv. Optional image_path paths are absolute or relative
    to data_root. Otherwise images are found recursively by image_id stem.
    Ambiguous image IDs, duplicate images, unreadable images, and cross-split
    lesion overlap are errors. No files are modified.
    """
    from PIL import Image

    data_root, split_root = Path(data_root), Path(split_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    index = {}
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index.setdefault(path.stem, []).append(path.resolve())
    metadata_path = data_root / "HAM10000_metadata.csv"
    metadata = None
    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path, dtype=str)
        if not {"image_id", "dx", "lesion_id"}.issubset(metadata.columns):
            raise ValueError("Metadata requires image_id, dx, and lesion_id")
        for column in ("image_id", "dx", "lesion_id"):
            _require_strings(metadata, column, metadata_path)
        if metadata.image_id.duplicated().any():
            raise ValueError("Metadata contains duplicate image_id values")
        metadata = metadata.set_index("image_id")

    frames, report = {}, {}
    for name in ("train", "val", "test"):
        source = split_root / f"{name}.csv"
        frame = pd.read_csv(source, dtype=str)
        if frame.empty or "image_id" not in frame:
            raise ValueError(f"{source}: requires nonempty rows and an image_id column")
        _require_strings(frame, "image_id", source)
        if frame.image_id.duplicated().any():
            raise ValueError(f"{source}: duplicate image_id values")
        if "label" in frame and "dx" not in frame:
            frame = frame.rename(columns={"label": "dx"})
        if "label" in frame and not frame.label.equals(frame.dx):
            raise ValueError(f"{source}: label and dx disagree")
        for column in ("dx", "lesion_id"):
            if metadata is not None:
                expected = frame.image_id.map(metadata[column])
                if expected.isna().any():
                    raise ValueError(f"{source}: image_id missing from metadata")
                if column in frame and not frame[column].equals(expected):
                    raise ValueError(f"{source}: {column} disagrees with metadata")
                frame[column] = expected
            elif column not in frame:
                raise ValueError(f"{source}: requires {column} or HAM10000_metadata.csv")
            _require_strings(frame, column, source)
        invalid = set(frame.dx) - set(CLASS_NAMES)
        if invalid:
            raise ValueError(f"{source}: unknown class labels: {sorted(invalid)}")
        paths = []
        for _, row in frame.iterrows():
            if "image_path" in frame:
                raw = row.image_path
                if pd.isna(raw) or not raw.strip():
                    raise ValueError(f"{source}: empty image_path")
                path = Path(raw)
                path = (path if path.is_absolute() else data_root / path).resolve()
                if path.stem != row.image_id:
                    raise ValueError(f"{source}: image_path and image_id disagree: {row.image_id}")
            else:
                candidates = index.get(row.image_id, [])
                if len(candidates) != 1:
                    raise ValueError(f"{source}: expected one image for {row.image_id}, found {len(candidates)}")
                path = candidates[0]
            if not path.is_file():
                raise FileNotFoundError(f"{source}: image does not exist: {path}")
            try:
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except Exception as exc:
                raise ValueError(f"{source}: unreadable image: {path}") from exc
            paths.append(str(path))
        frame["image_path"] = paths
        if frame.image_path.duplicated().any():
            raise ValueError(f"{source}: duplicate resolved image paths")
        frame["target"] = frame.dx.map(CLASS_TO_INDEX).astype(int)
        frames[name] = frame
        report[name] = {"images": len(frame), "lesions": frame.lesion_id.nunique(),
                        "class_counts": frame.dx.value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for column in ("image_id", "image_path", "lesion_id"):
            overlap = set(frames[left][column]) & set(frames[right][column])
            if overlap:
                raise ValueError(f"Data leakage between {left} and {right}: {column}: {sorted(overlap)[:5]}")
    # A lesion can have multiple images, but must have one diagnosis.
    combined = pd.concat(frames.values(), ignore_index=True)
    if combined.groupby("lesion_id").dx.nunique().gt(1).any():
        raise ValueError("A lesion_id has inconsistent class labels")
    return frames, report
