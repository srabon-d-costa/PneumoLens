

"""PneumoLens Phase 4 V2: complete, bounded near-duplicate screening.

This module does NOT establish patient independence without trusted provenance.
It only constructs a candidate dev split while keeping the original test locked.
For research/education, not clinical use.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ("NORMAL", "PNEUMONIA")
CLASS_TO_INDEX = {"NORMAL": 0, "PNEUMONIA": 1}
SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}
FILENAME_RE = re.compile(r"^(NORMAL|BACTERIA|VIRUS)-(\d+)-(\d+)\.(jpe?g|png)$", re.I)


@dataclass
class Source:
    kind: str
    location: Path
    zip_prefix: str = ""
    _archive: ZipFile | None = field(default=None, repr=False, compare=False)

    def iter_images(self):
        if self.kind == "folder":
            for original_split in ("train", "test"):
                for label in LABELS:
                    for p in sorted((self.location / original_split / label).iterdir()):
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                            yield original_split, label, p.relative_to(self.location).as_posix()
        elif self.kind == "zip":
            with ZipFile(self.location) as archive:
                members = sorted(n for n in archive.namelist() if not n.endswith("/"))
            for member in members:
                if not member.startswith(self.zip_prefix):
                    continue
                rel = member[len(self.zip_prefix):]
                parts = rel.split("/")
                if (len(parts) == 3 and parts[0].lower() in ("train", "test")
                        and parts[1].upper() in LABELS and Path(parts[2]).suffix.lower() in SUPPORTED_EXTENSIONS):
                    yield parts[0].lower(), parts[1].upper(), "/".join(parts)
        else:
            raise ValueError(f"Unsupported source kind: {self.kind}")

    def read(self, relative_path: str) -> bytes:
        if self.kind == "folder":
            p = (self.location / relative_path).resolve()
            if not p.is_relative_to(self.location.resolve()):
                raise ValueError("Unsafe relative path outside data source")
            return p.read_bytes()
        if self._archive is None:
            self._archive = ZipFile(self.location)
        return self._archive.read(self.zip_prefix + relative_path)


def _valid_root(path: Path) -> bool:
    return all((path / sp / lab).is_dir() for sp in ("train", "test") for lab in LABELS)


def _zip_prefixes(archive_path: Path) -> list[str]:
    try:
        with ZipFile(archive_path) as archive:
            names = [n.replace("\\", "/") for n in archive.namelist()]
    except Exception:
        return []
    patterns = defaultdict(set)
    for name in names:
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        m = re.match(r"^(.*?/)?(train|test)/(NORMAL|PNEUMONIA)/[^/]+$", name, re.I)
        if m:
            patterns[m.group(1) or ""].add((m.group(2).lower(), m.group(3).upper()))
    required = {(sp, lab) for sp in ("train", "test") for lab in LABELS}
    return sorted(p for p, present in patterns.items() if required.issubset(present))


def discover_source(explicit: str | None = None, search_root: str = "/kaggle/input") -> Source:
    """Reject ambiguous or missing datasets. No silent OCT fallback."""
    preferred = Path("/kaggle/input/datasets/srabondcosta/pneumonia/ZhangLabData/CellData/chest_xray")
    override = explicit or os.environ.get("PNEUMOLENS_DATASET_ROOT", "")
    if override:
        path = Path(override)
        if path.is_dir() and _valid_root(path):
            return Source("folder", path)
        if path.is_file() and path.suffix.lower() == ".zip":
            options = _zip_prefixes(path)
            if len(options) == 1:
                return Source("zip", path, options[0])
        raise FileNotFoundError(f"Explicit source is not a unique chest_xray train/test dataset: {override}")
    if _valid_root(preferred):
        return Source("folder", preferred)
    base = Path(search_root)
    if not base.exists():
        raise FileNotFoundError(f"Dataset root unavailable: {base}")
    folders = []
    archives = []
    for dirpath, dirs, filenames in os.walk(base):
        current = Path(dirpath)
        if _valid_root(current):
            folders.append(current)
            dirs[:] = []
            continue
        # Avoid traversing the 100k+ OCT files in mixed ZhangLab archives.
        dirs[:] = [x for x in dirs if x.lower() not in {"oct", "optical_coherence_tomography", "__macosx", ".git"}]
        archives.extend(current / fn for fn in filenames if fn.lower().endswith(".zip"))
    if len(folders) == 1:
        return Source("folder", folders[0])
    if len(folders) > 1:
        raise RuntimeError(f"Multiple chest_xray sources found. Set PNEUMOLENS_DATASET_ROOT: {folders[:4]}")
    candidates = [(f, prefix) for f in archives for prefix in _zip_prefixes(f)]
    if len(candidates) == 1:
        return Source("zip", candidates[0][0], candidates[0][1])
    raise RuntimeError(f"Expected one chest X-ray folder or ZIP, found {len(candidates)} options")


def series_candidate(filename: str) -> tuple[str, bool]:
    """Weak same-series key only. Prefix+number is NOT a patient identifier."""
    m = FILENAME_RE.match(filename)
    if m:
        return f"filename-series:{m.group(1).upper()}-{m.group(2)}", True
    return f"unparsed-filename:{filename}", False


def dct_basis(size: int = 32) -> np.ndarray:
    a = np.arange(size, dtype=np.float64)
    k = a.reshape(-1, 1)
    basis = np.cos(np.pi * (2 * a + 1) * k / (2 * size))
    basis[0, :] *= np.sqrt(1 / size)
    basis[1:, :] *= np.sqrt(2 / size)
    return basis


_DCT32 = dct_basis(32)


def perceptual_hash(gray_thumbnail: np.ndarray) -> int:
    """64-bit pHash (DCT), independent of Phase 1's collision-prone dHash."""
    arr = np.asarray(Image.fromarray(gray_thumbnail, "L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float64)
    coeff = _DCT32 @ arr @ _DCT32.T
    block = coeff[:8, :8].copy()
    median = np.median(block.flatten()[1:])
    bits = (block > median).flatten()
    bits[0] = False
    return sum(1 << i for i, bit in enumerate(bits) if bit)


def analyze_source(source: Source, fail_fast: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Single read/decode per image, secure hashes and compact private image signatures."""
    rows: list[dict] = []
    thumbs_by_path = {}
    try:
        from tqdm.auto import tqdm
        images = tqdm(list(source.iter_images()), desc="Building source inventory (first-run fallback)", unit="image")
    except ImportError:
        images = source.iter_images()
    for original_split, label, rel in images:
        filename = Path(rel).name
        series_key, parsed = series_candidate(filename)
        try:
            raw = source.read(rel)
            file_sha = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as image:
                transposed = ImageOps.exif_transpose(image)
                gray = transposed.convert("L")
                width, height = gray.size
                pixel_sha = hashlib.sha256(f"{width}x{height}:".encode("ascii") + gray.tobytes()).hexdigest()
                thumb = np.asarray(gray.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.uint8)
                image_mode = image.mode
            rows.append({
                "relative_path": rel, "filename": filename, "original_split": original_split,
                "label": label, "label_index": CLASS_TO_INDEX[label],
                "file_sha256": file_sha, "decoded_sha256": pixel_sha,
                "phash64": f"{perceptual_hash(thumb):016x}",
                "series_group_UNVERIFIED": series_key, "filename_series_parse_ok": parsed,
                "width": width, "height": height, "aspect_ratio": width / height,
                "image_mode": image_mode, "bytes": len(raw), "decode_ok": True,
            })
            thumbs_by_path[rel] = thumb
        except Exception as exc:
            if fail_fast:
                raise RuntimeError(f"Unable to inspect {rel}: {type(exc).__name__}: {exc}") from exc
            rows.append({"relative_path": rel, "filename": filename, "original_split": original_split,
                         "label": label, "decode_ok": False, "error": str(exc)})
    manifest = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No supported chest X-rays found")
    if not manifest.decode_ok.all():
        raise RuntimeError("Integrity failed: unreadable images detected")
    # Thumbnails must use exactly the same row order as the sorted manifest.
    if not manifest["relative_path"].is_unique:
        raise RuntimeError("Duplicate relative paths detected")
    thumb_stack = np.stack([thumbs_by_path[rel] for rel in manifest.relative_path])
    if source._archive is not None:
        source._archive.close()
        source._archive = None
    return manifest, thumb_stack


def duplicate_violations(df: pd.DataFrame) -> dict:
    out = {}
    for key in ("file_sha256", "decoded_sha256"):
        grouped = df.groupby(key, dropna=False)
        out[f"{key}_cross_original_split_groups"] = int((grouped["original_split"].nunique() > 1).sum())
        out[f"{key}_conflicting_label_groups"] = int((grouped["label"].nunique() > 1).sum())
        out[f"{key}_duplicate_groups"] = int((grouped.size() > 1).sum())
    return out


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def merge(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1



def exhaustive_phash_candidates(df: pd.DataFrame, max_distance: int = 6,
                                block_size: int = 128) -> tuple[pd.DataFrame, dict]:
    """Complete 64-bit pHash radius search: O(n^2) time, O(block*n) memory.

    Every unordered pair is evaluated; there is NO band-hash budget, sampling,
    early termination or implicit guarantee that pHash discovers all duplicates.
    """
    if not (0 <= max_distance <= 64):
        raise ValueError("max_distance must be in [0, 64]")
    if not 1 <= block_size <= 1024:
        raise ValueError("block_size must be in [1, 1024]")
    values = np.asarray([int(str(x), 16) for x in df.phash64], dtype=np.uint64)
    n = len(values)
    lookup = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    records = []
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        # Only comparisons where the other index > start are materialized.
        xors = np.bitwise_xor(values[start:stop, None], values[None, start + 1:])
        byte_view = np.ascontiguousarray(xors).view(np.uint8).reshape(stop - start, n - start - 1, 8)
        distance = np.zeros(xors.shape, dtype=np.uint8)
        for offset in range(8):
            distance += lookup[byte_view[:, :, offset]]
        a, b = np.where((distance <= max_distance) &
                        (np.arange(start, stop)[:, None] < np.arange(start + 1, n)[None, :]))
        records.extend((int(start + i), int(start + 1 + j), int(distance[i, j]))
                       for i, j in zip(a, b))
    result = pd.DataFrame(records, columns=["index_a", "index_b", "phash_hamming"])
    if not result.empty:
        result = result.sort_values(["phash_hamming", "index_a", "index_b"]).reset_index(drop=True)
    diagnostics = {
        "total_image_pairs_compared": n * (n - 1) // 2,
        "phash_candidates": len(result),
        "phash_distance_threshold": int(max_distance),
        "complete_exhaustive_phash_search": True,
        "screen_incomplete_due_to_resource_budget": False,
        "method": "All-pairs vectorized 64-bit pHash Hamming distance; no LSH cap",
        "limitation": "pHash radius does not guarantee detection of every visually similar radiograph or shared patient",
    }
    return result, diagnostics


def load_candidate_thumbnails(source: Source, df: pd.DataFrame,
                              candidates: pd.DataFrame,
                              full_thumbnails: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Read each candidate image ONCE, verifying original bytes against cached inventory."""
    if candidates.empty:
        return {}
    indices = sorted(set(candidates.index_a.astype(int)) | set(candidates.index_b.astype(int)))
    if full_thumbnails is not None:
        if len(full_thumbnails) != len(df):
            raise ValueError("Full thumbnail array is not aligned with the cached inventory")
        return {idx: full_thumbnails[idx] for idx in indices}
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(indices, desc="Checking only pHash candidate X-rays", unit="image")
    except ImportError:
        iterator = indices
    cache: dict[int, np.ndarray] = {}
    for idx in iterator:
        row = df.iloc[idx]
        raw = source.read(str(row.relative_path))
        sha = hashlib.sha256(raw).hexdigest()
        if sha != row.file_sha256:
            raise RuntimeError(f"Source bytes changed since Phase 4 v1 for {row.relative_path}")
        with Image.open(io.BytesIO(raw)) as image:
            gray = ImageOps.exif_transpose(image).convert("L")
            thumb = np.asarray(gray.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.uint8)
        if f"{perceptual_hash(thumb):016x}" != row.phash64:
            raise RuntimeError(f"pHash changed since Phase 4 v1 for {row.relative_path}")
        cache[idx] = thumb
    if source._archive is not None:
        source._archive.close()
        source._archive = None
    return cache


def score_phash_candidates(df: pd.DataFrame, candidates: pd.DataFrame,
                           thumbs: dict[int, np.ndarray], policy: dict) -> pd.DataFrame:
    """Record EVERY pHash candidate and its measured pixel-structure evidence.

    HIGH / REVIEW are heuristic manual-review flags, never confirmed duplicates.
    LOW_SCREENED means the chosen image-structure cutoffs were not met.
    """
    p = policy["near_duplicate_screen"]
    records = []
    for a, b, hd in candidates[["index_a", "index_b", "phash_hamming"]].itertuples(index=False, name=None):
        ra, rb = df.iloc[int(a)], df.iloc[int(b)]
        exact = (ra.file_sha256 == rb.file_sha256 or ra.decoded_sha256 == rb.decoded_sha256)
        aspect = abs(float(ra.aspect_ratio) - float(rb.aspect_ratio)) / max(float(ra.aspect_ratio), float(rb.aspect_ratio))
        corr, grad, mae = pair_similarity(thumbs[int(a)], thumbs[int(b)])
        high = (not exact and hd <= int(p["strong_phash_max"]) and
                corr >= float(p["strong_corr_min"]) and
                grad >= float(p["strong_gradient_min"]) and
                mae <= float(p["strong_mae_max"]) and
                aspect <= float(p["strong_aspect_max"]))
        review = (not exact and corr >= float(p["review_corr_min"]) and
                  grad >= float(p["review_gradient_min"]) and
                  aspect <= float(p["review_aspect_max"]))
        priority = ("EXACT_KNOWN" if exact else "HIGH" if high else "REVIEW" if review else "LOW_SCREENED")
        records.append({
            "index_a": int(a), "index_b": int(b), "path_a": str(ra.relative_path), "path_b": str(rb.relative_path),
            "label_a": ra.label, "label_b": rb.label, "source_split_a": ra.original_split,
            "source_split_b": rb.original_split, "phash_hamming": int(hd),
            "pixel_corr": round(float(corr), 7), "gradient_corr": round(float(grad), 7),
            "pixel_mae_fraction": round(float(mae), 7), "aspect_relative_delta": round(float(aspect), 7),
            "screening_priority": priority, "confirmed_duplicate": bool(exact),
            "human_decision": "EXACT_ALREADY_HANDLED" if exact else "PENDING_REVIEW" if priority in ("HIGH", "REVIEW") else "NOT_FLAGGED_BY_SCREEN",
        })
    cols = ["index_a", "index_b", "path_a", "path_b", "label_a", "label_b",
            "source_split_a", "source_split_b", "phash_hamming", "pixel_corr",
            "gradient_corr", "pixel_mae_fraction", "aspect_relative_delta",
            "screening_priority", "confirmed_duplicate", "human_decision"]
    frame = pd.DataFrame.from_records(records, columns=cols)
    if not frame.empty:
        order = {"HIGH": 0, "REVIEW": 1, "LOW_SCREENED": 2, "EXACT_KNOWN": 3}
        frame["_priority_rank"] = frame.screening_priority.map(order)
        frame = frame.sort_values(["_priority_rank", "phash_hamming", "pixel_corr", "index_a", "index_b"],
                                  ascending=[True, True, False, True, True]).drop(columns=["_priority_rank"]).reset_index(drop=True)
    return frame


def action_review_queue(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[scored.screening_priority.isin(["HIGH", "REVIEW"])].copy().reset_index(drop=True)


def screen_image_similarity(df: pd.DataFrame, thumbs: np.ndarray, policy: dict):
    """Backwards-compatible helper for unit tests with resident thumbnails."""
    candidates, diagnostics = exhaustive_phash_candidates(df, int(policy["near_duplicate_screen"]["max_phash_distance"]))
    cache = load_candidate_thumbnails(None, df, candidates, full_thumbnails=thumbs) if len(candidates) else {}
    full = score_phash_candidates(df, candidates, cache, policy)
    queue = action_review_queue(full)
    diagnostics.update({"phash_screened_pairs": len(full), "review_queue_pairs": len(queue),
                        "high_priority_pairs": int((queue.screening_priority == "HIGH").sum()),
                        "all_phash_candidates_image_scored": len(full) == len(candidates)})
    return queue, diagnostics

def _normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float32).ravel()
    y = b.astype(np.float32).ravel()
    x -= x.mean(); y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator < 1e-9:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def pair_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    corr = _normalized_corr(a, b)
    dx_a = np.diff(a.astype(np.float32), axis=1)
    dx_b = np.diff(b.astype(np.float32), axis=1)
    dy_a = np.diff(a.astype(np.float32), axis=0)
    dy_b = np.diff(b.astype(np.float32), axis=0)
    grad_corr = (_normalized_corr(dx_a, dx_b) + _normalized_corr(dy_a, dy_b)) / 2
    mae = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255.0)
    return corr, grad_corr, mae


def _join_duplicates(ds: DisjointSet, df: pd.DataFrame, keys: tuple[str, ...]):
    for key in keys:
        for _, group in df.groupby(key, sort=True):
            members = group.index.to_list()
            for j in members[1:]:
                ds.merge(members[0], j)


def build_split(df: pd.DataFrame, pairs: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, dict]:
    """Never uses original test rows in selecting candidate validation fold."""
    df = df.copy().reset_index(drop=True)
    # Stable order required for hashes, indices and reproducibility.
    if not df.relative_path.is_monotonic_increasing:
        raise ValueError("Manifest must be sorted by relative_path before splitting")
    train = df.index[df.original_split.eq("train")].to_list()
    test = df.index[df.original_split.eq("test")].to_list()
    ds = DisjointSet(len(df))
    _join_duplicates(ds, df, ("file_sha256", "decoded_sha256", "series_group_UNVERIFIED"))
    if not pairs.empty:
        for row in pairs.itertuples(index=False):
            if row.screening_priority == "HIGH" and row.source_split_a == row.source_split_b == "train":
                ds.merge(int(row.index_a), int(row.index_b))
    df["candidate_group"] = [f"G{ds.find(i):06d}" for i in range(len(df))]
    # Deduplicate only original train; original test is never reduced or reallocated.
    df["split_role"] = "TEST_LOCKED"
    df["duplicate_of"] = ""
    df.loc[train, "split_role"] = "UNASSIGNED_TRAIN_SOURCE"
    for _, grp in df.loc[train].groupby("decoded_sha256", sort=True):
        members = list(grp.index)
        keeper = min(members, key=lambda i: df.at[i, "relative_path"])
        for idx in members:
            if idx != keeper:
                df.at[idx, "split_role"] = "EXCLUDED_EXACT_COPY"
                df.at[idx, "duplicate_of"] = df.at[keeper, "relative_path"]
    # Cross-encoded file-byte exact copies may also have the same pixel hash;
    # decoded_sha256 is the deduplication criterion.
    retained = df.loc[df.split_role.eq("UNASSIGNED_TRAIN_SOURCE")]
    groups = retained.candidate_group.to_numpy()
    y = retained.label_index.to_numpy(dtype=np.int64)
    if len(set(groups)) < 5 or set(y) != {0, 1}:
        raise RuntimeError("Too few groups or missing classes for 5-fold split")
    n_splits = int(policy["validation"]["n_splits"])
    choices = []
    for seed in policy["validation"]["candidate_seeds"]:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold, (fit_ix, val_ix) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            yp = y[val_ix]
            if len(set(yp)) != 2 or len(set(y[fit_ix])) != 2:
                continue
            fraction = len(val_ix) / len(y)
            val_p = yp.mean()
            global_p = y.mean()
            fit_p = y[fit_ix].mean()
            score = 3 * abs(fraction - 1 / n_splits) + 2 * abs(val_p - global_p) + abs(fit_p - global_p)
            choices.append((score, int(seed), int(fold), fit_ix, val_ix))
    if not choices:
        raise RuntimeError("No grouped split retained both classes")
    best = min(choices, key=lambda c: (round(c[0], 12), c[1], c[2]))
    retained_idxs = retained.index.to_numpy()
    df.loc[retained_idxs[best[3]], "split_role"] = "TRAIN"
    df.loc[retained_idxs[best[4]], "split_role"] = "VALIDATION"
    assigned = df.split_role.value_counts().to_dict()
    train_groups = set(df.loc[df.split_role.eq("TRAIN"), "candidate_group"])
    val_groups = set(df.loc[df.split_role.eq("VALIDATION"), "candidate_group"])
    test_groups = set(df.loc[df.split_role.eq("TEST_LOCKED"), "candidate_group"])
    train_hashes = set(df.loc[df.split_role.eq("TRAIN"), "decoded_sha256"])
    val_hashes = set(df.loc[df.split_role.eq("VALIDATION"), "decoded_sha256"])
    test_hashes = set(df.loc[df.split_role.eq("TEST_LOCKED"), "decoded_sha256"])
    original_test = set(df.loc[df.original_split.eq("test"), "relative_path"])
    output_test = set(df.loc[df.split_role.eq("TEST_LOCKED"), "relative_path"])
    violations = duplicate_violations(df)
    cross_source_series = int((df.groupby("series_group_UNVERIFIED").original_split.nunique() > 1).sum())
    qa = {
        "all_rows_accounted_for": int(sum(assigned.values())) == len(df),
        "original_test_exactly_preserved": original_test == output_test,
        "train_validation_groups_disjoint": train_groups.isdisjoint(val_groups),
        "train_test_heuristic_groups_disjoint": train_groups.isdisjoint(test_groups),
        "validation_test_heuristic_groups_disjoint": val_groups.isdisjoint(test_groups),
        "train_validation_pixel_sha_disjoint": train_hashes.isdisjoint(val_hashes),
        "train_test_pixel_sha_disjoint": train_hashes.isdisjoint(test_hashes),
        "validation_test_pixel_sha_disjoint": val_hashes.isdisjoint(test_hashes),
        "no_pixel_sha_label_conflicts": violations["decoded_sha256_conflicting_label_groups"] == 0,
        "no_file_sha_label_conflicts": violations["file_sha256_conflicting_label_groups"] == 0,
        "no_heuristic_series_cross_original_test": cross_source_series == 0,
        "all_original_train_accounted_for": int((df.original_split == "train").sum()) == int((df.split_role != "TEST_LOCKED").sum()),
        "both_fit_classes_present": set(df.loc[df.split_role.eq("TRAIN"), "label"]) == set(LABELS),
        "both_validation_classes_present": set(df.loc[df.split_role.eq("VALIDATION"), "label"]) == set(LABELS),
    }
    retained_val = df[df.split_role == "VALIDATION"]
    retained_fit = df[df.split_role == "TRAIN"]
    source_p = float(retained.label_index.mean())
    val_p = float(retained_val.label_index.mean())
    val_fraction = len(retained_val) / len(retained)
    qa["validation_fraction_within_declared_tolerance"] = (
        abs(val_fraction - float(policy["validation"]["target_fraction"])) <=
        float(policy["validation"]["max_abs_fraction_deviation"]))
    qa["validation_class_rate_within_declared_tolerance"] = (
        abs(val_p - source_p) <=
        float(policy["validation"]["max_abs_positive_rate_deviation"]))
    qa["no_high_priority_cross_original_source"] = True if pairs.empty else not any(
        (pairs.screening_priority == "HIGH") & (pairs.source_split_a != pairs.source_split_b))
    qa["no_high_priority_cross_train_val"] = True
    qa["pending_cross_partition_review_pairs"] = 0
    if not pairs.empty:
        role_map = df.split_role.to_dict()
        split_roles = []
        for row in pairs.itertuples(index=False):
            a, b = role_map[int(row.index_a)], role_map[int(row.index_b)]
            roles = {a, b}
            crosses = (a != b and not roles.issubset({"TRAIN", "EXCLUDED_EXACT_COPY"})
                       and not roles.issubset({"VALIDATION", "EXCLUDED_EXACT_COPY"}))
            split_roles.append((a, b, crosses))
        pairs["role_a"] = [r[0] for r in split_roles]
        pairs["role_b"] = [r[1] for r in split_roles]
        pairs["cross_candidate_partitions"] = [r[2] for r in split_roles]
        qa["no_high_priority_cross_train_val"] = not any(
            (pairs.screening_priority == "HIGH") &
            (pairs.role_a.isin(["TRAIN", "VALIDATION"])) &
            (pairs.role_b.isin(["TRAIN", "VALIDATION"])) &
            (pairs.role_a != pairs.role_b))
        qa["pending_cross_partition_review_pairs"] = int(pairs.cross_candidate_partitions.sum())
    assigned = df.split_role.value_counts().to_dict()
    locked_test_exact_group_sizes = (df[df.original_split.eq("test")]
                                     .groupby("decoded_sha256").size())
    locked_test_redundant_exact_copies = int((locked_test_exact_group_sizes - 1).clip(lower=0).sum())
    counts = df[df.split_role.isin(["TRAIN", "VALIDATION"])].groupby(["split_role", "label"]).size().to_dict()
    c0, c1 = int(counts.get(("TRAIN", "NORMAL"), 0)), int(counts.get(("TRAIN", "PNEUMONIA"), 0))
    weights = {"NORMAL": (c0 + c1) / (2 * c0), "PNEUMONIA": (c0 + c1) / (2 * c1)}
    diagnostics = {
        "selected_seed": best[1], "selected_fold": best[2],
        "selected_score": float(best[0]), "tested_splits": len(choices),
        "source_count": len(df), "source_train_count": len(train), "source_test_count": len(test),
        "dedup_removed_from_original_train": int(assigned.get("EXCLUDED_EXACT_COPY", 0)),
        "locked_test_redundant_exact_copies_preserved": locked_test_redundant_exact_copies,
        "role_counts": {str(k): int(v) for k, v in assigned.items()},
        "validation_fraction_of_retained_train": float(assigned.get("VALIDATION", 0) / (assigned.get("VALIDATION", 0) + assigned.get("TRAIN", 0))),
        "class_counts": {f"{role}/{label}": int(v) for (role, label), v in counts.items()},
        "class_weights_train_only": weights, "qa": qa, "duplicate_diagnostics": violations,
        "series_groups_cross_original_split": cross_source_series,
        "verified_patient_id_count": 0,
        "patient_independence_verified": False,
    }
    return df, diagnostics


def split_fingerprint(df: pd.DataFrame) -> str:
    canonical = "\n".join(f"{r.relative_path}|{r.split_role}|{r.decoded_sha256}" for r in
                           df.sort_values("relative_path").itertuples(index=False))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def training_gate(diag: dict, screen: dict, policy: dict) -> dict:
    qa = diag["qa"]
    boolean_checks = {k: v for k, v in qa.items() if isinstance(v, bool)}
    technical_pass = (all(boolean_checks.values()) and
                      screen.get("complete_exhaustive_phash_search", False) and
                      screen.get("all_phash_candidates_image_scored", False) and
                      not screen.get("screen_incomplete_due_to_resource_budget", True))
    # 'Technical pass' alone never grants research/clinical training approval.
    return {
        "technical_split_gate": "PASS_AUTOMATED_CHECKS_WITH_LIMITATIONS" if technical_pass else "FAIL_NEEDS_INVESTIGATION",
        "research_training_gate": "HOLD_PENDING_HUMAN_REVIEW_AND_RISK_ACCEPTANCE",
        "patient_independence": "UNVERIFIED",
        "clinical_validation": "NOT_SUPPORTED",
        "why": [
            "Filename series keys are not trusted patient identifiers",
            "Any HIGH/REVIEW candidates need human adjudication; a negative pHash screen does not establish patient independence",
            "Original test is locked for model selection and threshold tuning",
        ],
        "preregistered_thresholds": policy["near_duplicate_screen"],
    }


def inventory_fingerprint(df: pd.DataFrame) -> str:
    canonical = "\n".join(f"{r.relative_path}|{r.file_sha256}" for r in
                          df.sort_values("relative_path").itertuples(index=False))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

