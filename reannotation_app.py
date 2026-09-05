#!/usr/bin/env python3
"""
Fresh annotation and refinement tool for the 2026 supplementary re-annotation batch.

Features:
  - Read images from defect_bench/data_sample/images
  - Start from empty labels and masks, then write new ground truth
  - Support manual bbox refinement (drag, resize, delete, add)
  - Support manual mask refinement (point operations, brush operations, maintain class colors)

Usage:
  cd defect_bench
  DEFECT_BENCH_OPEN_DATASET_ROOT=/path/to/reannotation_batch python reannotation_app.py

Then access in browser through the configured re-annotation port (5010 by default).
"""

import base64
import hashlib
import io
import json
import os
import shutil
import cv2
import numpy as np
import sys
import uuid
import threading
import time
from pathlib import Path, PurePosixPath

# defect_bench path resolution
_THIS_FILE = Path(__file__).resolve()
_DEFAULT_OPEN_DATASET_ROOT = next((p / "data_sample" for p in [_THIS_FILE.parent, *_THIS_FILE.parents] if p.name == "defect_bench"), _THIS_FILE.parent)
OPEN_DATASET_ROOT = Path(os.environ.get("DEFECT_BENCH_OPEN_DATASET_ROOT", str(_DEFAULT_OPEN_DATASET_ROOT)))

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image, ImageDraw

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import oss2
except ImportError:
    oss2 = None

# Import SAM service and Detection Agent
# Import from local defect_bench modules
try:
    from sam_logic import sam_service
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: SAM service not available. Mask refinement will be disabled.")
    SAM_AVAILABLE = False
    sam_service = None

try:
    from detection_agent import detection_service
    import asyncio
    DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: Detection service not available. Detection will be disabled.")
    DETECTION_AVAILABLE = False
    detection_service = None

app = Flask(__name__)

# The final DefectBench ontology.  Every saved bbox must use one compatible pair.
PRIMARY_TO_SUBTYPES: Dict[str, List[str]] = {
    "Crack": ["Linear crack", "Map cracking"],
    "Material_loss": ["Spalling", "Peeling"],
    "Stain": ["Corrosion", "Rust stain", "Leakage stain"],
    "External Fixings": ["Vegetation growth", "Surface contaminants", "Graffiti"],
}

# Directory configuration
BASE_DIR = OPEN_DATASET_ROOT
IMAGES_DIR = BASE_DIR / "images" if (BASE_DIR / "images").exists() else BASE_DIR
LABELS_DIR = BASE_DIR / "labels"
MASKS_DIR = BASE_DIR / "masks"
CANDIDATES_CSV = BASE_DIR / "candidates.csv"

# Algorithm outputs are immutable inputs.  Existing batches that only contain
# ``labels/`` and ``masks/`` remain supported; those folders are treated as the
# algorithm baseline.  Expert edits never overwrite either input folder.
ALGORITHM_LABELS_DIR = (
    BASE_DIR / "algorithm_labels" if (BASE_DIR / "algorithm_labels").exists()
    else BASE_DIR / "detections" if (BASE_DIR / "detections").exists()
    else LABELS_DIR
)
ALGORITHM_MASKS_DIR = BASE_DIR / "algorithm_masks" if (BASE_DIR / "algorithm_masks").exists() else MASKS_DIR
EXPERT_LABELS_DIR = BASE_DIR / "expert_labels"
EXPERT_MASKS_DIR = BASE_DIR / "expert_masks"
REVISIONS_DIR = BASE_DIR / "revisions"
DECISION_EVENTS_PATH = BASE_DIR / "decision_events.jsonl"
OUTPUT_ROOT = BASE_DIR
FINAL_EXPORTS_DIR = OUTPUT_ROOT / "final_exports"
# In production this should point at a mounted persistent disk.  Keeping it
# outside the source checkout prevents uploaded images and expert edits from
# disappearing on a redeploy.
STORAGE_ROOT = Path(os.environ.get("ANNOTATION_STORAGE_ROOT", str(_THIS_FILE.parent))).resolve()
IMPORT_ROOT = STORAGE_ROOT / "imported_datasets"


class R2Storage:
    """Small S3-compatible persistence layer used by the cloud deployment.

    The application always works against a local working directory.  When R2
    is configured, imported batches and expert outputs are mirrored to R2 so a
    free/ephemeral web service can restart without losing annotation data.
    """

    def __init__(self) -> None:
        self.bucket = os.environ.get("R2_BUCKET", "").strip()
        self.prefix = os.environ.get("R2_PREFIX", "defectbench").strip("/")
        self.client = None
        self._lock = threading.Lock()
        if not self.bucket:
            return
        endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
        access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        if not (boto3 and endpoint and access_key and secret_key):
            raise RuntimeError(
                "R2_BUCKET 已设置，但缺少 boto3 或 R2_ENDPOINT_URL、R2_ACCESS_KEY_ID、R2_SECRET_ACCESS_KEY。"
            )
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def key(self, relative: str) -> str:
        clean = relative.replace("\\", "/").lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def put_file(self, local_path: Path, relative: str) -> None:
        if self.enabled:
            self.client.upload_file(str(local_path), self.bucket, self.key(relative))

    def put_bytes(self, payload: bytes, relative: str, *, content_type: str = "application/json") -> None:
        if self.enabled:
            self.client.put_object(Bucket=self.bucket, Key=self.key(relative), Body=payload, ContentType=content_type)

    def get_bytes(self, relative: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self.key(relative))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as error:
            # R2 may surface a missing object as a generic ClientError.
            if getattr(error, "response", {}).get("Error", {}).get("Code") in {"NoSuchKey", "404", "NoSuchBucket"}:
                return None
            raise

    def put_tree(self, local_root: Path, remote_root: str) -> None:
        if not self.enabled:
            return
        for path in local_root.rglob("*"):
            if path.is_file():
                self.put_file(path, f"{remote_root.rstrip('/')}/{path.relative_to(local_root).as_posix()}")

    def get_tree(self, remote_root: str, local_root: Path) -> int:
        """Download a complete prefix into the ephemeral local working cache."""
        if not self.enabled:
            return 0
        prefix = self.key(remote_root.rstrip("/") + "/")
        paginator = self.client.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                remote_key = item["Key"]
                relative = remote_key[len(prefix):]
                if not relative or relative.endswith("/"):
                    continue
                target = local_root / PurePosixPath(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, remote_key, str(target))
                downloaded += 1
        return downloaded

    def get_file(self, relative: str, local_path: Path) -> bool:
        """Download one object only; used by large direct imports."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, self.key(relative), str(local_path))
            return True
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return False
            raise

    def object_exists(self, relative: str) -> bool:
        if not self.enabled:
            return False
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(relative))
            return True
        except Exception as error:
            return getattr(error, "response", {}).get("Error", {}).get("Code") not in {"NoSuchKey", "404", "NoSuchBucket"}

    def list_relative(self, relative_prefix: str) -> List[str]:
        if not self.enabled:
            return []
        prefix = self.key(relative_prefix.rstrip("/") + "/")
        results: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key != prefix and not key.endswith("/"):
                    results.append(key[len(prefix):])
        return results

    def copy_file(self, source_relative: str, target_relative: str) -> None:
        """Server-side R2 copy, so export files never occupy Render disk."""
        self.client.copy_object(Bucket=self.bucket, Key=self.key(target_relative), CopySource={"Bucket": self.bucket, "Key": self.key(source_relative)})

    def presign_put(self, relative: str, *, expires_in: int = 3600, content_type: str = "") -> str:
        """Create a short-lived browser upload URL without exposing R2 credentials."""
        if not self.enabled:
            raise RuntimeError("对象存储尚未配置。")
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": self.key(relative)},
            ExpiresIn=expires_in,
            HttpMethod="PUT",
        )

    def presign_get(self, relative: str, *, expires_in: int = 3600) -> str:
        if not self.enabled:
            raise RuntimeError("对象存储尚未配置。")
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self.key(relative)}, ExpiresIn=expires_in
        )

    def ensure_browser_upload_cors(self) -> None:
        """Allow direct PUT uploads only from the configured annotation site."""
        if not self.enabled:
            raise RuntimeError("对象存储尚未配置。")
        origin = os.environ.get(
            "DIRECT_UPLOAD_ALLOWED_ORIGIN", "https://defectbench-re-annotation.onrender.com"
        ).strip()
        if not origin:
            raise RuntimeError("DIRECT_UPLOAD_ALLOWED_ORIGIN 不能为空。")
        self.client.put_bucket_cors(
            Bucket=self.bucket,
            CORSConfiguration={"CORSRules": [{
                "AllowedOrigins": [origin],
                "AllowedMethods": ["PUT"],
                "AllowedHeaders": ["Content-Type"],
                "ExposeHeaders": ["ETag"],
                "MaxAgeSeconds": 3600,
            }]},
        )


class OSSStorage:
    """Alibaba Cloud OSS adapter with the same interface as the legacy R2 adapter."""

    def __init__(self) -> None:
        self.bucket_name = os.environ.get("OSS_BUCKET", "").strip()
        self.prefix = os.environ.get("OSS_PREFIX", "defectbench").strip("/")
        self.client = None
        self._lock = threading.Lock()
        if not self.bucket_name:
            return
        endpoint = os.environ.get("OSS_ENDPOINT", "").strip()
        access_key = os.environ.get("OSS_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("OSS_ACCESS_KEY_SECRET", "").strip()
        if not (oss2 and endpoint and access_key and secret_key):
            raise RuntimeError(
                "OSS_BUCKET 已设置，但缺少 oss2 或 OSS_ENDPOINT、OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET。"
            )
        self.client = oss2.Bucket(oss2.Auth(access_key, secret_key), endpoint, self.bucket_name)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def key(self, relative: str) -> str:
        clean = relative.replace("\\", "/").lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def put_file(self, local_path: Path, relative: str) -> None:
        if self.enabled:
            self.client.put_object_from_file(self.key(relative), str(local_path))

    def put_bytes(self, payload: bytes, relative: str, *, content_type: str = "application/json") -> None:
        if self.enabled:
            self.client.put_object(self.key(relative), payload, headers={"Content-Type": content_type})

    def get_bytes(self, relative: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        try:
            return self.client.get_object(self.key(relative)).read()
        except oss2.exceptions.NoSuchKey:
            return None

    def object_exists(self, relative: str) -> bool:
        if not self.enabled:
            return False
        try:
            self.client.get_object_meta(self.key(relative))
            return True
        except oss2.exceptions.NoSuchKey:
            return False

    def list_relative(self, relative_prefix: str) -> List[str]:
        if not self.enabled:
            return []
        prefix = self.key(relative_prefix.rstrip("/") + "/")
        return [item.key[len(prefix):] for item in oss2.ObjectIterator(self.client, prefix=prefix)
                if item.key != prefix and not item.key.endswith("/")]

    def put_tree(self, local_root: Path, remote_root: str) -> None:
        if not self.enabled:
            return
        for path in local_root.rglob("*"):
            if path.is_file():
                self.put_file(path, f"{remote_root.rstrip('/')}/{path.relative_to(local_root).as_posix()}")

    def get_tree(self, remote_root: str, local_root: Path) -> int:
        if not self.enabled:
            return 0
        prefix = self.key(remote_root.rstrip("/") + "/")
        downloaded = 0
        for item in oss2.ObjectIterator(self.client, prefix=prefix):
            relative = item.key[len(prefix):]
            if not relative or relative.endswith("/"):
                continue
            target = local_root / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.client.get_object_to_file(item.key, str(target))
            downloaded += 1
        return downloaded

    def get_file(self, relative: str, local_path: Path) -> bool:
        """Download one object only, keeping the web instance cache small."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.get_object_to_file(self.key(relative), str(local_path))
            return True
        except oss2.exceptions.NoSuchKey:
            return False

    def copy_file(self, source_relative: str, target_relative: str) -> None:
        """Copy server-side inside OSS; never write complete exports to Render disk."""
        self.client.copy_object(self.key(target_relative), self.bucket_name, self.key(source_relative))

    def presign_put(self, relative: str, *, expires_in: int = 3600, content_type: str = "") -> str:
        """Short-lived browser PUT URL.  The browser sends images direct to OSS."""
        if not self.enabled:
            raise RuntimeError("对象存储尚未配置。")
        # OSS includes Content-Type in a V1 signed URL.  It must exactly match
        # the browser PUT header, otherwise OSS returns SignatureDoesNotMatch.
        headers = {"Content-Type": content_type} if content_type else None
        return self.client.sign_url("PUT", self.key(relative), expires_in, headers=headers)

    def presign_get(self, relative: str, *, expires_in: int = 3600) -> str:
        """Short-lived browser GET URL for direct image and mask display."""
        if not self.enabled:
            raise RuntimeError("对象存储尚未配置。")
        return self.client.sign_url("GET", self.key(relative), expires_in)


# OSS takes precedence for Alibaba Cloud deployments.  The R2 fallback keeps
# the already deployed Render service usable until the OSS migration is live.
R2 = OSSStorage() if os.environ.get("OSS_BUCKET", "").strip() else R2Storage()
# Enabled only after the bucket CORS rule allows GET from the web application's
# origin.  Keeping this explicit prevents a CORS misconfiguration from making
# images disappear in the canvas.
DIRECT_BROWSER_MEDIA = os.environ.get("OSS_BROWSER_DIRECT_READ", "false").strip().lower() in {"1", "true", "yes"}
_direct_url_cache: Dict[str, Tuple[float, str]] = {}
ACTIVE_DATASET_DESCRIPTOR = "app_state/active_dataset.json"
_active_dataset_remote_root: Optional[str] = None
_active_dataset_local_import_root: Optional[Path] = None
_active_dataset_loaded = False
_active_dataset_manifest: Optional[Dict[str, Any]] = None

# Class color mapping (RGB)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Crack": (255, 0, 0),             # Red
    "Material_loss": (255, 140, 0),   # Orange
    "Stain": (30, 144, 255),          # Blue
    "External Fixings": (0, 200, 0),  # Green
}

# sub_type (detection model output) -> primary_class mapping
SUBTYPE_TO_PRIMARY_CLASS: Dict[str, str] = {
    "Concrete_Crack": "Crack",
    "Concrete_Delamination": "Material_loss",
    "Concrete_Spalling": "Material_loss",
    "Rust_Stain": "Stain",
    "Vegeterian": "External Fixings",
    "Degraded_Plaster": "Material_loss",
    "Craquelure": "Crack",
    "Tile_Crack": "Crack",
    "Tile_spalling": "Material_loss",
    "Water_Stain": "Stain",
    "Bulging": "Material_loss",
    "Contaminants": "External Fixings",
}
DEFAULT_COLOR = (255, 255, 0)  # Yellow

# Image list cache
_image_list_cache: List[Dict[str, Any]] = []
_image_list_loaded = False


def _configure_dataset_root(root: Path, *, output_under_dataset: bool = False) -> None:
    """Switch the active dataset while keeping algorithm inputs immutable."""
    global BASE_DIR, IMAGES_DIR, LABELS_DIR, MASKS_DIR, CANDIDATES_CSV
    global ALGORITHM_LABELS_DIR, ALGORITHM_MASKS_DIR
    global OUTPUT_ROOT, EXPERT_LABELS_DIR, EXPERT_MASKS_DIR, REVISIONS_DIR, DECISION_EVENTS_PATH, FINAL_EXPORTS_DIR
    global _image_list_cache, _image_list_loaded

    BASE_DIR = Path(root).resolve()
    IMAGES_DIR = BASE_DIR / "images" if (BASE_DIR / "images").exists() else BASE_DIR
    LABELS_DIR = BASE_DIR / "labels"
    MASKS_DIR = BASE_DIR / "masks"
    CANDIDATES_CSV = BASE_DIR / "candidates.csv"
    ALGORITHM_LABELS_DIR = (
        BASE_DIR / "algorithm_labels" if (BASE_DIR / "algorithm_labels").exists()
        else BASE_DIR / "detections" if (BASE_DIR / "detections").exists()
        else LABELS_DIR
    )
    ALGORITHM_MASKS_DIR = BASE_DIR / "algorithm_masks" if (BASE_DIR / "algorithm_masks").exists() else MASKS_DIR
    OUTPUT_ROOT = BASE_DIR / "annotation_output" if output_under_dataset else BASE_DIR
    EXPERT_LABELS_DIR = OUTPUT_ROOT / "expert_labels"
    EXPERT_MASKS_DIR = OUTPUT_ROOT / "expert_masks"
    REVISIONS_DIR = OUTPUT_ROOT / "revisions"
    DECISION_EVENTS_PATH = OUTPUT_ROOT / "decision_events.jsonl"
    FINAL_EXPORTS_DIR = OUTPUT_ROOT / "final_exports"
    _image_list_cache = []
    _image_list_loaded = False


def _remember_remote_dataset(import_root: Path, dataset_root: Path, remote_root: str) -> None:
    """Persist the active browser-imported batch so a new service instance can reopen it."""
    global _active_dataset_remote_root, _active_dataset_local_import_root, _active_dataset_loaded, _active_dataset_manifest
    if not R2.enabled:
        return
    descriptor = {
        "remote_root": remote_root,
        "dataset_relative_root": dataset_root.relative_to(import_root).as_posix(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "direct_upload": bool(_active_dataset_manifest),
        "manifest": _active_dataset_manifest,
    }
    R2.put_bytes(json.dumps(descriptor, ensure_ascii=False).encode("utf-8"), ACTIVE_DATASET_DESCRIPTOR)
    _active_dataset_remote_root = remote_root
    _active_dataset_local_import_root = import_root
    _active_dataset_loaded = True


def _ensure_active_dataset_loaded() -> None:
    """Restore the most recently imported cloud batch into the local cache once."""
    global _active_dataset_remote_root, _active_dataset_local_import_root, _active_dataset_loaded, _active_dataset_manifest
    if not R2.enabled or _active_dataset_loaded:
        return
    with R2._lock:
        if _active_dataset_loaded:
            return
        descriptor_bytes = R2.get_bytes(ACTIVE_DATASET_DESCRIPTOR)
        if not descriptor_bytes:
            _active_dataset_loaded = True
            return
        descriptor = json.loads(descriptor_bytes.decode("utf-8"))
        remote_root = str(descriptor["remote_root"])
        if descriptor.get("direct_upload"):
            cache_root = IMPORT_ROOT / "r2_active_dataset"
            relative_root = PurePosixPath(str(descriptor.get("dataset_relative_root") or "."))
            dataset_root = cache_root.joinpath(*relative_root.parts)
            for folder in ("images", "labels", "detections", "algorithm_labels", "masks", "algorithm_masks", "metadata", "expert_labels", "expert_masks", "revisions"):
                (dataset_root / folder).mkdir(parents=True, exist_ok=True)
            _configure_dataset_root(dataset_root)
            _apply_direct_manifest_layout(dataset_root)
            _active_dataset_remote_root = remote_root
            _active_dataset_local_import_root = cache_root
            _active_dataset_manifest = descriptor.get("manifest") or {}
            _active_dataset_loaded = True
            print(f"[R2] Restored direct dataset manifest from {remote_root}")
            return
        cache_root = IMPORT_ROOT / "r2_active_dataset"
        if cache_root.exists():
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        downloaded = R2.get_tree(remote_root, cache_root)
        relative_root = PurePosixPath(str(descriptor.get("dataset_relative_root") or "."))
        dataset_root = cache_root.joinpath(*relative_root.parts)
        if not downloaded or not dataset_root.is_dir():
            raise RuntimeError("R2 中未找到当前导入的数据集。")
        _configure_dataset_root(dataset_root)
        _active_dataset_remote_root = remote_root
        _active_dataset_local_import_root = cache_root
        _active_dataset_loaded = True
        print(f"[R2] Restored active dataset ({downloaded} files) from {remote_root}")


def _sync_active_dataset_file(path: Path) -> None:
    """Upload one changed expert output without re-uploading the original batch."""
    if R2.enabled and _active_dataset_remote_root:
        if not _active_dataset_local_import_root:
            return
        R2.put_file(path, f"{_active_dataset_remote_root}/{path.relative_to(_active_dataset_local_import_root).as_posix()}")


def _sync_active_dataset_tree(directory: Path) -> None:
    if R2.enabled and _active_dataset_remote_root and _active_dataset_local_import_root:
        R2.put_tree(
            directory,
            f"{_active_dataset_remote_root}/{directory.relative_to(_active_dataset_local_import_root).as_posix()}",
        )


def _mask_path(directory: Path, stem: str) -> Optional[Path]:
    for suffix in ("_mask.png", "_mask.jpg", ".png", ".jpg"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _sample_key(stem: str) -> str:
    """Normalise the image stem used by the supplied algorithm batch."""
    return stem[:-6] if stem.endswith("_image") else stem


def _baseline_label_path(stem: str) -> Path:
    """Resolve both batch-style ``<stem>.json`` and one-sample JSON inputs."""
    key = _sample_key(stem)
    for filename in (f"{stem}.json", f"{key}.json", f"{key}_detection.json"):
        matched = ALGORITHM_LABELS_DIR / filename
        if matched.exists():
            return matched
    for filename in ("detections.json", "annotations.json", "annotation.json"):
        candidate = ALGORITHM_LABELS_DIR / filename
        if candidate.exists():
            return candidate
    return matched


def _baseline_mask_path(stem: str) -> Optional[Path]:
    key = _sample_key(stem)
    for candidate_stem in (stem, key):
        matched = _mask_path(ALGORITHM_MASKS_DIR, candidate_stem)
        if matched:
            return matched
    for filename in ("algorithm_mask.png", "mask.png", "algorithm_mask.jpg", "mask.jpg"):
        candidate = ALGORITHM_MASKS_DIR / filename
        if candidate.exists():
            return candidate
    return None


def _metadata_path(stem: str) -> Optional[Path]:
    key = _sample_key(stem)
    candidate = BASE_DIR / "metadata" / f"{key}_metadata.json"
    return candidate if candidate.exists() else None


def _file_descriptor(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if not path or not path.exists():
        return None
    try:
        relative_path = str(path.relative_to(BASE_DIR))
    except ValueError:
        relative_path = str(path)
    return {"path": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _normalize_bbox(raw: Dict[str, Any], index: int, *, bbox_is_xyxy: bool = False) -> Optional[Dict[str, Any]]:
    """Accept both the legacy schema and common detector JSON schemas."""
    bbox = raw.get("bbox") or raw.get("bbox_xywh")
    if raw.get("bbox_xyxy"):
        x1, y1, x2, y2 = raw["bbox_xyxy"]
        bbox = [x1, y1, x2 - x1, y2 - y1]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if bbox_is_xyxy:
        x1, y1, x2, y2 = bbox
        bbox = [x1, y1, x2 - x1, y2 - y1]
    taxonomy = raw.get("taxonomy") or {}
    primary = taxonomy.get("primary_class") or raw.get("primary_class") or raw.get("defect_type") or raw.get("class_name")
    subtype = taxonomy.get("sub_type") or raw.get("sub_type") or raw.get("defect_subtype") or ""
    # YOLO's model labels are converted to the only ontology pairs that this
    # annotation tool permits experts to save.
    model_taxonomy = {
        "Concrete_Crack": ("Crack", "Linear crack"),
        "Craquelure": ("Crack", "Map cracking"),
        "Tile_Crack": ("Crack", "Linear crack"),
        "Concrete_Delamination": ("Material_loss", "Peeling"),
        "Concrete_Spalling": ("Material_loss", "Spalling"),
        "Tile_spalling": ("Material_loss", "Spalling"),
        "Bulging": ("Material_loss", "Peeling"),
        "Degraded_Plaster": ("Material_loss", "Peeling"),
        "Rust_Stain": ("Stain", "Rust stain"),
        "Water_Stain": ("Stain", "Leakage stain"),
        "Vegeterian": ("External Fixings", "Vegetation growth"),
        "Contaminants": ("External Fixings", "Surface contaminants"),
    }
    if raw.get("class_name") in model_taxonomy:
        primary, subtype = model_taxonomy[raw["class_name"]]
    if primary in PRIMARY_TO_SUBTYPES and subtype not in PRIMARY_TO_SUBTYPES[primary]:
        subtype = PRIMARY_TO_SUBTYPES[primary][0]
    return {
        "id": raw.get("instance_id") or raw.get("id") or raw.get("algorithm_instance_id") or f"instance-{index}",
        "bbox": bbox,
        "primary_class": primary,
        "sub_type": subtype,
        "score": raw.get("score") or raw.get("confidence"),
    }


def _load_bboxes_from_path(label_path: Path) -> List[Dict[str, Any]]:
    if not label_path.exists():
        return []
    try:
        with open(label_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = data.get("bboxes") or data.get("detections") or data.get("annotations") or data.get("annotations_in_crop") or []
        xyxy_input = "annotations_in_crop" in data
        normalized: List[Dict[str, Any]] = []
        for index, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            item = _normalize_bbox(raw, index, bbox_is_xyxy=xyxy_input)
            if item:
                normalized.append(item)
        return normalized
    except Exception as error:
        print(f"[Error] Failed to load annotations from {label_path}: {error}")
        return []


def _image_digest(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def _materialize_sample(stem: str) -> None:
    """Fetch only the requested sample and its small sidecar files into /tmp."""
    if not (_active_dataset_manifest and R2.enabled and _active_dataset_remote_root and _active_dataset_local_import_root):
        return
    root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
    prefix = root + "/" if root else ""
    sample_key = _sample_key(stem)
    for name in _active_dataset_manifest.get("files", []):
        if not any(name.startswith(prefix + folder + "/") for folder in ("images", "labels", "detections", "algorithm_labels", "masks", "algorithm_masks", "metadata")):
            continue
        if Path(name).stem.startswith(stem) or Path(name).stem.startswith(sample_key):
            local = _active_dataset_local_import_root / PurePosixPath(name)
            if not local.exists():
                R2.get_file(f"{_active_dataset_remote_root}/{name}", local)
    for name in (f"{prefix}expert_labels/{stem}.json", f"{prefix}expert_masks/{stem}_mask.png"):
        local = _active_dataset_local_import_root / PurePosixPath(name)
        if not local.exists():
            R2.get_file(f"{_active_dataset_remote_root}/{name}", local)


def _apply_direct_manifest_layout(dataset_root: Path) -> None:
    """Use the input folders that actually exist in a direct-upload manifest.

    The local cache creates empty compatibility folders.  They must not shadow
    an uploaded ``detections/`` or ``masks/`` directory.
    """
    global ALGORITHM_LABELS_DIR, ALGORITHM_MASKS_DIR
    if not _active_dataset_manifest:
        return
    root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
    prefix = root + "/" if root else ""
    files = _active_dataset_manifest.get("files", [])
    for folder in ("algorithm_labels", "detections", "labels"):
        if any(name.startswith(prefix + folder + "/") for name in files):
            ALGORITHM_LABELS_DIR = dataset_root / folder
            break
    for folder in ("algorithm_masks", "masks"):
        if any(name.startswith(prefix + folder + "/") for name in files):
            ALGORITHM_MASKS_DIR = dataset_root / folder
            break


def _direct_manifest_file(stem: str, folders: Tuple[str, ...], *, mask: bool = False) -> Optional[str]:
    """Find a sidecar object by the exact conventions used by the input batch."""
    if not _active_dataset_manifest:
        return None
    root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
    prefix = root + "/" if root else ""
    key = _sample_key(stem)
    if mask:
        expected = {f"{stem}_mask.png", f"{key}_mask.png", f"{stem}.png", f"{key}.png", f"{stem}_mask.jpg", f"{key}_mask.jpg"}
    else:
        expected = {f"{stem}.json", f"{key}.json", f"{key}_detection.json"}
    for folder in folders:
        folder_prefix = prefix + folder + "/"
        for name in _active_dataset_manifest.get("files", []):
            if name.startswith(folder_prefix) and Path(name).name in expected:
                return name
    return None


def _direct_object_bytes(relative: Optional[str]) -> Optional[bytes]:
    if not (relative and R2.enabled and _active_dataset_remote_root):
        return None
    return R2.get_bytes(f"{_active_dataset_remote_root}/{relative}")


def _direct_browser_url(relative: Optional[str]) -> Optional[str]:
    """Presign an OSS/R2 object for the browser when direct media is enabled."""
    if not (DIRECT_BROWSER_MEDIA and relative and R2.enabled and _active_dataset_remote_root):
        return None
    try:
        object_path = f"{_active_dataset_remote_root}/{relative}"
        cached = _direct_url_cache.get(object_path)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]
        url = R2.presign_get(object_path, expires_in=3600)
        # Reuse the exact URL for five minutes so browser preloading also
        # benefits the request that follows it.
        _direct_url_cache[object_path] = (now + 300, url)
        return url
    except Exception as error:
        print(f"[Direct media] Failed to sign GET URL: {error}")
        return None


def _bboxes_from_bytes(payload: Optional[bytes]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    try:
        data = json.loads(payload.decode("utf-8"))
        items = data.get("bboxes") or data.get("detections") or data.get("annotations") or data.get("annotations_in_crop") or []
        xyxy_input = "annotations_in_crop" in data
        normalized: List[Dict[str, Any]] = []
        for index, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            item = _normalize_bbox(raw, index, bbox_is_xyxy=xyxy_input)
            if item:
                normalized.append(item)
        return normalized
    except Exception as error:
        print(f"[Direct import] Failed to parse detection JSON: {error}")
        return []


def _mask_from_bytes(payload: Optional[bytes]) -> Optional[np.ndarray]:
    if not payload:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    except Exception as error:
        print(f"[Direct import] Failed to parse mask PNG: {error}")
        return None


def _load_image_list():
    """Load all images from images directory and categorize by naming pattern"""
    global _image_list_cache, _image_list_loaded
    _ensure_active_dataset_loaded()
    if _image_list_loaded:
        return
    
    print("[Cache] Loading image list...")
    _image_list_cache = []

    if _active_dataset_manifest:
        root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
        prefix = root + "/images/" if root else "images/"
        for remote_name in sorted(_active_dataset_manifest.get("files", [])):
            relative = remote_name[len(prefix):] if remote_name.startswith(prefix) else ""
            if not relative or "/" in relative or Path(relative).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = Path(relative).stem
            _image_list_cache.append({"stem": stem, "name": relative, "path": str(IMAGES_DIR / relative), "dataset": "civil" if stem.startswith("civil_") else "open" if stem.startswith("open_") else "other", "has_label": True, "has_mask": True})
        _image_list_loaded = True
        return
    
    if not IMAGES_DIR.exists():
        print(f"[Cache] Images directory not found: {IMAGES_DIR}")
        _image_list_loaded = True
        return
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(IMAGES_DIR.glob(f"*{ext}"))
    # Windows globbing is case-insensitive, so ``*.jpg`` and ``*.JPG`` may
    # return the same path twice.
    image_files = list({image_file.resolve() for image_file in image_files})
    
    for img_file in sorted(image_files):
        stem = img_file.stem
        img_name = img_file.name
        # In a one-sample flat input directory, do not treat the supplied mask
        # itself as a second image to annotate.
        if IMAGES_DIR == BASE_DIR and (stem.lower() in {"mask", "algorithm_mask"} or stem.lower().endswith("_mask")):
            continue
        
        # This re-annotation batch encodes the provenance in its filename.
        # It is a filtering aid only; it never becomes an annotation label.
        if stem.startswith('civil_'):
            dataset = 'civil'
        elif stem.startswith('open_'):
            dataset = 'open'
        else:
            dataset = 'other'
        
        # Expert output has priority for display; the algorithm result remains
        # available separately as the baseline.
        expert_label_path = EXPERT_LABELS_DIR / f"{stem}.json"
        has_label = expert_label_path.exists() or _baseline_label_path(stem).exists()
        
        # Check if mask exists
        has_mask = bool(_mask_path(EXPERT_MASKS_DIR, stem) or _baseline_mask_path(stem))
        
        _image_list_cache.append({
            'stem': stem,
            'name': img_name,
            'path': str(img_file),
            'dataset': dataset,
            'has_label': has_label,
            'has_mask': has_mask
        })
    
    print(f"[Cache] Loaded {len(_image_list_cache)} images")
    _image_list_loaded = True


def _get_filtered_images(dataset: Optional[str] = None, primary_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get filtered image list based on dataset and primary_class"""
    _load_image_list()
    
    filtered = _image_list_cache.copy()
    
    # Filter by dataset
    if dataset:
        filtered = [img for img in filtered if img['dataset'] == dataset]
    
    # Filter by primary_class (need to check labels)
    if primary_class:
        matching_images = []
        for img in filtered:
            label_path = EXPERT_LABELS_DIR / f"{img['stem']}.json"
            if not label_path.exists():
                label_path = _baseline_label_path(img['stem'])
            if any(bbox.get("primary_class") == primary_class for bbox in _load_bboxes_from_path(label_path)):
                matching_images.append(img)
        
        filtered = matching_images
    
    return filtered


def _load_bboxes_for_image(stem: str, *, baseline: bool = False) -> List[Dict[str, Any]]:
    """Load the expert version, falling back to the immutable algorithm version."""
    if baseline:
        return _load_bboxes_from_path(_baseline_label_path(stem))
    expert_path = EXPERT_LABELS_DIR / f"{stem}.json"
    # The supplied detector uses ``<sample>_detection.json`` while the image
    # uses ``<sample>_image.jpg``.  Reuse the baseline resolver rather than
    # assuming an exact ``<image-stem>.json`` filename.
    return _load_bboxes_from_path(expert_path if expert_path.exists() else _baseline_label_path(stem))


def _load_mask_for_image(stem: str, *, baseline: bool = False) -> Optional[np.ndarray]:
    """Load expert mask, falling back to the immutable algorithm mask."""
    mask_path = _baseline_mask_path(stem) if baseline else (
        _mask_path(EXPERT_MASKS_DIR, stem) or _baseline_mask_path(stem)
    )
    
    if not mask_path:
        return None
    
    try:
        # PIL avoids OpenCV's Windows Unicode-path limitation (the test data
        # lives under a Chinese directory name).
        return np.asarray(Image.open(mask_path).convert("RGB"))
    except Exception as e:
        print(f"[Error] Failed to load mask for {stem}: {e}")
        return None


def _get_color_for_class(primary_class: Optional[str]) -> Tuple[int, int, int]:
    """Get color for the given class"""
    if primary_class:
        return CLASS_COLORS.get(primary_class, DEFAULT_COLOR)
    return DEFAULT_COLOR


def _get_class_for_color(color: Tuple[int, int, int], tolerance: int = 10) -> Optional[str]:
    """Get class name from color (for mask editing)"""
    r, g, b = color
    for class_name, class_color in CLASS_COLORS.items():
        cr, cg, cb = class_color
        if abs(r - cr) <= tolerance and abs(g - cg) <= tolerance and abs(b - cb) <= tolerance:
            return class_name
    return None


def _append_decision_event(
    stem: str,
    image_path: Path,
    before_bboxes: List[Dict[str, Any]],
    after_bboxes: List[Dict[str, Any]],
    actor: str,
    reason_note: str,
    mask_changed: bool,
    algorithm_baseline: Dict[str, Any],
) -> int:
    """Persist an append-only expert decision record and a revision snapshot."""
    sample_revisions = REVISIONS_DIR / stem
    sample_revisions.mkdir(parents=True, exist_ok=True)
    revision = len(list(sample_revisions.glob("revision_*.json"))) + 1
    previous_hash = ""
    if revision > 1:
        prior = sample_revisions / f"revision_{revision - 1:04d}.json"
        if prior.exists():
            previous_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
    snapshot = {
        "sample_id": stem,
        "revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "image_sha256": _image_digest(image_path),
        "bboxes": after_bboxes,
        "mask_path": str((EXPERT_MASKS_DIR / f"{stem}_mask.png").relative_to(BASE_DIR)) if mask_changed else None,
        "parent_snapshot_sha256": previous_hash,
        "algorithm_baseline": algorithm_baseline,
    }
    snapshot_path = sample_revisions / f"revision_{revision:04d}.json"
    snapshot_bytes = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    snapshot_path.write_bytes(snapshot_bytes)
    event = {
        "event_id": str(uuid.uuid4()),
        "sample_id": stem,
        "revision": revision,
        "parent_snapshot_sha256": previous_hash,
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "created_at": snapshot["created_at"],
        "actor": {"id": actor, "role": "expert"},
        "action": "annotation.commit",
        "reason_note": reason_note,
        "before": {"bbox_count": len(before_bboxes)},
        "after": {"bbox_count": len(after_bboxes), "mask_changed": mask_changed},
        "algorithm_baseline": algorithm_baseline,
    }
    DECISION_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_EVENTS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    _sync_active_dataset_file(snapshot_path)
    _sync_active_dataset_file(DECISION_EVENTS_PATH)
    return revision


# HTML Template (contains frontend interaction logic)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>DefectBench Re-annotation</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            display: grid;
            grid-template-columns: 220px minmax(0, 1fr);
            align-items: flex-start;
            gap: 18px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ddd;
        }
        .controls {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            flex-wrap: wrap;
            overflow-x: visible;
            padding-bottom: 2px;
        }
        .control-group, .save-group {
            display: flex;
            align-items: center;
            gap: 7px;
            padding: 8px;
            border: 1px solid #d9e1ea;
            border-radius: 7px;
            background: #f8fafc;
            flex: 0 0 auto;
        }
        .save-group {
            width: auto;
            justify-content: flex-end;
            background: #f2f7ff;
            border-color: #b9d5f5;
        }
        .top-action-group { display: flex; align-items: center; gap: 8px; }
        .filter-menu { position: relative; }
        .filter-icon-button {
            width: 40px; height: 40px; padding: 0; display: inline-flex; align-items: center;
            justify-content: center; background: #fff; color: #334e68; border-color: #b8c5d1;
        }
        .filter-icon-button:hover, .filter-icon-button[aria-expanded="true"] { background: #e7f1ff; color: #0758b7; border-color: #82b5eb; }
        .filter-icon-button svg { width: 19px; height: 19px; fill: none; stroke: currentColor; stroke-width: 2; }
        .filter-panel {
            display: none; position: absolute; right: 0; top: calc(100% + 8px); z-index: 900;
            width: min(560px, calc(100vw - 48px)); padding: 12px; border: 1px solid #c7d6e5;
            border-radius: 9px; background: #fff; box-shadow: 0 10px 28px rgba(15, 40, 65, .18);
        }
        .filter-panel.open { display: block; }
        .filter-section { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; padding: 8px 0; }
        .filter-section.image-filter { flex-wrap: nowrap; }
        .filter-section + .filter-section { border-top: 1px solid #e5edf5; }
        .filter-section-title { flex: 0 0 68px; color: #52606d; font-size: 12px; font-weight: bold; }
        .import-button { background: #fff; color: #0758b7; border-color: #82b5eb; font-weight: 600; }
        .import-button:hover { background: #e7f1ff; }
        .export-button { background: #fff; color: #08705c; border-color: #80c9bb; font-weight: 600; }
        .export-button:hover { background: #e7f8f3; }
        .save-break { flex-basis: 100%; height: 0; }
        .control-label {
            color: #52606d;
            font-size: 12px;
            font-weight: bold;
            white-space: nowrap;
        }
        .editor-settings {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            margin: 0 0 12px;
            padding: 8px 10px;
            border: 1px solid #d9e1ea;
            border-radius: 6px;
            background: #fff;
            color: #334e68;
            font-size: 13px;
        }
        .editor-settings .control-label { color: #0b5cab; }
        .editor-settings label { display: inline-flex; align-items: center; gap: 6px; }
        .bbox-instruction, .mask-options { min-height: 46px; margin-bottom: 10px; }
        .bbox-instruction { display: flex; align-items: center; color: #555; font-size: 13px; }
        .mask-options { display: flex; align-items: center; justify-content: flex-start; flex-wrap: nowrap; gap: 10px 14px; }
        .brush-setting { display: inline-flex; align-items: center; gap: 6px; color: #4a5568; }
        .mask-options .editor-settings { flex: 0 0 auto; margin: 0; }
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
        select, input, button {
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="range"] {
            padding: 0;
            height: 20px;
            vertical-align: middle;
            accent-color: #0969da;
        }
        button {
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        button:hover { background-color: #45a049; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        .info {
            margin-bottom: 12px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 4px;
            font-size: 14px;
        }
        .main-content {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            background-color: #fafafa;
            min-width: 0;
        }
        .panel-title {
            display: flex;
            align-items: center;
            min-height: 42px;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .panel-title .editor-settings { margin: 0 0 0 auto; padding: 5px 8px; font-weight: normal; }
        .toolbar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .toolbar button {
            padding: 6px 12px;
            font-size: 13px;
        }
        .toolbar button.active {
            background-color: #2196F3;
        }
        .toolbar .reset-button {
            margin-left: auto; width: 34px; height: 32px; padding: 0; display: inline-flex;
            align-items: center; justify-content: center; background: #fff; color: #52606d; border-color: #b8c5d1;
        }
        .toolbar .reset-button:hover { background: #eef4fa; color: #0758b7; border-color: #82b5eb; }
        .toolbar .reset-button svg { width: 17px; height: 17px; fill: none; stroke: currentColor; stroke-width: 2; }
        .canvas-container {
            position: relative;
            background-color: #000;
            border: 2px solid #333;
            border-radius: 4px;
            overflow: hidden;
            text-align: center;
        }
        canvas {
            max-width: 100%;
            max-height: 600px;
            display: block;
            cursor: crosshair;
        }
        .color-legend {
            margin-top: 15px;
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
            font-size: 12px;
        }
        .color-item {
            display: inline-block;
            margin-right: 15px;
            margin-bottom: 5px;
        }
        .color-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 1px solid #333;
            vertical-align: middle;
            margin-right: 5px;
        }
        .status {
            margin-top: 10px;
            padding: 8px;
            background-color: #fff3cd;
            border-radius: 4px;
            font-size: 13px;
        }
        .status.success {
            background-color: #d4edda;
        }
        .status.error {
            background-color: #f8d7da;
        }
        .save-feedback {
            position: fixed; top: 18px; left: 50%; transform: translate(-50%, -18px);
            z-index: 1000; padding: 11px 18px; border-radius: 7px; background: #1f2937;
            color: white; box-shadow: 0 5px 18px rgba(0, 0, 0, .25); opacity: 0;
            pointer-events: none; transition: opacity .18s ease, transform .18s ease; font-weight: 600;
        }
        .save-feedback.visible { opacity: 1; transform: translate(-50%, 0); }
        .save-feedback.success { background: #17803d; }
        .save-feedback.error { background: #b42318; }
        #saveButton.saving { opacity: .72; cursor: wait; }
        #saveButton {
            background: #0969da;
            border-color: #0969da;
            padding: 11px 18px;
            font-size: 15px;
            font-weight: bold;
            box-shadow: 0 3px 8px rgba(9, 105, 218, .28);
        }
        #saveButton:hover { background: #0758b7; }
        @media (max-width: 900px) {
            .header { grid-template-columns: 1fr; }
            .controls { justify-content: flex-start; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DefectBench Re-annotation</h1>
            <div class="controls">
                <div class="top-action-group">
                    <div class="filter-menu" id="filterMenu">
                        <button id="filterToggle" class="filter-icon-button" type="button" onclick="toggleFilterPanel(event)" title="图片与页码筛选" aria-label="图片与页码筛选" aria-expanded="false">
                            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5h18l-7 8v5l-4 2v-7z"></path></svg>
                        </button>
                        <div class="filter-panel" id="filterPanel">
                            <div class="filter-section image-filter">
                                <span class="filter-section-title">图片筛选</span>
                                <select id="datasetSelect">
                                    <option value="">All sources</option><option value="civil">Self-collected</option><option value="open">Open-source</option><option value="other">Other</option>
                                </select>
                                <select id="primaryClassSelect">
                                    <option value="">All Classes</option><option value="Crack">Crack</option><option value="Material_loss">Material_loss</option><option value="Stain">Stain</option><option value="External Fixings">External Fixings</option>
                                </select>
                                <button onclick="loadImage(0)">Load First</button>
                            </div>
                            <div class="filter-section">
                                <span class="filter-section-title">页码筛选</span>
                                <input id="rangeStart" type="number" min="1" placeholder="Start" style="width:64px;"><span>–</span><input id="rangeEnd" type="number" min="1" placeholder="End" style="width:64px;">
                                <button onclick="applyWorkRange()">Range</button>
                                <input id="gotoNumber" type="number" min="1" placeholder="#" style="width:56px;"><button onclick="jumpToGlobalImage()">Go</button>
                            </div>
                        </div>
                    </div>
                    <button id="importButton" class="import-button" type="button" onclick="document.getElementById('datasetFolderInput').click()">导入文件夹</button>
                    <button id="ossBulkButton" class="import-button" type="button" onclick="openOssBulkDialog()">OSS 批量上传</button>
                    <button id="localPathButton" class="import-button" type="button" onclick="openLocalPathDialog()">本地路径导入</button>
                    <button id="exportButton" class="export-button" type="button" onclick="exportFinalDataset()">导出最终数据集</button>
                    <input id="datasetFolderInput" type="file" webkitdirectory directory multiple hidden onchange="importDatasetFolder(event)">
                </div>
                <div class="save-break"></div>
                <div class="save-group">
                    <span class="control-label">Review & save</span>
                    <button onclick="previousImage()">Previous</button><button onclick="nextImage()">Next</button>
                    <button id="saveButton" onclick="saveChanges()">Save Changes</button>
                </div>
            </div>
        </div>
        <div id="saveFeedback" class="save-feedback" role="status" aria-live="polite"></div>
        <div id="localPathDialog" style="display:none; position:fixed; inset:0; z-index:1100; background:rgba(0,0,0,.45); align-items:center; justify-content:center;">
            <div style="width:min(560px, calc(100vw - 36px)); padding:22px; border-radius:9px; background:#fff; box-shadow:0 12px 32px rgba(0,0,0,.28);">
                <h3 style="margin:0 0 10px;">本地路径导入</h3>
                <p style="margin:0 0 14px; color:#52606d; font-size:13px; line-height:1.5;">仅适用于本机运行。输入包含 images/、detections/ 和 masks/ 的数据集根目录；保存结果将写入该目录的 annotation_output/。</p>
                <input id="localDatasetPath" type="text" placeholder="例如：E:\\data\\defect_dataset" style="width:100%; margin-bottom:14px;">
                <div style="display:flex; justify-content:flex-end; gap:8px;">
                    <button type="button" onclick="closeLocalPathDialog()" style="background:#64748b;">取消</button>
                    <button id="localPathImportButton" type="button" onclick="importLocalDataset()">开始导入</button>
                </div>
            </div>
        </div>
        <div id="ossBulkDialog" style="display:none; position:fixed; inset:0; z-index:1100; background:rgba(0,0,0,.45); align-items:center; justify-content:center;">
            <div style="width:min(720px, calc(100vw - 36px)); padding:22px; border-radius:9px; background:#fff; box-shadow:0 12px 32px rgba(0,0,0,.28);">
                <h3 style="margin:0 0 8px;">OSS 批量上传（推荐 4GB 以上数据）</h3>
                <p style="margin:0 0 12px; color:#52606d; font-size:13px; line-height:1.5;">在存有原始数据的电脑安装 ossutil 2.0。先创建批次，复制命令并在该电脑执行；中断后执行同一条命令即可断点续传。上传完成后回到这里登记并载入。</p>
                <div style="display:flex; gap:8px; align-items:center; margin-bottom:10px;"><button type="button" onclick="createOssBulkBatch()">1. 创建批次</button></div>
                <div id="ossBulkState" role="status" style="display:none; margin:0 0 10px; padding:8px 10px; border-radius:5px; font-size:13px; line-height:1.45;"></div>
                <label style="font-size:13px; display:block; margin-bottom:5px;">本地数据集目录（用于生成命令）</label>
                <input id="ossBulkLocalPath" type="text" placeholder="例如：E:\\data\\test_input" oninput="renderOssBulkCommand()" style="width:100%; margin-bottom:9px;">
                <label style="font-size:13px; display:block; margin-bottom:5px;">上传命令</label>
                <textarea id="ossBulkCommand" readonly rows="4" style="width:100%; font-family:Consolas, monospace; font-size:12px; margin-bottom:9px;"></textarea>
                <div style="display:flex; gap:8px; align-items:center; margin-bottom:14px;"><button type="button" onclick="copyOssBulkCommand()">复制命令</button><input id="ossBulkUploadId" type="text" placeholder="批次标识" style="flex:1;"><button id="ossBulkRegisterButton" type="button" onclick="registerOssBulkBatch()">2. 登记并载入</button></div>
                <div style="display:flex; justify-content:flex-end;"><button type="button" onclick="closeOssBulkDialog()" style="background:#64748b;">关闭</button></div>
            </div>
        </div>
        
        <div class="info" id="infoDiv">
            Algorithm results are shown as the baseline. Expert edits are saved separately with an append-only decision history.
        </div>
        
        <div class="main-content">
            <!-- BBox Editing Panel -->
            <div class="panel">
                <div class="panel-title">BBox Editing
                    <div class="editor-settings">
                        <span class="control-label">Box display</span>
                        <label title="Only changes how boxes are displayed">Line width <input id="bboxLineWidth" type="range" min="1" max="12" value="4" oninput="updateDisplaySettings()" style="width:90px;"><span id="bboxLineWidthValue">4</span>px</label>
                    </div>
                </div>
                <div class="toolbar">
                    <button id="bboxViewBtn" class="active" onclick="setBBoxMode('view')">View</button>
                    <button id="bboxDrawBtn" onclick="setBBoxMode('draw')">Draw</button>
                    <button id="bboxDeleteBtn" onclick="setBBoxMode('delete')">Delete</button>
                    <button onclick="clearBBoxes()">Clear All</button>
                    <button class="reset-button" type="button" onclick="resetBBoxesToAlgorithm()" title="恢复算法初始检测框" aria-label="恢复算法初始检测框"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5"></path></svg></button>
                </div>
                <div class="bbox-instruction">
                    Manual annotation mode. Draw each defect box, then select both its primary class and subtype.
                </div>
                <div class="canvas-container">
                    <canvas id="bboxCanvas"></canvas>
                </div>
                <div class="color-legend">
                    <strong>BBox Color Legend:</strong>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(255,0,0);"></span>Crack</span>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(255,140,0);"></span>Material_loss</span>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(30,144,255);"></span>Stain</span>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(0,200,0);"></span>External Fixings</span>
                </div>
                <div class="status" id="bboxStatus"></div>
            </div>
            
            <!-- Label Selection Dialog -->
            <div id="labelDialog" style="display: none; position: fixed; inset: 0; background-color: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;">
                <div style="background-color: white; border-radius: 8px; padding: 24px; max-width: 400px; width: 90%;">
                    <h3 style="font-size: 18px; font-weight: bold; margin-bottom: 16px;">Select Defect Class</h3>
                    <div id="labelOptions" style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px;">
                        <!-- Options will be dynamically generated by JavaScript -->
                    </div>
                    <button onclick="cancelLabelSelection()" style="width: 100%; padding: 8px; background-color: #e0e0e0; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">Cancel</button>
                </div>
            </div>
            
            <!-- Mask Editing Panel -->
            <div class="panel">
                <div class="panel-title">Mask Editing
                    <div class="editor-settings">
                        <span class="control-label">Mask display</span>
                        <label title="Only changes how the mask overlay is displayed">Overlay opacity <input id="maskOpacity" type="range" min="10" max="100" value="65" oninput="updateDisplaySettings()" style="width:90px;"><span id="maskOpacityValue">65</span>%</label>
                    </div>
                </div>
                <div class="toolbar">
                    <button id="maskViewBtn" class="active" onclick="setMaskMode('view')">View</button>
                    <button id="maskBrushBtn" onclick="setMaskMode('brush-add')">Brush</button>
                    <button id="maskEraseBtn" onclick="setMaskMode('brush-remove')">Erase</button>
                    <button onclick="clearMask()">Clear Mask</button>
                    <button class="reset-button" type="button" onclick="resetMaskToAlgorithm()" title="恢复算法初始 Mask" aria-label="恢复算法初始 Mask"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5"></path></svg></button>
                </div>
                <div class="mask-options">
                    <label>Class Selection (for adding mask):</label>
                    <select id="maskClassSelect" onchange="drawMaskCanvas()">
                            <option value="Crack">Crack (Red)</option>
                            <option value="Material_loss">Material_loss (Orange)</option>
                            <option value="Stain">Stain (Blue)</option>
                            <option value="External Fixings">External Fixings (Green)</option>
                    </select>
                    <div class="brush-setting">
                        <label>Brush Size:</label><input type="range" id="brushSize" min="1" max="100" value="20" style="width: 100px;"><span id="brushSizeValue">20px</span>
                    </div>
                </div>
                <div class="canvas-container" id="maskContainer" style="position: relative;">
                    <canvas id="maskCanvas"></canvas>
                    <canvas id="maskBrushLayer" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; cursor: crosshair; z-index: 10;"></canvas>
                </div>
                <div class="status" id="maskStatus"></div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let currentIndex = 0;
        let currentDataset = null;
        let currentPrimaryClass = null;
        let currentImageData = null;
        let currentBBoxes = [];
        let algorithmBBoxes = [];
        let currentMask = null;
        let algorithmMask = null;
        let originalMask = null;
        // Global numbering is the stable alphabetical order of all 5,324 files.
        // Range navigation applies only while both filters are set to All.
        let workRangeStart = null;
        let workRangeEnd = null;
        
        // Canvas and context
        const bboxCanvas = document.getElementById('bboxCanvas');
        const bboxCtx = bboxCanvas.getContext('2d');
        const maskCanvas = document.getElementById('maskCanvas');
        const maskCtx = maskCanvas.getContext('2d');
        const maskBrushLayer = document.getElementById('maskBrushLayer');
        const maskBrushCtx = maskBrushLayer.getContext('2d');
        
        // Modes
        let bboxMode = 'view'; // 'view', 'draw', 'delete'
        let maskMode = 'view'; // 'view', 'point', 'brush'
        
        // Interaction state
        let bboxDragging = null;
        let bboxDrawing = null;
        let bboxResizing = null;
        let pendingBBox = null;
        
        // Available defect classes
        const defectClasses = [
            'Crack',
            'Material_loss',
            'Stain',
            'External Fixings'
        ];
        const subtypesByPrimary = {
            'Crack': ['Linear crack', 'Map cracking'],
            'Material_loss': ['Spalling', 'Peeling'],
            'Stain': ['Corrosion', 'Rust stain', 'Leakage stain'],
            'External Fixings': ['Vegetation growth', 'Surface contaminants', 'Graffiti']
        };
        let maskDrawing = false;
        let maskIsDrawing = false;
        let maskCursorPos = null;
        let maskPoints = [];
        let maskLastBrushPos = null;
        
        // Color mapping
        const classColors = {
            'Crack': [255, 0, 0],
            'Material_loss': [255, 140, 0],
            'Stain': [30, 144, 255],
            'External Fixings': [0, 200, 0]
        };

        function toggleFilterPanel(event) {
            event.stopPropagation();
            const panel = document.getElementById('filterPanel');
            const toggle = document.getElementById('filterToggle');
            const isOpen = panel.classList.toggle('open');
            toggle.setAttribute('aria-expanded', String(isOpen));
        }

        function closeFilterPanel() {
            document.getElementById('filterPanel').classList.remove('open');
            document.getElementById('filterToggle').setAttribute('aria-expanded', 'false');
        }

        function openLocalPathDialog() {
            document.getElementById('localPathDialog').style.display = 'flex';
            document.getElementById('localDatasetPath').focus();
        }

        function closeLocalPathDialog() {
            document.getElementById('localPathDialog').style.display = 'none';
        }

        let ossBulkInfo = null;

        function setOssBulkState(message, type = 'info') {
            const state = document.getElementById('ossBulkState');
            state.textContent = message;
            state.style.display = 'block';
            state.style.color = type === 'error' ? '#8f1d20' : type === 'success' ? '#146c32' : '#1f4f7a';
            state.style.background = type === 'error' ? '#f8d7da' : type === 'success' ? '#d4edda' : '#e7f3ff';
        }

        function openOssBulkDialog() {
            document.getElementById('ossBulkDialog').style.display = 'flex';
        }

        function closeOssBulkDialog() {
            document.getElementById('ossBulkDialog').style.display = 'none';
        }

        function renderOssBulkCommand() {
            const output = document.getElementById('ossBulkCommand');
            if (!ossBulkInfo) {
                output.value = '请先点击“创建批次”。';
                return;
            }
            const localPath = document.getElementById('ossBulkLocalPath').value.trim() || 'E:\\data\\your_dataset';
            const checkpoint = localPath.replace(/[\\/]+$/, '') + '\\ossutil-checkpoint';
            output.value = `ossutil cp -r -f -u "${localPath}" "${ossBulkInfo.destination}" --checkpoint-dir "${checkpoint}" -j 10 --parallel 10`;
        }

        async function createOssBulkBatch() {
            setOssBulkState('正在创建唯一 OSS 上传批次…');
            try {
                const response = await fetch('/api/oss_bulk/create', { method: 'POST' });
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.error || '创建批次失败');
                ossBulkInfo = data;
                document.getElementById('ossBulkUploadId').value = data.upload_id;
                setOssBulkState(`批次已创建：${data.upload_id}。复制命令并在源电脑执行。`, 'success');
                renderOssBulkCommand();
            } catch (error) {
                setOssBulkState('创建失败：' + error.message, 'error');
            }
        }

        async function copyOssBulkCommand() {
            const command = document.getElementById('ossBulkCommand').value;
            if (!command || command.startsWith('请先')) return;
            try {
                await navigator.clipboard.writeText(command);
                setOssBulkState('命令已复制。请在源电脑的终端执行；中断后执行同一命令可继续。', 'success');
            } catch (_) {
                document.getElementById('ossBulkCommand').select();
                document.execCommand('copy');
            }
        }

        async function registerOssBulkBatch() {
            const uploadId = document.getElementById('ossBulkUploadId').value.trim();
            const button = document.getElementById('ossBulkRegisterButton');
            if (!uploadId) return;
            button.disabled = true;
            setOssBulkState('正在扫描 OSS 中已上传的文件…');
            try {
                const registerResponse = await fetch('/api/oss_bulk/register', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ upload_id: uploadId })
                });
                const registered = await registerResponse.json();
                if (!registerResponse.ok || !registered.success) throw new Error(registered.error || '登记失败');
                setOssBulkState(`已发现 ${registered.files} 个文件，正在校验目录并激活数据集…`);
                const completeResponse = await fetch('/api/direct_upload/complete', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ upload_id: uploadId })
                });
                const completed = await completeResponse.json();
                if (!completeResponse.ok || !completed.success) throw new Error(completed.error || '校验或载入失败');
                resetDatasetUi();
                await updateImageList();
                setOssBulkState(`载入成功：${completed.total} 张图片。正在关闭窗口…`, 'success');
                await new Promise(resolve => setTimeout(resolve, 500));
                closeOssBulkDialog();
                showStatus('bboxStatus', `已登记并载入 ${completed.dataset_name}，共 ${completed.total} 张图片`, 'success');
            } catch (error) {
                console.error('OSS bulk registration failed:', error);
                setOssBulkState('登记失败：' + error.message, 'error');
            } finally {
                button.disabled = false;
            }
        }

        function resetDatasetUi() {
            currentDataset = null;
            currentPrimaryClass = null;
            currentIndex = 0;
            workRangeStart = null;
            workRangeEnd = null;
            loadedAnnotationState = null;
            document.getElementById('datasetSelect').value = '';
            document.getElementById('primaryClassSelect').value = '';
            document.getElementById('rangeStart').value = '';
            document.getElementById('rangeEnd').value = '';
            document.getElementById('gotoNumber').value = '';
        }

        async function importLocalDataset() {
            const pathInput = document.getElementById('localDatasetPath');
            const datasetPath = pathInput.value.trim();
            if (!datasetPath) {
                pathInput.focus();
                return;
            }
            if (!confirmDiscardUnsavedChanges()) return;
            const button = document.getElementById('localPathImportButton');
            button.disabled = true;
            button.textContent = '正在读取…';
            try {
                const response = await fetch('/api/open_local_dataset', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: datasetPath })
                });
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.error || '导入失败');
                resetDatasetUi();
                await updateImageList();
                closeLocalPathDialog();
                closeFilterPanel();
                showStatus('bboxStatus', `已读取 ${data.dataset_name}，结果将保存至 ${data.output_folder}`, 'success');
            } catch (error) {
                console.error('Failed to open local dataset:', error);
                showStatus('bboxStatus', `本地路径导入失败：${error.message}`, 'error');
            } finally {
                button.disabled = false;
                button.textContent = '开始导入';
            }
        }

        document.addEventListener('click', (event) => {
            if (!document.getElementById('filterMenu').contains(event.target)) closeFilterPanel();
        });

        async function importDatasetFolder(event) {
            const input = event.target;
            if (!input.files || !input.files.length) return;
            if (!confirmDiscardUnsavedChanges()) {
                input.value = '';
                return;
            }

            const importButton = document.getElementById('importButton');
            const originalText = importButton.textContent;

            importButton.disabled = true;
            importButton.textContent = '正在导入…';
            try {
                const runtimeResponse = await fetch('/api/runtime');
                const runtime = await runtimeResponse.json();
                const files = Array.from(input.files);
                let response;
                if (runtime.cloud_storage) {
                    response = await importDatasetFolderDirect(files, importButton);
                } else {
                    const formData = new FormData();
                    for (const file of files) formData.append('files', file, file.webkitRelativePath || file.name);
                    response = await fetch('/api/import_dataset', { method: 'POST', body: formData });
                }
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.error || '导入失败');

                resetDatasetUi();
                await updateImageList();
                closeFilterPanel();
                showStatus('bboxStatus', `已导入 ${data.dataset_name}，共 ${data.total} 张图片`, 'success');
            } catch (error) {
                console.error('Failed to import dataset:', error);
                showStatus('bboxStatus', `导入失败：${error.message}`, 'error');
            } finally {
                input.value = '';
                importButton.disabled = false;
                importButton.textContent = originalText;
            }
        }

        const DIRECT_UPLOAD_SESSION_KEY = 'defectbench.direct-upload-session.v1';

        function directUploadEntries(files) {
            return files.map(file => ({
                path: file.webkitRelativePath || file.name,
                size: file.size,
                modified: file.lastModified,
                content_type: file.type || ''
            }));
        }

        function directUploadFingerprint(entries) {
            return entries.map(entry => `${entry.path}|${entry.size}|${entry.modified}`).sort().join('\\n');
        }

        async function importDatasetFolderDirect(files, importButton) {
            const entries = directUploadEntries(files);
            const fingerprint = directUploadFingerprint(entries);
            const filesByPath = new Map(files.map(file => [file.webkitRelativePath || file.name, file]));
            let savedSession = null;
            try { savedSession = JSON.parse(localStorage.getItem(DIRECT_UPLOAD_SESSION_KEY) || 'null'); } catch (_) {}

            let startResponse;
            if (savedSession && savedSession.fingerprint === fingerprint && savedSession.upload_id) {
                importButton.textContent = '正在恢复未完成上传…';
                startResponse = await fetch('/api/direct_upload/resume', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ upload_id: savedSession.upload_id, files: entries })
                });
            } else {
                startResponse = await fetch('/api/direct_upload/start', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ files: entries })
                });
            }
            let start = await startResponse.json();
            if (!startResponse.ok || !start.success) {
                // A stale local record should not block a new upload attempt.
                if (savedSession) localStorage.removeItem(DIRECT_UPLOAD_SESSION_KEY);
                throw new Error(start.error || '无法创建或恢复直传任务');
            }
            localStorage.setItem(DIRECT_UPLOAD_SESSION_KEY, JSON.stringify({ upload_id: start.upload_id, fingerprint }));
            if (!Array.isArray(start.uploads)) throw new Error('直传任务文件清单不完整');

            let completed = Number(start.completed || 0);
            const updateProgress = () => {
                importButton.textContent = `正在直传对象存储… ${completed}/${files.length}`;
            };
            let nextIndex = 0;
            const uploadWorker = async () => {
                while (nextIndex < start.uploads.length) {
                    const index = nextIndex++;
                    const upload = start.uploads[index];
                    const file = filesByPath.get(upload.path);
                    if (!file) throw new Error(`本地未找到文件：${upload.path}`);
                    let lastError = '';
                    let putResponse;
                    // Browser-to-object-storage uploads occasionally lose one
                    // connection. Retry the individual file rather than forcing
                    // the operator to restart a complete batch.
                    for (let attempt = 1; attempt <= 3; attempt++) {
                        try {
                            const contentType = file.type || '';
                            putResponse = await fetch(upload.url, {
                                method: 'PUT', body: file,
                                headers: contentType ? { 'Content-Type': contentType } : undefined
                            });
                            if (putResponse.ok) break;
                            const body = (await putResponse.text()).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
                            lastError = `HTTP ${putResponse.status}${body ? `：${body.slice(0, 180)}` : ''}`;
                        } catch (error) {
                            lastError = error.message || '网络连接失败';
                        }
                        if (attempt < 3) await new Promise(resolve => setTimeout(resolve, attempt * 800));
                    }
                    if (!putResponse || !putResponse.ok) {
                        throw new Error(`上传失败：${upload.path}（${lastError || '未知错误'}）`);
                    }
                    completed += 1;
                    updateProgress();
                }
            };
            updateProgress();
            await Promise.all(Array.from({ length: Math.min(4, start.uploads.length) }, uploadWorker));
            importButton.textContent = '正在校验并载入…';
            const completeResponse = await fetch('/api/direct_upload/complete', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ upload_id: start.upload_id })
            });
            if (completeResponse.ok) localStorage.removeItem(DIRECT_UPLOAD_SESSION_KEY);
            return completeResponse;
        }
        
        // Initialize
        document.getElementById('datasetSelect').addEventListener('change', async (e) => {
            currentDataset = e.target.value || null;
            currentIndex = 0;
            await updateImageList();
        });
        
        document.getElementById('brushSize').addEventListener('input', (e) => {
            document.getElementById('brushSizeValue').textContent = e.target.value + 'px';
        });
        
        document.getElementById('primaryClassSelect').addEventListener('change', async (e) => {
            currentPrimaryClass = e.target.value || null;
            currentIndex = 0;
            await updateImageList();
            if (currentImageData && currentImageData.total > 0) {
                loadImage(0);
            } else {
                currentImageData = null;
                currentBBoxes = [];
                currentMask = null;
                drawBBoxCanvas();
                drawMaskCanvas();
                document.getElementById('infoDiv').textContent = `No images found with ${currentPrimaryClass || 'specified class'}`;
            }
        });
        
        // Update image list
        let loadedAnnotationState = null;

        function annotationStateSignature() {
            const boxes = currentBBoxes.map(({ bbox, primary_class, sub_type }) => ({ bbox, primary_class, sub_type }));
            const mask = currentMask instanceof HTMLCanvasElement ? currentMask.toDataURL('image/png') : '';
            return JSON.stringify(boxes) + '|' + mask;
        }

        function hasUnsavedChanges() {
            return loadedAnnotationState !== null && annotationStateSignature() !== loadedAnnotationState;
        }

        function confirmDiscardUnsavedChanges() {
            return !hasUnsavedChanges() || window.confirm('当前图片有未保存的标注修改。切换图片将丢失这些修改，是否继续？');
        }

        async function updateImageList() {
            const params = new URLSearchParams();
            if (currentDataset) params.append('dataset', currentDataset);
            if (currentPrimaryClass) params.append('primary_class', currentPrimaryClass);
            
            try {
                const response = await fetch(`/api/images?${params}`);
                const data = await response.json();
                if (data.total > 0) {
                    loadImage(0);
                } else {
                    currentImageData = null;
                    currentBBoxes = [];
                    currentMask = null;
                    loadedAnnotationState = null;
                    drawBBoxCanvas();
                    drawMaskCanvas();
                    document.getElementById('infoDiv').textContent = `No images found`;
                }
            } catch (error) {
                console.error('Failed to load image list:', error);
            }
        }
        
        // Load image
        async function loadImage(index, bypassWorkRange = false) {
            if (!confirmDiscardUnsavedChanges()) return;
            if (!bypassWorkRange && !currentDataset && !currentPrimaryClass && workRangeStart !== null) {
                if (index < workRangeStart - 1) index = workRangeStart - 1;
                if (index > workRangeEnd - 1) {
                    showStatus('bboxStatus', `Reached selected range: ${workRangeStart}-${workRangeEnd}`, 'success');
                    return;
                }
            }
            try {
                const params = new URLSearchParams({
                    index: index.toString()
                });
                if (currentDataset) params.append('dataset', currentDataset);
                if (currentPrimaryClass) params.append('primary_class', currentPrimaryClass);
                
                const response = await fetch(`/api/image/${index}?${params}`);
                const data = await response.json();
                
                if (data.error) {
                    showStatus('bboxStatus', data.error, 'error');
                    return;
                }
                
                currentIndex = index;
                currentImageData = data;
                currentBBoxes = data.bboxes || [];
                algorithmBBoxes = data.algorithm_bboxes || [];
                // Default the mask paint class to the current annotation's
                // detected class.  Expert boxes take precedence; when an
                // expert has not edited the image yet, use the algorithm box.
                const detectedClass = (currentBBoxes[0] || algorithmBBoxes[0] || {}).primary_class;
                const maskClassSelect = document.getElementById('maskClassSelect');
                if (detectedClass && [...maskClassSelect.options].some(option => option.value === detectedClass)) {
                    maskClassSelect.value = detectedClass;
                }
                
                // Ensure each bbox has id
                currentBBoxes.forEach((bbox, idx) => {
                    if (bbox.id === undefined) {
                        bbox.id = idx;
                    }
                });
                
                // Clear pending box
                pendingBBox = null;
                cancelLabelSelection();
                
                // Prefer an expert mask, but always fall back to the algorithm
                // mask for display.  This keeps a newly imported binary mask
                // visible before the first expert save.
                const displayMaskData = data.mask || data.algorithm_mask;
                if (displayMaskData) {
                    const maskImg = await loadImageFromBase64(displayMaskData);
                    const img = new Image();
                    await new Promise((resolve) => {
                        img.onload = () => {
                            currentMask = document.createElement('canvas');
                            currentMask.width = img.width;
                            currentMask.height = img.height;
                            const maskCtx2 = currentMask.getContext('2d');
                            maskCtx2.drawImage(maskImg, 0, 0, img.width, img.height);
                            resolve();
                        };
                        setCanvasImageSource(img, currentImageData.image);
                    });
                    originalMask = await cloneImage(maskImg);
                } else {
                    currentMask = null;
                    originalMask = null;
                }
                algorithmMask = data.algorithm_mask ? await cloneImage(await loadImageFromBase64(data.algorithm_mask)) : null;
                loadedAnnotationState = annotationStateSignature();
                
                // Update info
                document.getElementById('infoDiv').innerHTML = `
                    <strong>Global #${data.global_index}</strong> | Filtered position ${index + 1} / ${data.total} | 
                    ${data.filename} | Algorithm boxes: ${algorithmBBoxes.length} | Expert boxes: ${currentBBoxes.length} | 
                    Mask: ${currentMask ? `Yes (${data.mask_source || 'algorithm'})` : 'No'}
                `;
                
                // Draw
                drawBBoxCanvas();
                drawMaskCanvas();
                
                showStatus('bboxStatus', 'Image loaded successfully', 'success');
                showStatus('maskStatus', 'Image loaded successfully', 'success');
                prefetchMedia(data.prefetch_media || []);
                
            } catch (error) {
                console.error('Failed to load image:', error);
                showStatus('bboxStatus', 'Load failed: ' + error.message, 'error');
            }
        }

        async function globalTotal() {
            const response = await fetch('/api/images');
            const data = await response.json();
            return data.total;
        }

        function clearFiltersForGlobalNavigation() {
            currentDataset = null;
            currentPrimaryClass = null;
            document.getElementById('datasetSelect').value = '';
            document.getElementById('primaryClassSelect').value = '';
        }

        async function applyWorkRange() {
            const start = Number.parseInt(document.getElementById('rangeStart').value, 10);
            const end = Number.parseInt(document.getElementById('rangeEnd').value, 10);
            const total = await globalTotal();
            if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start || end > total) {
                showStatus('bboxStatus', `Enter a valid range between 1 and ${total}.`, 'error');
                return;
            }
            clearFiltersForGlobalNavigation();
            workRangeStart = start;
            workRangeEnd = end;
            await loadImage(start - 1, true);
            showStatus('bboxStatus', `Work range set to global images ${start}-${end}.`, 'success');
        }

        async function jumpToGlobalImage() {
            const target = Number.parseInt(document.getElementById('gotoNumber').value, 10);
            const total = await globalTotal();
            if (!Number.isInteger(target) || target < 1 || target > total) {
                showStatus('bboxStatus', `Enter a valid global number between 1 and ${total}.`, 'error');
                return;
            }
            clearFiltersForGlobalNavigation();
            await loadImage(target - 1, true);
        }

        function previousImage() {
            if (currentIndex > 0) loadImage(currentIndex - 1);
        }

        function nextImage() {
            loadImage(currentIndex + 1);
        }
        
        // Load image from base64
        function loadImageFromBase64(base64Str) {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    retainMedia(base64Str, img);
                    resolve(img);
                };
                img.onerror = () => reject(new Error('Image or mask request failed'));
                setCanvasImageSource(img, base64Str);
            });
        }

        function setCanvasImageSource(img, source) {
            // Direct OSS URLs are cross-origin.  CORS + this flag keep the
            // image usable by the canvases and by mask export.
            if (/^https?:\/\//i.test(source)) img.crossOrigin = 'anonymous';
            img.src = source;
        }

        async function imageDataForApi() {
            const source = currentImageData && currentImageData.image;
            if (!source || source.startsWith('data:')) return source;
            const response = await fetch(source);
            if (!response.ok) throw new Error('Failed to fetch source image');
            const blob = await response.blob();
            return await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error('Failed to encode source image'));
                reader.readAsDataURL(blob);
            });
        }

        // Keep a bounded image cache: the current/seen images stay available
        // while the next three samples are fetched in the background.  The
        // bound avoids consuming several GB of browser memory on a long batch.
        const mediaCache = new Map();
        const MAX_MEDIA_CACHE_ITEMS = 16;

        function retainMedia(source, image) {
            if (!source || !/^https?:\/\//i.test(source)) return;
            if (mediaCache.has(source)) mediaCache.delete(source);
            mediaCache.set(source, image);
            while (mediaCache.size > MAX_MEDIA_CACHE_ITEMS) {
                mediaCache.delete(mediaCache.keys().next().value);
            }
        }

        function prefetchMedia(sources) {
            sources.forEach(source => {
                if (!source || mediaCache.has(source)) return;
                const image = new Image();
                image.onload = () => retainMedia(source, image);
                image.onerror = () => mediaCache.delete(source);
                setCanvasImageSource(image, source);
                // Reserve a slot immediately so repeated image switches do
                // not launch duplicate downloads for the same object.
                mediaCache.set(source, image);
                while (mediaCache.size > MAX_MEDIA_CACHE_ITEMS) {
                    mediaCache.delete(mediaCache.keys().next().value);
                }
            });
        }
        
        // Clone Image object
        function cloneImage(img) {
            return new Promise((resolve) => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                const newImg = new Image();
                newImg.onload = () => resolve(newImg);
                newImg.src = canvas.toDataURL();
            });
        }

        function updateDisplaySettings() {
            document.getElementById('bboxLineWidthValue').textContent = document.getElementById('bboxLineWidth').value;
            document.getElementById('maskOpacityValue').textContent = document.getElementById('maskOpacity').value;
            drawBBoxCanvas();
            drawMaskCanvas();
        }

        // The supplied algorithm masks are binary PNGs.  Their non-zero area
        // uses the currently selected defect type's fixed colour.  Expert
        // masks painted in a known class colour retain that class colour.
        function buildDisplayMask(mask, width, height) {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(mask, 0, 0, width, height);
            const pixels = ctx.getImageData(0, 0, width, height);
            const selectedColor = classColors[document.getElementById('maskClassSelect').value] || [255, 255, 0];
            for (let i = 0; i < pixels.data.length; i += 4) {
                const visible = pixels.data[i + 3] > 0 && (pixels.data[i] > 0 || pixels.data[i + 1] > 0 || pixels.data[i + 2] > 0);
                if (visible) {
                    const existingColor = [pixels.data[i], pixels.data[i + 1], pixels.data[i + 2]];
                    const knownClassColor = Object.values(classColors).find(color =>
                        Math.abs(existingColor[0] - color[0]) <= 10 &&
                        Math.abs(existingColor[1] - color[1]) <= 10 &&
                        Math.abs(existingColor[2] - color[2]) <= 10
                    );
                    const [r, g, b] = knownClassColor || selectedColor;
                    pixels.data[i] = r;
                    pixels.data[i + 1] = g;
                    pixels.data[i + 2] = b;
                } else {
                    pixels.data[i + 3] = 0;
                }
            }
            ctx.putImageData(pixels, 0, 0);
            return canvas;
        }
        
        // Draw BBox Canvas (keeping the same drawing logic from original)
        let imageWidth = 0;
        let currentDisplayScale = 1;
        let currentAdjustedLineWidth = 2;
        let bboxRenderVersion = 0;
        let activeBBoxIndex = null;
        
        function drawBBoxCanvas() {
            if (!currentImageData || !currentImageData.image) return;
            const renderVersion = ++bboxRenderVersion;
            
            const img = new Image();
            img.onload = () => {
                // Mouse movement can request many redraws.  Ignore an older
                // image callback so it cannot overwrite the newest box state.
                if (renderVersion !== bboxRenderVersion) return;
                imageWidth = img.width;
                bboxCanvas.width = img.width;
                bboxCanvas.height = img.height;
                
                const maxWidth = bboxCanvas.parentElement.clientWidth - 20;
                const maxHeight = 600;
                const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
                bboxCanvas.style.width = (img.width * scale) + 'px';
                bboxCanvas.style.height = (img.height * scale) + 'px';
                
                currentDisplayScale = scale;
                
                const baseLineWidth = Number(document.getElementById('bboxLineWidth').value || 4);
                const baseFontSize = 12;
                const baseHandleSize = 6;
                const baseLabelHeight = 18;
                
                const adjustedLineWidth = Math.max(baseLineWidth, Math.min(baseLineWidth / scale, 8));
                const adjustedFontSize = Math.max(baseFontSize, Math.min(baseFontSize / scale, 24));
                const adjustedHandleSize = Math.max(baseHandleSize, Math.min(baseHandleSize / scale, 12));
                const adjustedLabelHeight = Math.max(baseLabelHeight, Math.min(baseLabelHeight / scale, 30));
                const adjustedLabelPadding = Math.max(2, Math.min(2 / scale, 6));
                
                currentAdjustedLineWidth = adjustedLineWidth;
                
                bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
                bboxCtx.drawImage(img, 0, 0, bboxCanvas.width, bboxCanvas.height);

                // Algorithm boxes are loaded as the initial expert-editable
                // annotation, so only the current solid boxes are displayed.
                currentBBoxes.forEach((bbox, idx) => {
                    const [x, y, w, h] = bbox.bbox;
                    const color = bbox.primary_class ? classColors[bbox.primary_class] || [255, 255, 0] : [255, 255, 0];
                    
                    bboxCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                    bboxCtx.lineWidth = idx === activeBBoxIndex ? adjustedLineWidth * 1.7 : adjustedLineWidth;
                    bboxCtx.strokeRect(x, y, w, h);
                    
                    if (bboxMode === 'view') {
                        const points = [
                            [x, y], [x + w, y], [x, y + h], [x + w, y + h]
                        ];
                        bboxCtx.fillStyle = 'white';
                        bboxCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                        bboxCtx.lineWidth = adjustedLineWidth;
                        points.forEach(([px, py]) => {
                            bboxCtx.fillRect(px - adjustedHandleSize/2, py - adjustedHandleSize/2, adjustedHandleSize, adjustedHandleSize);
                            bboxCtx.strokeRect(px - adjustedHandleSize/2, py - adjustedHandleSize/2, adjustedHandleSize, adjustedHandleSize);
                        });
                    }
                    
                    const label = bbox.primary_class || 'Unknown';
                    const labelWidth = label.length * (adjustedFontSize * 0.6);
                    bboxCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                    bboxCtx.fillRect(x, y - adjustedLabelHeight, labelWidth, adjustedLabelHeight);
                    bboxCtx.fillStyle = 'black';
                    bboxCtx.font = `${adjustedFontSize}px Arial`;
                    bboxCtx.fillText(label, x + adjustedLabelPadding, y - adjustedLabelPadding);
                });
                
                if (pendingBBox) {
                    const [x, y, w, h] = pendingBBox.bbox;
                    bboxCtx.strokeStyle = 'yellow';
                    bboxCtx.lineWidth = adjustedLineWidth;
                    bboxCtx.setLineDash([5 * (1/scale), 5 * (1/scale)]);
                    bboxCtx.strokeRect(x, y, w, h);
                    bboxCtx.setLineDash([]);
                    
                    const labelText = 'Waiting for class selection...';
                    const labelWidth = labelText.length * (adjustedFontSize * 0.6);
                    bboxCtx.fillStyle = 'yellow';
                    bboxCtx.fillRect(x, y - adjustedLabelHeight, labelWidth, adjustedLabelHeight);
                    bboxCtx.fillStyle = 'black';
                    bboxCtx.font = `${adjustedFontSize}px Arial`;
                    bboxCtx.fillText(labelText, x + adjustedLabelPadding, y - adjustedLabelPadding);
                }
            };
            setCanvasImageSource(img, currentImageData.image);
        }
        
        // Get mouse coordinates in original image
        function getImageCoords(e) {
            const rect = bboxCanvas.getBoundingClientRect();
            const displayScale = bboxCanvas.width / rect.width;
            const x = (e.clientX - rect.left) * displayScale;
            const y = (e.clientY - rect.top) * displayScale;
            return { x, y, displayX: x, displayY: y };
        }
        
        // Get mask image coordinates
        function getMaskImageCoords(e) {
            const rect = maskCanvas.getBoundingClientRect();
            const scaleX = maskCanvas.width / rect.width;
            const scaleY = maskCanvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }
        
        // Draw Mask Canvas
        function drawMaskCanvas() {
            if (!currentImageData || !currentImageData.image) return;
            
            const img = new Image();
            img.onload = () => {
                maskCanvas.width = img.width;
                maskCanvas.height = img.height;
                maskBrushLayer.width = img.width;
                maskBrushLayer.height = img.height;
                
                const maxWidth = maskCanvas.parentElement.clientWidth - 20;
                const maxHeight = 600;
                const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
                maskCanvas.style.width = (img.width * scale) + 'px';
                maskCanvas.style.height = (img.height * scale) + 'px';
                maskBrushLayer.style.width = (img.width * scale) + 'px';
                maskBrushLayer.style.height = (img.height * scale) + 'px';
                
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                maskCtx.drawImage(img, 0, 0);
                
                if (currentMask && currentMask.width === img.width && currentMask.height === img.height) {
                    maskCtx.globalAlpha = Number(document.getElementById('maskOpacity').value) / 100;
                    maskCtx.drawImage(buildDisplayMask(currentMask, img.width, img.height), 0, 0);
                    maskCtx.globalAlpha = 1.0;
                } else if (currentMask) {
                    maskCtx.globalAlpha = Number(document.getElementById('maskOpacity').value) / 100;
                    maskCtx.drawImage(buildDisplayMask(currentMask, img.width, img.height), 0, 0);
                    maskCtx.globalAlpha = 1.0;
                }
                
                maskPoints.forEach(p => {
                    maskCtx.beginPath();
                    maskCtx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
                    maskCtx.fillStyle = p.label === 1 ? '#00ff00' : '#ff0000';
                    maskCtx.fill();
                    maskCtx.strokeStyle = 'white';
                    maskCtx.lineWidth = 2;
                    maskCtx.stroke();
                });
            };
            setCanvasImageSource(img, currentImageData.image);
        }
        
        // BBox mode setting
        function setBBoxMode(mode) {
            bboxMode = mode;
            document.getElementById('bboxViewBtn').classList.toggle('active', mode === 'view');
            document.getElementById('bboxDrawBtn').classList.toggle('active', mode === 'draw');
            document.getElementById('bboxDeleteBtn').classList.toggle('active', mode === 'delete');
            bboxCanvas.style.cursor = mode === 'draw' ? 'crosshair' : mode === 'delete' ? 'not-allowed' : 'default';
        }
        
        // Mask mode setting
        function setMaskMode(mode) {
            maskMode = mode;
            document.getElementById('maskViewBtn').classList.toggle('active', mode === 'view');
            document.getElementById('maskBrushBtn').classList.toggle('active', mode === 'brush-add');
            document.getElementById('maskEraseBtn').classList.toggle('active', mode === 'brush-remove');

            const isInteractive = (mode === 'brush-add' || mode === 'brush-remove');
            maskBrushLayer.style.pointerEvents = isInteractive ? 'auto' : 'none';
            maskBrushLayer.style.cursor = isInteractive ? 'crosshair' : 'default';
            showStatus('maskStatus', isInteractive ? 'Brush is active: drag on the image to paint.' : 'Mask view mode.', 'success');
        }
        
        // Check if click is on resize handle
        function getResizeHandle(bbox, x, y) {
            const [bx, by, bw, bh] = bbox.bbox;
            const size = 8;
            const handles = [
                { name: 'tl', x: bx, y: by },
                { name: 'tr', x: bx + bw, y: by },
                { name: 'bl', x: bx, y: by + bh },
                { name: 'br', x: bx + bw, y: by + bh }
            ];
            for (const handle of handles) {
                if (Math.abs(x - handle.x) < size && Math.abs(y - handle.y) < size) {
                    return handle.name;
                }
            }
            return null;
        }
        
        // BBox Canvas events (keeping same logic from original)
        bboxCanvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            if (bboxMode === 'draw') {
                const coords = getImageCoords(e);
                bboxDrawing = { startX: coords.displayX, startY: coords.displayY };
            } else if (bboxMode === 'delete') {
                const coords = getImageCoords(e);
                const clickedIndex = currentBBoxes.findIndex(bbox => {
                    const [bx, by, bw, bh] = bbox.bbox;
                    return coords.x >= bx && coords.x <= bx + bw && coords.y >= by && coords.y <= by + bh;
                });
                if (clickedIndex >= 0) {
                    currentBBoxes.splice(clickedIndex, 1);
                    drawBBoxCanvas();
                    showStatus('bboxStatus', 'Deleted one BBox', 'success');
                }
            } else if (bboxMode === 'view') {
                const coords = getImageCoords(e);
                if (bboxMode === 'view') {
                    for (let i = currentBBoxes.length - 1; i >= 0; i--) {
                        const handle = getResizeHandle(currentBBoxes[i], coords.x, coords.y);
                        if (handle) {
                            activeBBoxIndex = i;
                            bboxResizing = { index: i, handle: handle, startX: coords.x, startY: coords.y, startBox: [...currentBBoxes[i].bbox] };
                            showStatus('bboxStatus', 'Resizing box — release to finish', 'success');
                            return;
                        }
                    }
                }
                let clickedIndex = -1;
                for (let i = currentBBoxes.length - 1; i >= 0; i--) {
                    const bbox = currentBBoxes[i];
                    const [bx, by, bw, bh] = bbox.bbox;
                    if (coords.x >= bx && coords.x <= bx + bw && coords.y >= by && coords.y <= by + bh) {
                        clickedIndex = i;
                        break;
                    }
                }
                if (clickedIndex >= 0) {
                    activeBBoxIndex = clickedIndex;
                    bboxDragging = { index: clickedIndex, startX: coords.x, startY: coords.y, startBox: [...currentBBoxes[clickedIndex].bbox] };
                    showStatus('bboxStatus', 'Moving box — drag, then release to finish', 'success');
                    drawBBoxCanvas();
                }
            }
        });
        
        bboxCanvas.addEventListener('mousemove', (e) => {
            if (bboxDrawing) {
                const coords = getImageCoords(e);
                drawBBoxCanvas();
                bboxCtx.strokeStyle = 'yellow';
                bboxCtx.lineWidth = currentAdjustedLineWidth;
                const dashSize = Math.max(5, Math.min(5 / currentDisplayScale, 15));
                bboxCtx.setLineDash([dashSize, dashSize]);
                bboxCtx.strokeRect(bboxDrawing.startX, bboxDrawing.startY, coords.displayX - bboxDrawing.startX, coords.displayY - bboxDrawing.startY);
                bboxCtx.setLineDash([]);
            } else if (bboxDragging) {
                const coords = getImageCoords(e);
                const bbox = currentBBoxes[bboxDragging.index];
                const [bx, by, bw, bh] = bboxDragging.startBox;
                const dx = coords.x - bboxDragging.startX;
                const dy = coords.y - bboxDragging.startY;
                bbox.bbox = [bx + dx, by + dy, bw, bh];
                drawBBoxCanvas();
            } else if (bboxResizing) {
                const coords = getImageCoords(e);
                const bbox = currentBBoxes[bboxResizing.index];
                const [bx, by, bw, bh] = bboxResizing.startBox;
                const dx = coords.x - bboxResizing.startX;
                const dy = coords.y - bboxResizing.startY;
                if (bboxResizing.handle === 'tl') {
                    bbox.bbox = [bx + dx, by + dy, bw - dx, bh - dy];
                } else if (bboxResizing.handle === 'tr') {
                    bbox.bbox = [bx, by + dy, bw + dx, bh - dy];
                } else if (bboxResizing.handle === 'bl') {
                    bbox.bbox = [bx + dx, by, bw - dx, bh + dy];
                } else if (bboxResizing.handle === 'br') {
                    bbox.bbox = [bx, by, bw + dx, bh + dy];
                }
                if (bbox.bbox[2] < 10) bbox.bbox[2] = 10;
                if (bbox.bbox[3] < 10) bbox.bbox[3] = 10;
                drawBBoxCanvas();
            }
        });
        
        bboxCanvas.addEventListener('mouseup', (e) => {
            if (bboxDrawing) {
                const coords = getImageCoords(e);
                const x = Math.min(bboxDrawing.startX, coords.displayX);
                const y = Math.min(bboxDrawing.startY, coords.displayY);
                const w = Math.abs(coords.displayX - bboxDrawing.startX);
                const h = Math.abs(coords.displayY - bboxDrawing.startY);
                if (w > 10 && h > 10) {
                    const bboxCoords = [x, y, w, h];
                    const newId = currentBBoxes.length > 0 ? Math.max(...currentBBoxes.map(b => b.id || 0)) + 1 : 0;
                    pendingBBox = { id: newId, bbox: bboxCoords };
                    showLabelDialog();
                }
                bboxDrawing = null;
            }
            bboxDragging = null;
            bboxResizing = null;
            activeBBoxIndex = null;
            drawBBoxCanvas();
        });
        document.addEventListener('mouseup', () => {
            if (bboxDragging || bboxResizing) {
                bboxDragging = null;
                bboxResizing = null;
                activeBBoxIndex = null;
                drawBBoxCanvas();
            }
        });
        
        // Mask Canvas interaction events (keeping same logic from original)
        maskBrushLayer.addEventListener('mousedown', (e) => {
            const pos = getMaskImageCoords(e);
            if (maskMode === 'point-pos' || maskMode === 'point-neg') {
                const label = maskMode === 'point-pos' ? 1 : 0;
                maskPoints.push({ x: pos.x, y: pos.y, label });
                drawMaskCanvas();
                showStatus('maskStatus', `Added ${maskPoints.length} points, click "Execute Refine" to apply SAM`, 'success');
            } else if (maskMode === 'brush-add' || maskMode === 'brush-remove') {
                maskIsDrawing = true;
                maskLastBrushPos = null;
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
                drawMaskBrush(pos.x, pos.y);
            }
        });
        
        // Execute SAM refine
        async function executeMaskRefine() {
            if (maskPoints.length === 0) {
                showStatus('maskStatus', 'Please add points first', 'error');
                return;
            }
            showStatus('maskStatus', 'Executing SAM refine...', '');
            try {
                const sourceImageData = await imageDataForApi();
                const points = maskPoints.map(p => [Math.round(p.x), Math.round(p.y)]);
                const labels = maskPoints.map(p => p.label);
                const bboxes = currentBBoxes.map(b => {
                    const [x, y, w, h] = b.bbox;
                    return [x, y, x + w, y + h];
                });
                const response = await fetch('/api/refine_mask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_data: sourceImageData,
                        mask_data: currentMask ? currentMask.toDataURL() : null,
                        points: points,
                        labels: labels,
                        bboxes: bboxes
                    })
                });
                const responseData = await response.json();
                if (responseData.success) {
                    const img = new Image();
                    img.onload = () => {
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = img.width;
                        tempCanvas.height = img.height;
                        const tempCtx = tempCanvas.getContext('2d');
                        tempCtx.drawImage(img, 0, 0);
                        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                        const pixelData = imageData.data;
                        const selectedClass = document.getElementById('maskClassSelect').value;
                        const color = classColors[selectedClass];
                        for (let i = 0; i < pixelData.length; i += 4) {
                            if (pixelData[i] > 0 || pixelData[i + 1] > 0 || pixelData[i + 2] > 0) {
                                pixelData[i] = color[0];
                                pixelData[i + 1] = color[1];
                                pixelData[i + 2] = color[2];
                                pixelData[i + 3] = 255;
                            }
                        }
                        tempCtx.putImageData(imageData, 0, 0);
                        if (!currentMask || currentMask.width !== img.width || currentMask.height !== img.height) {
                            currentMask = document.createElement('canvas');
                            currentMask.width = img.width;
                            currentMask.height = img.height;
                        }
                        const maskCtx = currentMask.getContext('2d');
                        maskCtx.clearRect(0, 0, currentMask.width, currentMask.height);
                        maskCtx.drawImage(tempCanvas, 0, 0);
                        originalMask = cloneImage(currentMask);
                        maskPoints = [];
                        drawMaskCanvas();
                        showStatus('maskStatus', 'SAM refine completed', 'success');
                    };
                    img.src = responseData.mask;
                } else {
                    showStatus('maskStatus', 'Refine failed: ' + responseData.error, 'error');
                }
            } catch (error) {
                console.error('SAM refine error:', error);
                showStatus('maskStatus', 'Refine failed: ' + error.message, 'error');
            }
        }
        
        maskBrushLayer.addEventListener('mousemove', (e) => {
            const pos = getMaskImageCoords(e);
            maskCursorPos = pos;
            if (maskIsDrawing) {
                drawMaskBrush(pos.x, pos.y);
            } else {
                updateBrushPreview();
            }
        });
        
        maskBrushLayer.addEventListener('mouseup', () => {
            if (maskIsDrawing) {
                maskIsDrawing = false;
                const brushDataURL = maskBrushLayer.toDataURL('image/png');
                applyBrushToMask(brushDataURL);
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
            }
            maskLastBrushPos = null;
        });
        
        maskBrushLayer.addEventListener('mouseleave', () => {
            if (maskIsDrawing) {
                maskIsDrawing = false;
                const brushDataURL = maskBrushLayer.toDataURL('image/png');
                applyBrushToMask(brushDataURL);
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
            }
            maskCursorPos = null;
            maskLastBrushPos = null;
            maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
        });
        
        // Draw brush
        function drawMaskBrush(x, y) {
            const brushSize = parseInt(document.getElementById('brushSize').value);
            maskBrushCtx.beginPath();
            if (maskLastBrushPos) {
                maskBrushCtx.moveTo(maskLastBrushPos.x, maskLastBrushPos.y);
                maskBrushCtx.lineTo(x, y);
                maskBrushCtx.lineWidth = brushSize * 2;
                maskBrushCtx.lineCap = 'round';
                maskBrushCtx.strokeStyle = maskMode === 'brush-add' ? 'rgba(0, 255, 0, 1.0)' : 'rgba(255, 0, 0, 1.0)';
                maskBrushCtx.stroke();
            } else {
                maskBrushCtx.arc(x, y, brushSize, 0, 2 * Math.PI);
                maskBrushCtx.fillStyle = maskMode === 'brush-add' ? 'rgba(0, 255, 0, 1.0)' : 'rgba(255, 0, 0, 1.0)';
                maskBrushCtx.fill();
            }
            maskLastBrushPos = { x, y };
        }
        
        // Update brush preview
        function updateBrushPreview() {
            if (!maskIsDrawing && maskCursorPos) {
                const brushSize = parseInt(document.getElementById('brushSize').value);
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
                if (maskMode === 'brush-add' || maskMode === 'brush-remove') {
                    maskBrushCtx.beginPath();
                    maskBrushCtx.arc(maskCursorPos.x, maskCursorPos.y, brushSize, 0, 2 * Math.PI);
                    maskBrushCtx.strokeStyle = maskMode === 'brush-add' ? 'lime' : 'red';
                    maskBrushCtx.lineWidth = 2;
                    maskBrushCtx.stroke();
                }
            }
        }
        
        // Apply brush to mask
        function applyBrushToMask(brushDataURL) {
            const brushImg = new Image();
            brushImg.onload = () => {
                const img = new Image();
                img.onload = () => {
                    if (!currentMask) {
                        currentMask = document.createElement('canvas');
                        currentMask.width = img.width;
                        currentMask.height = img.height;
                        const maskCtx2 = currentMask.getContext('2d');
                        maskCtx2.clearRect(0, 0, currentMask.width, currentMask.height);
                    } else {
                        if (currentMask.width !== img.width || currentMask.height !== img.height) {
                            const oldMask = currentMask;
                            currentMask = document.createElement('canvas');
                            currentMask.width = img.width;
                            currentMask.height = img.height;
                            const maskCtx2 = currentMask.getContext('2d');
                            maskCtx2.drawImage(oldMask, 0, 0, img.width, img.height);
                        }
                    }
                    const maskCtx2 = currentMask.getContext('2d');
                    applyBrushOperation(maskCtx2, brushImg, img.width, img.height);
                };
                setCanvasImageSource(img, currentImageData.image);
            };
            brushImg.src = brushDataURL;
        }
        
        // Apply brush operation
        function applyBrushOperation(ctx, brushImg, maskWidth, maskHeight) {
            if (maskMode === 'brush-add') {
                const selectedClass = document.getElementById('maskClassSelect').value;
                const color = classColors[selectedClass];
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskWidth;
                tempCanvas.height = maskHeight;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(brushImg, 0, 0, maskWidth, maskHeight);
                const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (imageData.data[i + 3] > 0) {
                        imageData.data[i] = color[0];
                        imageData.data[i + 1] = color[1];
                        imageData.data[i + 2] = color[2];
                        imageData.data[i + 3] = 255;
                    }
                }
                tempCtx.putImageData(imageData, 0, 0);
                ctx.drawImage(tempCanvas, 0, 0);
            } else if (maskMode === 'brush-remove') {
                const imageData = ctx.getImageData(0, 0, maskWidth, maskHeight);
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskWidth;
                tempCanvas.height = maskHeight;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(brushImg, 0, 0, maskWidth, maskHeight);
                const brushImageData = tempCtx.getImageData(0, 0, maskWidth, maskHeight);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (brushImageData.data[i + 3] > 0 && imageData.data[i + 3] > 0) {
                        imageData.data[i] = 0;
                        imageData.data[i + 1] = 0;
                        imageData.data[i + 2] = 0;
                    }
                }
                ctx.putImageData(imageData, 0, 0);
            }
            drawMaskCanvas();
        }
        
        // Show label selection dialog
        function showLabelDialog() {
            if (!pendingBBox) return;
            const dialog = document.getElementById('labelDialog');
            const optionsDiv = document.getElementById('labelOptions');
            optionsDiv.innerHTML = `
                <label style="display:block; margin: 8px 0 4px; font-weight:bold;">Primary class</label>
                <select id="dialogPrimaryClass" style="width:100%; padding:8px;"></select>
                <label style="display:block; margin: 12px 0 4px; font-weight:bold;">Subtype</label>
                <select id="dialogSubtype" style="width:100%; padding:8px;"></select>
                <button id="confirmLabelButton" style="width:100%; margin-top:16px; padding:10px; background:#2196F3; color:white; border:0; border-radius:4px; cursor:pointer; font-weight:bold;">Confirm BBox</button>
            `;
            const primarySelect = document.getElementById('dialogPrimaryClass');
            defectClasses.forEach(className => {
                const option = document.createElement('option');
                option.value = className;
                option.textContent = className;
                primarySelect.appendChild(option);
            });
            primarySelect.onchange = updateDialogSubtypes;
            updateDialogSubtypes();
            document.getElementById('confirmLabelButton').onclick = handleLabelSelected;
            dialog.style.display = 'flex';
        }
        
        function updateDialogSubtypes() {
            const primarySelect = document.getElementById('dialogPrimaryClass');
            const subtypeSelect = document.getElementById('dialogSubtype');
            if (!primarySelect || !subtypeSelect) return;
            subtypeSelect.innerHTML = '';
            (subtypesByPrimary[primarySelect.value] || []).forEach(subType => {
                const option = document.createElement('option');
                option.value = subType;
                option.textContent = subType;
                subtypeSelect.appendChild(option);
            });
        }

        // Save one explicitly selected ontology pair for the pending box.
        function handleLabelSelected() {
            if (!pendingBBox) return;
            const className = document.getElementById('dialogPrimaryClass').value;
            const subType = document.getElementById('dialogSubtype').value;
            if (!className || !subType || !(subtypesByPrimary[className] || []).includes(subType)) {
                showStatus('bboxStatus', 'Select a valid primary class and subtype.', 'error');
                return;
            }
            const newBBox = {
                id: pendingBBox.id,
                bbox: pendingBBox.bbox,
                primary_class: className,
                sub_type: subType
            };
            currentBBoxes.push(newBBox);
            drawBBoxCanvas();
            showStatus('bboxStatus', `Added one BBox: ${className} / ${subType}`, 'success');
            cancelLabelSelection();
        }
        
        // Cancel label selection
        function cancelLabelSelection() {
            pendingBBox = null;
            const dialog = document.getElementById('labelDialog');
            dialog.style.display = 'none';
        }
        
        function resetBBoxesToAlgorithm() {
            if (!confirm('恢复为算法初始检测框？当前框的未保存修改会被替换；点击 Save Changes 后才会写入保存结果。')) return;
            currentBBoxes = algorithmBBoxes.map((bbox, index) => ({
                id: index,
                bbox: [...bbox.bbox],
                primary_class: bbox.primary_class,
                sub_type: bbox.sub_type || ''
            }));
            pendingBBox = null;
            cancelLabelSelection();
            setBBoxMode('view');
            drawBBoxCanvas();
            showStatus('bboxStatus', '已恢复算法初始检测框；请保存以提交', 'success');
        }

        async function resetMaskToAlgorithm() {
            if (!confirm('恢复为算法初始 Mask？当前 Mask 的未保存修改会被替换；点击 Save Changes 后才会写入保存结果。')) return;
            currentMask = algorithmMask ? await cloneImage(algorithmMask) : null;
            originalMask = algorithmMask ? await cloneImage(algorithmMask) : null;
            maskPoints = [];
            maskLastBrushPos = null;
            maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
            setMaskMode('view');
            drawMaskCanvas();
            showStatus('maskStatus', '已恢复算法初始 Mask；请保存以提交', 'success');
        }

        // Clear BBoxes
        function clearBBoxes() {
            if (confirm('Are you sure you want to clear all BBoxes?')) {
                currentBBoxes = [];
                pendingBBox = null;
                cancelLabelSelection();
                drawBBoxCanvas();
                showStatus('bboxStatus', 'Cleared all BBoxes', 'success');
            }
        }
        
        // Clear Mask
        function clearMask() {
            if (confirm('Are you sure you want to clear the Mask?')) {
                currentMask = null;
                originalMask = null;
                maskPoints = [];
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
                drawMaskCanvas();
                showStatus('maskStatus', 'Cleared Mask', 'success');
            }
        }
        
        // Save changes
        async function saveChanges() {
            if (pendingBBox) {
                cancelLabelSelection();
                showStatus('bboxStatus', 'Cancelled pending box', 'error');
                return;
            }
            try {
                setSaveFeedback('Saving expert annotation…');
                let maskDataURL = null;
                if (currentMask) {
                    try {
                        if (currentMask instanceof HTMLCanvasElement) {
                            maskDataURL = currentMask.toDataURL('image/png');
                        } else {
                            showStatus('maskStatus', 'Mask format error, cannot save', 'error');
                            return;
                        }
                    } catch (e) {
                        showStatus('maskStatus', 'Failed to convert mask: ' + e.message, 'error');
                        return;
                    }
                }
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        index: currentIndex,
                        dataset: currentDataset,
                        primary_class: currentPrimaryClass,
                        bboxes: currentBBoxes,
                        mask: maskDataURL,
                        actor: 'expert',
                        reason_note: ''
                    })
                });
                const data = await response.json();
                if (data.success) {
                    showStatus('bboxStatus', `Saved expert revision v${data.revision || '?'} successfully`, 'success');
                    showStatus('maskStatus', maskDataURL ? 'Mask saved successfully' : 'Mask not modified', 'success');
                    setSaveFeedback(`Saved successfully · revision v${data.revision || '?'}`, 'success');
                    loadedAnnotationState = annotationStateSignature();
                    setTimeout(() => {
                        loadImage(currentIndex);
                    }, 500);
                } else {
                    setSaveFeedback('Save failed: ' + data.error, 'error');
                    showStatus('bboxStatus', 'Save failed: ' + data.error, 'error');
                    showStatus('maskStatus', 'Save failed: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Save failed:', error);
                setSaveFeedback('Save failed: ' + error.message, 'error');
                showStatus('bboxStatus', 'Save failed: ' + error.message, 'error');
                showStatus('maskStatus', 'Save failed: ' + error.message, 'error');
            }
        }

        async function exportFinalDataset() {
            if (hasUnsavedChanges()) {
                showStatus('bboxStatus', '当前图片有未保存修改，请先保存后再导出完整数据集', 'error');
                return;
            }
            const button = document.getElementById('exportButton');
            const originalText = button.textContent;
            button.disabled = true;
            button.textContent = '正在导出…';
            try {
                const response = await fetch('/api/export_final_dataset', { method: 'POST' });
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.error || '导出失败');
                setSaveFeedback(`已导出完整结果：${data.total} 张图片`, 'success');
                showStatus('bboxStatus', `完整结果已导出至 ${data.folder}`, 'success');
                showStatus('maskStatus', `完整结果已导出：${data.total} 张图片`, 'success');
            } catch (error) {
                console.error('Export failed:', error);
                showStatus('bboxStatus', `导出失败：${error.message}`, 'error');
            } finally {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
        
        // Show status
        function showStatus(elementId, message, type = '') {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = 'status ' + type;
            setTimeout(() => {
                element.textContent = '';
                element.className = 'status';
            }, 3000);
        }

        let saveFeedbackTimer = null;
        function setSaveFeedback(message, state = 'saving') {
            const feedback = document.getElementById('saveFeedback');
            const button = document.getElementById('saveButton');
            if (saveFeedbackTimer) clearTimeout(saveFeedbackTimer);
            feedback.textContent = message;
            feedback.className = `save-feedback visible ${state === 'saving' ? '' : state}`;
            if (state === 'saving') {
                button.disabled = true;
                button.classList.add('saving');
                button.textContent = 'Saving…';
                return;
            }
            button.disabled = false;
            button.classList.remove('saving');
            button.textContent = 'Save Changes';
            saveFeedbackTimer = setTimeout(() => feedback.classList.remove('visible'), 2800);
        }
        
        // Run detection
        async function runDetection() {
            if (!currentImageData || !currentImageData.image) {
                showStatus('bboxStatus', 'Please load image first', 'error');
                return;
            }
            const model = document.getElementById('detectionModelSelect').value;
            showStatus('bboxStatus', 'Running detection...', '');
            try {
                const sourceImageData = await imageDataForApi();
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_data: sourceImageData,
                        model: model,
                        filename: currentImageData.filename || 'unknown.jpg'
                    })
                });
                const data = await response.json();
                if (data.success && data.bboxes) {
                    const maxId = currentBBoxes.length > 0 ? Math.max(...currentBBoxes.map(b => b.id || 0)) : -1;
                    let nextId = maxId + 1;
                    const existingCount = currentBBoxes.length;
                    data.bboxes.forEach((bbox) => {
                        currentBBoxes.push({
                            id: nextId++,
                            bbox: bbox.bbox,
                            primary_class: bbox.primary_class,
                            sub_type: bbox.sub_type
                        });
                    });
                    const newCount = currentBBoxes.length - existingCount;
                    drawBBoxCanvas();
                    showStatus('bboxStatus', `Detection completed, added ${newCount} boxes (total ${currentBBoxes.length})`, 'success');
                } else {
                    showStatus('bboxStatus', 'Detection failed: ' + (data.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                console.error('Detection failed:', error);
                showStatus('bboxStatus', 'Detection failed: ' + error.message, 'error');
            }
        }
        
        // Add to candidates
        async function addToCandidates() {
            if (!currentImageData) {
                showStatus('bboxStatus', 'No image loaded', 'error');
                return;
            }
            let primaryClass = null;
            if (currentBBoxes.length > 0) {
                const classCount = {};
                currentBBoxes.forEach(bbox => {
                    const pc = bbox.primary_class;
                    if (pc) {
                        classCount[pc] = (classCount[pc] || 0) + 1;
                    }
                });
                const classNames = Object.keys(classCount);
                if (classNames.length > 0) {
                    primaryClass = classNames.reduce((a, b) => classCount[a] > classCount[b] ? a : b);
                    if (classNames.length > 1) {
                        const counts = classNames.map(c => `${c}: ${classCount[c]}`).join(', ');
                        console.log(`[Candidate] Image contains multiple classes: ${counts}, selected most common: ${primaryClass}`);
                    }
                }
            }
            if (!primaryClass) {
                showStatus('bboxStatus', 'Current image has no bbox annotations, cannot add to candidates', 'error');
                return;
            }
            try {
                const response = await fetch('/api/add_candidate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        index: currentIndex,
                        dataset: currentDataset,
                        primary_class: primaryClass,
                        filename: currentImageData.filename
                    })
                });
                const data = await response.json();
                if (data.success) {
                    showStatus('bboxStatus', `Added to candidates (${primaryClass}): ${data.message}`, 'success');
                } else {
                    showStatus('bboxStatus', 'Failed to add to candidates: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Failed to add to candidates:', error);
                showStatus('bboxStatus', 'Failed to add to candidates: ' + error.message, 'error');
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            const isInputElement = e.target.tagName === 'INPUT' || 
                                   e.target.tagName === 'TEXTAREA' || 
                                   e.target.tagName === 'SELECT';
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                if (currentIndex > 0) loadImage(currentIndex - 1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                loadImage(currentIndex + 1);
            } else if ((e.key === 'c' || e.key === 'C') && !e.ctrlKey && !e.metaKey) {
                if (e.target.tagName === 'SELECT') {
                    e.preventDefault();
                    e.stopPropagation();
                    e.target.blur();
                } else if (!isInputElement) {
                    e.preventDefault();
                    addToCandidates();
                }
            } else if ((e.key === 's' || e.key === 'S') && !e.ctrlKey && !e.metaKey) {
                if (e.target.tagName === 'SELECT') {
                    e.preventDefault();
                    e.stopPropagation();
                    e.target.blur();
                } else if (!isInputElement) {
                    e.preventDefault();
                    saveChanges();
                }
            }
        });
        
        // Initialize
        fetch('/api/runtime').then(response => response.json()).then(runtime => {
            if (runtime.cloud_storage) {
                document.getElementById('localPathButton').style.display = 'none';
                document.getElementById('infoDiv').textContent = 'Cloud mode: upload a dataset folder. Data is persisted to object storage; local computer paths are unavailable.';
            } else {
                document.getElementById('ossBulkButton').style.display = 'none';
            }
        }).catch(() => {});
        updateImageList();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/runtime")
def api_runtime():
    return jsonify({"cloud_storage": R2.enabled, "retention_days": int(os.environ.get("R2_RETENTION_DAYS", "31"))})


@app.route("/api/images")
def api_images():
    """Get total number of filtered images"""
    dataset = request.args.get("dataset") or None
    primary_class = request.args.get("primary_class") or None
    
    filtered = _get_filtered_images(dataset, primary_class)
    return jsonify({"total": len(filtered)})


@app.route("/api/export_final_dataset", methods=["POST"])
def api_export_final_dataset():
    """Create a complete, canonical final dataset from expert-or-algorithm results."""
    try:
        _load_image_list()
        export_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_final"
        # A direct import can be several GB.  Create its export with R2-to-R2
        # copies instead of downloading every image into Render's 10GB disk.
        if _active_dataset_manifest and R2.enabled and _active_dataset_remote_root:
            root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
            prefix = root + "/" if root else ""
            remote_export = f"{_active_dataset_remote_root}/{prefix}final_exports/{export_id}"
            files = list(_active_dataset_manifest.get("files", []))
            samples = []
            for image_info in _image_list_cache:
                stem, name = image_info["stem"], image_info["name"]
                image_source = f"{prefix}images/{name}"
                R2.copy_file(f"{_active_dataset_remote_root}/{image_source}", f"{remote_export}/images/{name}")
                expert_label = f"{prefix}expert_labels/{stem}.json"
                algorithm_label = next((item for item in files if item.startswith(prefix + "detections/") or item.startswith(prefix + "algorithm_labels/") or item.startswith(prefix + "labels/") if Path(item).stem.startswith(stem) or Path(item).stem.startswith(_sample_key(stem))), None)
                label_source = expert_label if R2.get_bytes(f"{_active_dataset_remote_root}/{expert_label}") is not None else algorithm_label
                if label_source:
                    R2.copy_file(f"{_active_dataset_remote_root}/{label_source}", f"{remote_export}/labels/{stem}.json")
                expert_mask = f"{prefix}expert_masks/{stem}_mask.png"
                algorithm_mask = next((item for item in files if any(item.startswith(prefix + folder + "/") for folder in ("masks", "algorithm_masks")) and (Path(item).stem.startswith(stem) or Path(item).stem.startswith(_sample_key(stem)))), None)
                mask_source = expert_mask if R2.get_bytes(f"{_active_dataset_remote_root}/{expert_mask}") is not None else algorithm_mask
                if mask_source:
                    R2.copy_file(f"{_active_dataset_remote_root}/{mask_source}", f"{remote_export}/masks/{stem}_mask.png")
                samples.append({"sample_id": stem, "image": f"images/{name}", "label": f"labels/{stem}.json" if label_source else None, "mask": f"masks/{stem}_mask.png" if mask_source else None})
            manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "total_images": len(samples), "selection_rule": "expert result when present; otherwise algorithm result", "samples": samples}
            R2.put_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"), f"{remote_export}/manifest.json")
            return jsonify({"success": True, "total": len(samples), "folder": f"final_exports/{export_id}", "cloud_persisted": True, "direct": True})
        export_dir = FINAL_EXPORTS_DIR / export_id
        images_output = export_dir / "images"
        labels_output = export_dir / "labels"
        masks_output = export_dir / "masks"
        metadata_output = export_dir / "metadata"
        for directory in (images_output, labels_output, masks_output, metadata_output):
            directory.mkdir(parents=True, exist_ok=False)

        manifest_samples: List[Dict[str, Any]] = []
        for image_info in _image_list_cache:
            stem = image_info["stem"]
            image_path = Path(image_info["path"])
            shutil.copy2(image_path, images_output / image_path.name)

            expert_label_path = EXPERT_LABELS_DIR / f"{stem}.json"
            label_source = "expert" if expert_label_path.exists() else "algorithm"
            canonical_boxes = _load_bboxes_for_image(stem)
            canonical_label = {
                "image_path": image_path.name,
                "bboxes": [
                    {
                        "instance_id": f"{stem}_{index}",
                        "taxonomy": {
                            "primary_class": bbox.get("primary_class"),
                            "sub_type": bbox.get("sub_type", ""),
                        },
                        "bbox": bbox["bbox"],
                    }
                    for index, bbox in enumerate(canonical_boxes)
                ],
            }
            label_output_path = labels_output / f"{stem}.json"
            label_output_path.write_text(json.dumps(canonical_label, ensure_ascii=False, indent=2), encoding="utf-8")

            expert_mask_path = _mask_path(EXPERT_MASKS_DIR, stem)
            baseline_mask_path = _baseline_mask_path(stem)
            selected_mask_path = expert_mask_path or baseline_mask_path
            mask_source = "expert" if expert_mask_path else "algorithm" if baseline_mask_path else None
            mask_output_path = None
            if selected_mask_path:
                mask_output_path = masks_output / f"{stem}_mask.png"
                with Image.open(selected_mask_path) as mask_image:
                    mask_image.save(mask_output_path, "PNG")

            metadata_path = _metadata_path(stem)
            if metadata_path:
                shutil.copy2(metadata_path, metadata_output / metadata_path.name)

            manifest_samples.append({
                "sample_id": stem,
                "image": f"images/{image_path.name}",
                "label": f"labels/{label_output_path.name}",
                "mask": f"masks/{mask_output_path.name}" if mask_output_path else None,
                "label_source": label_source,
                "mask_source": mask_source,
                "metadata": f"metadata/{metadata_path.name}" if metadata_path else None,
            })

        if DECISION_EVENTS_PATH.exists():
            shutil.copy2(DECISION_EVENTS_PATH, export_dir / "decision_events.jsonl")
        if REVISIONS_DIR.exists():
            shutil.copytree(REVISIONS_DIR, export_dir / "revisions")
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(manifest_samples),
            "label_format": "bbox = [x, y, width, height]",
            "selection_rule": "expert result when present; otherwise immutable algorithm result",
            "samples": manifest_samples,
        }
        (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _sync_active_dataset_tree(export_dir)
        return jsonify({
            "success": True,
            "total": len(manifest_samples),
            "folder": str(export_dir.relative_to(BASE_DIR)),
            "cloud_persisted": R2.enabled,
        })
    except Exception as error:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/api/open_local_dataset", methods=["POST"])
def api_open_local_dataset():
    """Use an existing local dataset in place and write results below annotation_output/."""
    if R2.enabled:
        return jsonify({"success": False, "error": "在线部署无法访问访问者电脑路径，请使用“导入文件夹”。"}), 400
    data = request.get_json(silent=True) or {}
    raw_path = str(data.get("path") or "").strip()
    if not raw_path:
        return jsonify({"success": False, "error": "请输入本地数据集根目录。"}), 400
    try:
        dataset_root = Path(raw_path).expanduser().resolve()
        if not dataset_root.is_dir():
            raise ValueError("该路径不存在或不是文件夹。")
        has_labels = any((dataset_root / name).is_dir() for name in ("detections", "algorithm_labels", "labels"))
        has_masks = any((dataset_root / name).is_dir() for name in ("masks", "algorithm_masks"))
        if not (dataset_root / "images").is_dir() or not has_labels or not has_masks:
            raise ValueError("目录需包含 images/、detections/（或 labels/）和 masks/ 子文件夹。")
        _configure_dataset_root(dataset_root, output_under_dataset=True)
        _load_image_list()
        if not _image_list_cache:
            raise ValueError("images/ 子文件夹中没有可读取的 JPG 或 PNG 图片。")
        return jsonify({
            "success": True,
            "dataset_name": dataset_root.name,
            "total": len(_image_list_cache),
            "output_folder": str(OUTPUT_ROOT),
        })
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


def _safe_upload_relative_path(raw_path: Any) -> str:
    """Validate a browser supplied relative filename before using it as an R2 key."""
    relative_name = str(raw_path or "").replace("\\", "/")
    parts = PurePosixPath(relative_name).parts
    if not relative_name or relative_name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("导入文件包含无效的路径。")
    return "/".join(parts)


def _find_imported_dataset_root(destination: Path) -> Optional[Path]:
    candidates = [destination]
    candidates.extend(path.parent for path in destination.rglob("images") if path.is_dir())
    return next(
        (candidate for candidate in sorted(candidates, key=lambda path: (len(path.parts), str(path)))
         if (candidate / "images").is_dir()
         and ((candidate / "detections").is_dir() or (candidate / "algorithm_labels").is_dir() or (candidate / "labels").is_dir())
         and ((candidate / "masks").is_dir() or (candidate / "algorithm_masks").is_dir())),
        None,
    )


def _direct_upload_ttl() -> int:
    """Keep large-batch browser upload URLs usable for a realistic session."""
    return max(3600, min(int(os.environ.get("DIRECT_UPLOAD_URL_TTL", "21600")), 86400))


def _valid_direct_upload_id(upload_id: str) -> bool:
    suffix = upload_id[len("direct_"):] if upload_id.startswith("direct_") else ""
    return len(suffix) == 32 and all(ch in "0123456789abcdef" for ch in suffix)


@app.route("/api/oss_bulk/create", methods=["POST"])
def api_oss_bulk_create():
    """Create a unique OSS prefix for an operator using ossutil locally."""
    if not isinstance(R2, OSSStorage) or not R2.enabled:
        return jsonify({"success": False, "error": "OSS 批量上传仅在已配置阿里云 OSS 时可用。"}), 400
    upload_id = "direct_" + uuid.uuid4().hex
    relative = f"datasets/{upload_id}"
    return jsonify({
        "success": True,
        "upload_id": upload_id,
        "bucket": R2.bucket_name,
        "object_prefix": R2.key(relative) + "/",
        "destination": f"oss://{R2.bucket_name}/{R2.key(relative)}/",
    })


@app.route("/api/oss_bulk/register", methods=["POST"])
def api_oss_bulk_register():
    """Build the app manifest for files uploaded outside the browser."""
    if not R2.enabled:
        return jsonify({"success": False, "error": "对象存储尚未配置。"}), 400
    upload_id = str((request.get_json(silent=True) or {}).get("upload_id") or "")
    if not _valid_direct_upload_id(upload_id):
        return jsonify({"success": False, "error": "批次标识无效。请使用网页生成的批次标识。"}), 400
    try:
        remote_root = f"datasets/{upload_id}"
        files = sorted(name for name in R2.list_relative(remote_root) if name != ".upload_manifest.json")
        if not files:
            raise ValueError("该批次尚未发现上传文件。请确认 ossutil 的目标路径。")
        manifest = {"files": files, "created_at": datetime.now(timezone.utc).isoformat(), "upload_method": "ossutil"}
        R2.put_bytes(json.dumps(manifest, ensure_ascii=False).encode("utf-8"), f"{remote_root}/.upload_manifest.json")
        return jsonify({"success": True, "upload_id": upload_id, "files": len(files)})
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/api/direct_upload/start", methods=["POST"])
def api_direct_upload_start():
    if not R2.enabled:
        return jsonify({"success": False, "error": "直传需要先配置 OSS 或 Cloudflare R2。"}), 400
    data = request.get_json(silent=True) or {}
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        return jsonify({"success": False, "error": "请选择要上传的文件夹。"}), 400
    max_files = int(os.environ.get("DIRECT_UPLOAD_MAX_FILES", "20000"))
    if len(entries) > max_files:
        return jsonify({"success": False, "error": f"单次最多支持 {max_files} 个文件。"}), 400
    try:
        import_id = "direct_" + uuid.uuid4().hex
        expires_in = _direct_upload_ttl()
        # R2 can configure CORS through its S3 API.  OSS CORS is configured in
        # the OSS console once, and then uses the same signed-upload protocol.
        if isinstance(R2, R2Storage):
            R2.ensure_browser_upload_cors()
        seen = set()
        uploads: List[Dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("上传文件信息格式错误。")
            relative_name = _safe_upload_relative_path(entry.get("path"))
            if relative_name in seen:
                raise ValueError(f"存在重复文件路径：{relative_name}")
            seen.add(relative_name)
            if int(entry.get("size") or 0) < 0:
                raise ValueError("文件大小无效。")
            content_type = str(entry.get("content_type") or "").strip()[:200]
            remote_relative = f"datasets/{import_id}/{relative_name}"
            uploads.append({"path": relative_name, "url": R2.presign_put(remote_relative, expires_in=expires_in, content_type=content_type)})
        # Persist the file list before the browser begins uploading.  Completion
        # can now validate and open the batch without listing/downloading 3.8GB.
        manifest = {"files": sorted(seen), "created_at": datetime.now(timezone.utc).isoformat()}
        R2.put_bytes(json.dumps(manifest, ensure_ascii=False).encode("utf-8"), f"datasets/{import_id}/.upload_manifest.json")
        return jsonify({"success": True, "upload_id": import_id, "uploads": uploads, "expires_in": expires_in})
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/api/direct_upload/resume", methods=["POST"])
def api_direct_upload_resume():
    """Return fresh signed URLs only for files absent from an interrupted batch."""
    if not R2.enabled:
        return jsonify({"success": False, "error": "直传需要先配置 OSS 或 Cloudflare R2。"}), 400
    data = request.get_json(silent=True) or {}
    upload_id = str(data.get("upload_id") or "")
    entries = data.get("files")
    suffix = upload_id[len("direct_"):] if upload_id.startswith("direct_") else ""
    if len(suffix) != 32 or not all(ch in "0123456789abcdef" for ch in suffix):
        return jsonify({"success": False, "error": "上传批次标识无效。"}), 400
    if not isinstance(entries, list) or not entries:
        return jsonify({"success": False, "error": "请选择要继续上传的文件夹。"}), 400
    try:
        manifest_bytes = R2.get_bytes(f"datasets/{upload_id}/.upload_manifest.json")
        if not manifest_bytes:
            raise ValueError("未找到之前的上传批次，请重新开始导入。")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        expected = set(str(item) for item in manifest.get("files", []))
        supplied: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("上传文件信息格式错误。")
            relative_name = _safe_upload_relative_path(entry.get("path"))
            if relative_name in supplied:
                raise ValueError(f"存在重复文件路径：{relative_name}")
            supplied[relative_name] = entry
        if set(supplied) != expected:
            raise ValueError("所选文件夹与未完成批次不一致，请重新选择原文件夹。")

        expires_in = _direct_upload_ttl()
        uploads: List[Dict[str, str]] = []
        for relative_name in sorted(expected):
            if R2.object_exists(f"datasets/{upload_id}/{relative_name}"):
                continue
            content_type = str(supplied[relative_name].get("content_type") or "").strip()[:200]
            uploads.append({
                "path": relative_name,
                "url": R2.presign_put(f"datasets/{upload_id}/{relative_name}", expires_in=expires_in, content_type=content_type),
            })
        return jsonify({"success": True, "upload_id": upload_id, "uploads": uploads, "completed": len(expected) - len(uploads), "expires_in": expires_in})
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/api/direct_upload/complete", methods=["POST"])
def api_direct_upload_complete():
    if not R2.enabled:
        return jsonify({"success": False, "error": "直传需要先配置 OSS 或 Cloudflare R2。"}), 400
    upload_id = str((request.get_json(silent=True) or {}).get("upload_id") or "")
    suffix = upload_id[len("direct_"):] if upload_id.startswith("direct_") else ""
    if len(suffix) != 32 or not all(ch in "0123456789abcdef" for ch in suffix):
        return jsonify({"success": False, "error": "上传批次标识无效。"}), 400
    remote_root = f"datasets/{upload_id}"
    try:
        manifest_bytes = R2.get_bytes(f"{remote_root}/.upload_manifest.json")
        if not manifest_bytes:
            raise ValueError("上传清单不存在，请重新开始导入。")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        files = [str(item) for item in manifest.get("files", [])]
        missing = [name for name in files if not R2.object_exists(f"{remote_root}/{name}")]
        if missing:
            raise ValueError(f"仍有 {len(missing)} 个文件未上传完成，请重新选择同一文件夹后继续上传。")
        candidates = []
        for name in files:
            parts = PurePosixPath(name).parts
            if "images" in parts:
                candidates.append("/".join(parts[:parts.index("images")]))
        relative_root = next((root for root in sorted(set(candidates), key=len)
            if any(item.startswith((root + "/" if root else "") + "images/") for item in files)
            and any(any(item.startswith((root + "/" if root else "") + folder + "/") for item in files)
                    for folder in ("detections", "algorithm_labels", "labels"))
            and any(any(item.startswith((root + "/" if root else "") + folder + "/") for item in files)
                    for folder in ("masks", "algorithm_masks"))), None)
        if relative_root is None:
            raise ValueError("未找到 images/、detections/（或 labels/）和 masks/ 子文件夹。")
        manifest["dataset_relative_root"] = relative_root
        destination = IMPORT_ROOT / f"dataset_{upload_id}"
        destination.mkdir(parents=True, exist_ok=True)
        dataset_root = destination.joinpath(*PurePosixPath(relative_root).parts)
        for folder in ("images", "labels", "detections", "algorithm_labels", "masks", "algorithm_masks", "metadata", "expert_labels", "expert_masks", "revisions"):
            (dataset_root / folder).mkdir(parents=True, exist_ok=True)
        global _active_dataset_loaded, _active_dataset_local_import_root, _active_dataset_manifest
        _active_dataset_loaded = True
        _active_dataset_local_import_root = destination
        _active_dataset_manifest = manifest
        _configure_dataset_root(dataset_root)
        _apply_direct_manifest_layout(dataset_root)
        _load_image_list()
        if not _image_list_cache:
            raise ValueError("images/ 子文件夹中没有可读取的 JPG 或 PNG 图片。")
        _remember_remote_dataset(destination, dataset_root, remote_root)
        return jsonify({"success": True, "dataset_name": dataset_root.name, "total": len(_image_list_cache), "imported_files": len(files), "cloud_persisted": True, "direct": True})
    except Exception as error:
        print(f"[Direct import] Failed: {error}")
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/api/import_dataset", methods=["POST"])
def api_import_dataset():
    """Import a browser-selected dataset folder and make it the active batch."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "请选择一个包含数据文件的文件夹。"}), 400

    import_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = IMPORT_ROOT / f"dataset_{import_id}"
    destination.mkdir(parents=True, exist_ok=False)
    destination_root = destination.resolve()

    try:
        saved_count = 0
        for uploaded in files:
            relative_name = (uploaded.filename or "").replace("\\", "/")
            parts = PurePosixPath(relative_name).parts
            if not relative_name or relative_name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
                raise ValueError("导入文件包含无效的路径。")
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if destination_root not in target.resolve().parents:
                raise ValueError("导入文件路径超出目标目录。")
            uploaded.save(str(target))
            saved_count += 1

        # webkitdirectory uploads usually add one wrapper folder.  Locate the
        # nearest nested root that follows the agreed input convention.
        candidates = [destination]
        candidates.extend(path.parent for path in destination.rglob("images") if path.is_dir())
        dataset_root = next(
            (
                candidate for candidate in sorted(candidates, key=lambda path: (len(path.parts), str(path)))
                if (candidate / "images").is_dir()
                and ((candidate / "detections").is_dir() or (candidate / "algorithm_labels").is_dir() or (candidate / "labels").is_dir())
                and ((candidate / "masks").is_dir() or (candidate / "algorithm_masks").is_dir())
            ),
            None,
        )
        if dataset_root is None:
            raise ValueError("未找到 images/、detections/（或 labels/）和 masks/ 子文件夹。")

        # Do not restore the previously active cloud batch while validating a
        # newly uploaded one.
        global _active_dataset_loaded, _active_dataset_local_import_root, _active_dataset_manifest
        if R2.enabled:
            _active_dataset_loaded = True
            _active_dataset_local_import_root = destination
            _active_dataset_manifest = None
        _configure_dataset_root(dataset_root)
        _load_image_list()
        if not _image_list_cache:
            raise ValueError("images/ 子文件夹中没有可读取的 JPG 或 PNG 图片。")
        if R2.enabled:
            remote_root = f"datasets/{import_id}"
            R2.put_tree(destination, remote_root)
            _remember_remote_dataset(destination, dataset_root, remote_root)
        return jsonify({
            "success": True,
            "dataset_name": dataset_root.name,
            "total": len(_image_list_cache),
            "imported_files": saved_count,
            "cloud_persisted": R2.enabled,
        })
    except Exception as error:
        print(f"[Import] Failed: {error}")
        return jsonify({"success": False, "error": str(error)}), 400


@app.route("/api/history/<stem>")
def api_history(stem: str):
    """Return the append-only decision trail for one sample."""
    events: List[Dict[str, Any]] = []
    if DECISION_EVENTS_PATH.exists():
        with open(DECISION_EVENTS_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                    if event.get("sample_id") == stem:
                        events.append(event)
                except json.JSONDecodeError:
                    continue
    return jsonify({"sample_id": stem, "events": events})


@app.route("/api/media/<int:index>/<kind>")
def api_media(index: int, kind: str):
    """Return one cloud object as normal image bytes, never as Base64 JSON."""
    if not (_active_dataset_manifest and R2.enabled and _active_dataset_remote_root):
        return jsonify({"error": "cloud media is unavailable"}), 404
    if kind not in {"image", "mask", "algorithm_mask"}:
        return jsonify({"error": "invalid media type"}), 404

    dataset = request.args.get("dataset") or None
    primary_class = request.args.get("primary_class") or None
    filtered = _get_filtered_images(dataset, primary_class)
    if index < 0 or index >= len(filtered):
        return jsonify({"error": "index out of range"}), 404

    image_info = filtered[index]
    stem = image_info["stem"]
    root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
    prefix = root + "/" if root else ""
    if kind == "image":
        relative = f"{prefix}images/{image_info['name']}"
        suffix = Path(image_info["name"]).suffix.lower()
        mimetype = "image/png" if suffix == ".png" else "image/jpeg"
    else:
        algorithm_relative = _direct_manifest_file(stem, ("algorithm_masks", "masks"), mask=True)
        if kind == "algorithm_mask":
            relative = algorithm_relative
        else:
            # A save always writes the expert label and mask together.  Try the
            # expert result first, then use the imported algorithm mask.
            expert_relative = f"{prefix}expert_masks/{stem}_mask.png"
            payload = _direct_object_bytes(expert_relative)
            if payload is not None:
                response = Response(payload, mimetype="image/png")
                response.headers["Cache-Control"] = "no-store"
                return response
            relative = algorithm_relative
        mimetype = "image/png"

    payload = _direct_object_bytes(relative)
    if payload is None:
        return jsonify({"error": "media not found"}), 404
    response = Response(payload, mimetype=mimetype)
    response.headers["Cache-Control"] = "public, max-age=3600" if kind == "image" else "no-store"
    return response


@app.route("/api/image/<int:index>")
def api_image(index: int):
    """Get image data by index"""
    dataset = request.args.get("dataset") or None
    primary_class = request.args.get("primary_class") or None
    
    filtered = _get_filtered_images(dataset, primary_class)
    
    if index < 0 or index >= len(filtered):
        return jsonify({"error": "index out of range"})
    
    img_info = filtered[index]
    stem = img_info['stem']
    direct_mode = bool(_active_dataset_manifest and R2.enabled and _active_dataset_remote_root)
    if not direct_mode:
        _materialize_sample(stem)
    img_path = Path(img_info['path'])
    # Stable user-facing number: position in the full, unfiltered batch.
    _load_image_list()
    global_index = next(
        (position for position, item in enumerate(_image_list_cache, start=1) if item['stem'] == stem),
        None,
    )
    
    query = request.query_string.decode("utf-8")
    media_query = f"?{query}" if query else ""
    if direct_mode:
        # The browser makes independent requests for media.  The annotation
        # response is consequently small and box editing can start at once.
        root = str(_active_dataset_manifest.get("dataset_relative_root") or "").strip("/")
        prefix = root + "/" if root else ""
        image_data_url = _direct_browser_url(f"{prefix}images/{img_info['name']}") or f"/api/media/{index}/image{media_query}"
        next_image_url = (
            _direct_browser_url(f"{prefix}images/{filtered[index + 1]['name']}")
            if index + 1 < len(filtered) else None
        )
        # Supply a short look-ahead window without downloading the files in
        # Flask.  The browser starts these direct object requests after the
        # current annotation is ready.
        prefetch_media: List[str] = []
        for lookahead in filtered[index + 1:index + 4]:
            next_stem = lookahead["stem"]
            next_image = _direct_browser_url(f"{prefix}images/{lookahead['name']}")
            next_mask_rel = _direct_manifest_file(next_stem, ("algorithm_masks", "masks"), mask=True)
            next_mask = _direct_browser_url(next_mask_rel)
            if next_image:
                prefetch_media.append(next_image)
            if next_mask:
                prefetch_media.append(next_mask)
    else:
        img = Image.open(img_path)
        img_rgb = img.convert("RGB")
        buffer = io.BytesIO()
        img_rgb.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{img_base64}"
        next_image_url = None
        prefetch_media = []
    
    # For a browser-direct import the working cache is deliberately sparse.
    # Read its two input sidecars from object storage by their manifest paths,
    # rather than letting an empty cache directory make the annotation appear
    # to be missing.  Local imports retain the normal file-based behaviour.
    has_expert_version = False
    if direct_mode:
        algorithm_label_rel = _direct_manifest_file(stem, ("algorithm_labels", "detections", "labels"))
        algorithm_label_payload = _direct_object_bytes(algorithm_label_rel)
        algorithm_bboxes = _bboxes_from_bytes(algorithm_label_payload)

        expert_label_payload = _direct_object_bytes(f"{prefix}expert_labels/{stem}.json")
        has_expert_version = expert_label_payload is not None
        bboxes = _bboxes_from_bytes(expert_label_payload) if has_expert_version else algorithm_bboxes
    else:
        bboxes = _load_bboxes_for_image(stem)
        algorithm_bboxes = _load_bboxes_for_image(stem, baseline=True)
        has_expert_version = (EXPERT_LABELS_DIR / f"{stem}.json").exists() or bool(_mask_path(EXPERT_MASKS_DIR, stem))
    metadata_path = _metadata_path(stem)
    algorithm_metadata: Dict[str, Any] = {}
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                algorithm_metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            algorithm_metadata = {}
    bbox_list = []
    for bbox_data in bboxes:
        bbox_list.append({
            "bbox": bbox_data["bbox"],
            "primary_class": bbox_data.get("primary_class"),
            "sub_type": bbox_data.get("sub_type", ""),
        })

    algorithm_bbox_list = []
    for bbox_data in algorithm_bboxes:
        algorithm_bbox_list.append({
            "id": bbox_data.get("id"),
            "bbox": bbox_data["bbox"],
            "primary_class": bbox_data.get("primary_class"),
            "sub_type": bbox_data.get("sub_type", ""),
            "score": bbox_data.get("score"),
        })
    
    # Load mask.  For cloud batches, return media URLs rather than decoding
    # and Base64-encoding a potentially large PNG inside the web worker.
    if direct_mode:
        algorithm_mask_rel = _direct_manifest_file(stem, ("algorithm_masks", "masks"), mask=True)
        algorithm_mask_data_url = _direct_browser_url(algorithm_mask_rel) if algorithm_mask_rel else None
        if algorithm_mask_data_url is None and algorithm_mask_rel:
            algorithm_mask_data_url = f"/api/media/{index}/algorithm_mask{media_query}"
        preferred_mask_rel = f"{prefix}expert_masks/{stem}_mask.png" if has_expert_version else algorithm_mask_rel
        mask_data_url = _direct_browser_url(preferred_mask_rel) if preferred_mask_rel else None
        if mask_data_url is None and algorithm_mask_rel:
            mask_data_url = f"/api/media/{index}/mask{media_query}"
        mask_rgb = None
        algorithm_mask_rgb = None
    else:
        mask_rgb = _load_mask_for_image(stem)
        algorithm_mask_rgb = _load_mask_for_image(stem, baseline=True)
        mask_data_url = None
        if mask_rgb is None and algorithm_mask_rgb is not None:
            mask_rgb = algorithm_mask_rgb
        if mask_rgb is not None:
            mask_buffer = io.BytesIO()
            Image.fromarray(mask_rgb).save(mask_buffer, format="PNG")
            mask_data_url = "data:image/png;base64," + base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
        algorithm_mask_data_url = None
        if algorithm_mask_rgb is not None:
            algorithm_buffer = io.BytesIO()
            Image.fromarray(algorithm_mask_rgb).save(algorithm_buffer, format="PNG")
            algorithm_mask_data_url = "data:image/png;base64," + base64.b64encode(algorithm_buffer.getvalue()).decode("utf-8")
    
    return jsonify({
        "index": index,
        "global_index": global_index,
        "total": len(filtered),
        "filename": img_info['name'],
        "image": image_data_url,
        "next_image": next_image_url,
        "prefetch_media": prefetch_media,
        "bboxes": bbox_list,
        "algorithm_bboxes": algorithm_bbox_list,
        "mask": mask_data_url,
        "algorithm_mask": algorithm_mask_data_url,
        "algorithm_metadata": algorithm_metadata,
        "has_expert_version": has_expert_version,
        "mask_source": "expert" if has_expert_version and mask_data_url is not None else "algorithm" if mask_data_url is not None else None,
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    """Save bbox and mask modifications"""
    try:
        data = request.json
        index = data.get("index")
        dataset = data.get("dataset")
        primary_class = data.get("primary_class")
        bboxes = data.get("bboxes", [])
        mask_base64 = data.get("mask")
        actor = str(data.get("actor") or "expert").strip()[:128] or "expert"
        reason_note = str(data.get("reason_note") or "").strip()[:2000]
        
        # Get current image info
        filtered = _get_filtered_images(dataset, primary_class)
        if index < 0 or index >= len(filtered):
            return jsonify({"success": False, "error": "index out of range"})
        
        img_info = filtered[index]
        stem = img_info['stem']
        _materialize_sample(stem)
        before_bboxes = _load_bboxes_for_image(stem)
        
        # Save bbox modifications
        label_path = EXPERT_LABELS_DIR / f"{stem}.json"
        label_data = {
            "image_path": img_info['name'],
            "bboxes": []
        }
        
        for i, bbox_data in enumerate(bboxes):
            bbox_xywh = bbox_data.get("bbox", [])
            if len(bbox_xywh) != 4:
                raise ValueError(f"bbox {i} must contain [x, y, width, height]")

            primary = bbox_data.get("primary_class")
            subtype = bbox_data.get("sub_type")
            if primary not in PRIMARY_TO_SUBTYPES:
                raise ValueError(f"bbox {i} has an invalid primary class: {primary!r}")
            if subtype not in PRIMARY_TO_SUBTYPES[primary]:
                raise ValueError(
                    f"bbox {i} subtype {subtype!r} is not valid for primary class {primary!r}"
                )
            
            taxonomy = {
                "primary_class": primary,
                "sub_type": subtype,
            }
            
            instance_id = f"{stem}_{i}"
            
            label_data["bboxes"].append({
                "instance_id": instance_id,
                "taxonomy": taxonomy,
                "bbox": bbox_xywh  # [x, y, w, h] format
            })
        
        # Save label file
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)
        
        # Save mask modifications
        if mask_base64:
            try:
                mask_path = EXPERT_MASKS_DIR / f"{stem}_mask.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                
                if "," in mask_base64:
                    mask_data = base64.b64decode(mask_base64.split(",", 1)[1])
                else:
                    mask_data = base64.b64decode(mask_base64)
                
                mask_img = Image.open(io.BytesIO(mask_data))
                mask_img.save(mask_path, "PNG")
                print(f"[Save] Mask saved: {mask_path} (size: {mask_img.size})")
            except Exception as e:
                print(f"[Save] Failed to save mask: {e}")
                import traceback
                traceback.print_exc()
        
        # Reload image list cache
        global _image_list_loaded
        _image_list_loaded = False
        _load_image_list()

        _sync_active_dataset_file(label_path)
        if mask_base64:
            _sync_active_dataset_file(EXPERT_MASKS_DIR / f"{stem}_mask.png")
        
        revision = _append_decision_event(
            stem=stem,
            image_path=Path(img_info["path"]),
            before_bboxes=before_bboxes,
            after_bboxes=label_data["bboxes"],
            actor=actor,
            reason_note=reason_note,
            mask_changed=bool(mask_base64),
            algorithm_baseline={
                "detection": _file_descriptor(_baseline_label_path(stem)),
                "mask": _file_descriptor(_baseline_mask_path(stem)),
                "metadata": _file_descriptor(_metadata_path(stem)),
            },
        )
        return jsonify({"success": True, "revision": revision})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/refine_mask", methods=["POST"])
def api_refine_mask():
    """Refine mask using SAM (point operations)"""
    if not SAM_AVAILABLE or sam_service is None:
        return jsonify({"success": False, "error": "SAM service not available"})
    
    try:
        data = request.json
        image_base64 = data.get("image_data")
        mask_base64 = data.get("mask_data")
        points = data.get("points", [])
        labels = data.get("labels", [])
        bboxes = data.get("bboxes", [])
        
        if not image_base64:
            return jsonify({"success": False, "error": "image_data is required"})
        
        if len(points) == 0:
            return jsonify({"success": False, "error": "points are required"})
        
        # Decode image
        image_data = base64.b64decode(image_base64.split(",")[1] if "," in image_base64 else image_base64)
        image_np = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_np is None:
            return jsonify({"success": False, "error": "Failed to decode image"})
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Decode current mask (if exists)
        current_mask = None
        if mask_base64:
            mask_data = base64.b64decode(mask_base64.split(",")[1] if "," in mask_base64 else mask_base64)
            mask_np = np.frombuffer(mask_data, np.uint8)
            mask_img = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                if len(mask_img.shape) == 3:
                    if mask_img.shape[2] == 4:
                        current_mask = mask_img[:, :, 3]
                    else:
                        current_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                else:
                    current_mask = mask_img
                
                if current_mask.shape[:2] != image_np.shape[:2]:
                    current_mask = cv2.resize(current_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            current_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        # Ensure SAM is initialized
        if not sam_service.initialized:
            sam_service.initialize()
        
        # Call SAM refine
        refined_mask = sam_service.refine_mixed(
            image_np=image_np,
            current_mask=current_mask,
            points=points,
            labels=labels,
            bboxes=bboxes if bboxes else None,
            brush_mask_b64=None,
            operation='point'
        )
        
        # Convert mask to base64
        mask_pil = Image.fromarray(refined_mask)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format="PNG")
        mask_base64_result = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
        mask_data_url = f"data:image/png;base64,{mask_base64_result}"
        
        return jsonify({
            "success": True,
            "mask": mask_data_url
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Run detection model and return detection results"""
    if not DETECTION_AVAILABLE or detection_service is None:
        return jsonify({"success": False, "error": "Detection service not available"})
    
    try:
        data = request.json
        image_base64 = data.get("image_data")
        model = data.get("model", "ensemble")
        filename = data.get("filename", "unknown.jpg")
        
        if not image_base64:
            return jsonify({"success": False, "error": "image_data is required"})
        
        # Decode image
        image_data = base64.b64decode(image_base64.split(",")[1] if "," in image_base64 else image_base64)
        image_np = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_np is None:
            return jsonify({"success": False, "error": "Failed to decode image"})
        
        # Set detection model
        detection_service.model = model
        
        # Run detection (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            detection_service.detect(image_np, text_prompt="Detect defects in this image.", filename=filename)
        )
        loop.close()
        
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})
        
        # Convert detection results format
        annotations = result.get("annotations_in_crop", [])
        bboxes = []
        h, w = image_np.shape[:2]
        
        for ann in annotations:
            # Detection results use normalized coordinates, convert to pixel coordinates
            x_center_norm = ann.get("x_center_norm", 0)
            y_center_norm = ann.get("y_center_norm", 0)
            width_norm = ann.get("width_norm", 0)
            height_norm = ann.get("height_norm", 0)
            
            # Convert to pixel coordinates [x, y, w, h]
            x_center = x_center_norm * w
            y_center = y_center_norm * h
            bbox_w = width_norm * w
            bbox_h = height_norm * h
            x = x_center - bbox_w / 2
            y = y_center - bbox_h / 2
            
            # Get class name (sub_type)
            class_name = ann.get("class_name", "unknown")
            
            # Map to primary_class
            primary_class = SUBTYPE_TO_PRIMARY_CLASS.get(class_name, "Crack")
            
            bboxes.append({
                "bbox": [x, y, bbox_w, bbox_h],  # [x, y, w, h] format
                "primary_class": primary_class,
                "sub_type": class_name  # Keep original class name
            })
        
        return jsonify({
            "success": True,
            "bboxes": bboxes
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/add_candidate", methods=["POST"])
def api_add_candidate():
    """Add current image to candidates list"""
    try:
        data = request.json
        index = data.get("index")
        dataset = data.get("dataset")
        primary_class = data.get("primary_class")
        filename = data.get("filename")
        
        if not primary_class:
            return jsonify({"success": False, "error": "primary_class is required"})
        
        # Read or create candidates CSV
        if CANDIDATES_CSV.exists():
            df_candidates = pd.read_csv(CANDIDATES_CSV)
        else:
            df_candidates = pd.DataFrame(columns=["primary_class", "dataset", "index", "filename", "added_at"])
        
        # Check if already exists (avoid duplicates)
        if len(df_candidates[(df_candidates["primary_class"] == primary_class) & 
                            (df_candidates["filename"] == filename)]) > 0:
            return jsonify({
                "success": False,
                "error": "This image already exists in the candidates list"
            })
        
        # Add new record
        from datetime import datetime
        new_row = {
            "primary_class": primary_class,
            "dataset": dataset or "",
            "index": index,
            "filename": filename,
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df_candidates = pd.concat([df_candidates, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save CSV
        CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_candidates.to_csv(CANDIDATES_CSV, index=False, encoding='utf-8')
        
        new_count = len(df_candidates[df_candidates["primary_class"] == primary_class])
        return jsonify({
            "success": True,
            "message": f"Added to candidates ({new_count} total)"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Load image list cache
    _load_image_list()
    
    print("=" * 80)
    port = int(os.environ.get("REANNOTATION_PORT", "5010"))
    print("DefectBench Re-annotation Tool")
    print("=" * 80)
    print(f"Images: {len(_image_list_cache)} files")
    print(f"Access URL: http://localhost:{port}")
    print("=" * 80)
    app.run(host="0.0.0.0", port=port, debug=False)
