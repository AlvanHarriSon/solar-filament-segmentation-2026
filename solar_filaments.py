"""Run the frozen SpatialQuality-Prune submission on full-disk GONG JPEGs."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from ultralytics import YOLO

from filament_models import (
    FEATURES, QualityHead, Refiner, add_instance, attributes, extra_boxes,
    mask_features, predict_scores, quality_scores, refine_prob, subset_labels,
)

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
COLUMNS = ["filament_id", "segmentation_rle"]
SHAPE = (2048, 2048)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FilamentPredictor:
    """Load the detector, seed-conditioned ROI refiner and four spatial quality heads."""

    def __init__(self, weights_dir=None, device="cuda"):
        directory = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or pass device='cpu'.")
        self.config = json.loads((directory / "inference.json").read_text(encoding="utf8"))
        for filename, expected in self.config["weight_sha256"].items():
            if file_sha256(directory / filename) != expected:
                raise ValueError(f"Weight checksum mismatch: {filename}")
        if self.config["features"] != FEATURES:
            raise ValueError("The ridge model's input columns disagree with the implementation.")
        self.detector = YOLO(str(directory / "detector.pt"), task="segment")
        self.refiner = Refiner().to(self.device).eval()
        self.refiner.load_state_dict(torch.load(directory / "refiner.pt", map_location="cpu", weights_only=True))
        self.heads = []
        for fold in range(4):
            head = QualityHead().to(self.device).eval()
            head.load_state_dict(torch.load(directory / f"fold_{fold}.pt", map_location="cpu", weights_only=True))
            self.heads.append(head)
        cv2.setNumThreads(3)

    def _detect(self, path, threshold):
        result = self.detector.predict(
            str(path), conf=threshold, iou=.60, imgsz=1536, retina_masks=True,
            agnostic_nms=True, device=str(self.device), verbose=False,
        )[0]
        if tuple(result.orig_shape) != SHAPE:
            raise ValueError(f"Expected 2048 by 2048 native coordinates for {path.name}")
        return [{"box":box, "score":score}
                for box,score in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),strict=True)]

    @torch.inference_mode()
    def predict_image(self, image_path):
        """Return a native disjoint label map (0 = background) and stage counts."""
        path = Path(image_path)
        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"))
        if gray.shape != SHAPE:
            raise ValueError(f"Expected a 2048 by 2048 image, got {gray.shape}")
        high = self._detect(path, .30)
        low = [d for d in self._detect(path, .10) if d["score"] < .30]
        labels = np.zeros(SHAPE, np.int32)
        records, tensors = [], []
        for primary, proposals in (
            (True, sorted(high, key=lambda det: -det["score"])),
            (False, extra_boxes(high, low)),
        ):
            areas = np.bincount(labels.ravel()).tolist()
            for det in proposals:
                probability, side = refine_prob(gray, det["box"], self.refiner, device=self.device)
                mask = probability > .55
                area = np.count_nonzero(mask)
                if area < 96 or area < .005 * max(96., side) ** 2:
                    continue
                if primary:
                    available = mask & (labels == 0)
                    if np.count_nonzero(available) < 96:
                        continue
                    labels[available] = len(records) + 1
                elif not add_instance(labels, areas, mask):
                    continue
                record, tensor = mask_features(gray, labels, len(records)+1, det, probability)
                records.append(record)
                tensors.append(tensor)
            if primary:
                primary_count = len(records)
        spatial = np.asarray(tensors, dtype=np.uint8).reshape(-1, 5, 32, 32)
        scalars = np.asarray([attributes(r) for r in records], dtype=np.float32).reshape(-1, len(FEATURES))
        keep = np.ones(len(records), bool)
        keep[primary_count:] = quality_scores(records[primary_count:], self.config["ridge"]) >= self.config["ridge_cutoff"]
        after_ridge = int(keep.sum())
        scores = predict_scores(self.heads, spatial, scalars)
        keep &= scores >= self.config["spatial_cutoff"]
        return subset_labels(labels, keep), dict(
            high_boxes=len(high), low_boxes=len(low), primary_instances=primary_count,
            additional_instances=len(records)-primary_count, after_ridge=after_ridge,
            final_instances=int(keep.sum()),
        )


def encode(mask):
    return coco.encode(np.asfortranarray(mask, dtype=np.uint8))["counts"].decode("ascii")


def decode(counts):
    return coco.decode({"size":list(SHAPE), "counts":counts.encode("ascii")})


def validate_csv(path, manifest):
    """Verify sequential IDs, manifest counts, every RLE and native pixel ownership."""
    rows = 0
    with Path(path).open(newline="", encoding="utf8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected CSV columns")
        for image in manifest:
            occupied = np.zeros(SHAPE, bool)
            for idx in range(1, image["instances"]+1):
                row = next(reader, None)
                if row is None or row["filament_id"] != f"{image['image_id']}_{idx}":
                    raise ValueError("Instance ID or row count disagrees with the manifest")
                counts = row["segmentation_rle"]
                mask = decode(counts)
                if mask.shape != SHAPE or not mask.any() or encode(mask) != counts:
                    raise ValueError("Decoded COCO RLE failed round-trip verification")
                if np.any(occupied & mask):
                    raise ValueError("Two instances claim the same pixel")
                occupied |= mask.astype(bool)
                rows += 1
        if next(reader, None) is not None:
            raise ValueError("Extra CSV rows after processing the manifest")
    return rows


def run_submission(image_dir, output_csv, *, weights_dir=None, device="cuda", expected_images=180, predictor=None):
    """Infer from all official JPEGs in image_dir; atomically publish a verified CSV."""
    folder = Path(image_dir)
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    ids = [p.stem for p in images]
    if len(images) != expected_images or len(set(ids)) != len(ids):
        raise ValueError(f"Expected {expected_images} unique physical JPEGs, found {len(images)}")
    if not all(re.fullmatch(r"\d{14}[A-Za-z]{2}", key) for key in ids):
        raise ValueError("Expected official GONG image IDs as JPEG filenames")
    predictor = predictor if predictor is not None else FilamentPredictor(weights_dir, device)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        temporary = Path(scratch) / "predictions.csv"
        with temporary.open("x", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\r\n")
            writer.writeheader()
            for i, path in enumerate(images, 1):
                labels, info = predictor.predict_image(path)
                n = info["final_instances"]
                if labels.shape != SHAPE or labels.dtype != np.int32 or int(labels.max()) != n:
                    raise ValueError("Invalid native label map")
                for idx in range(1, n+1):
                    mask = labels == idx
                    counts = encode(mask)
                    if not mask.any() or not np.array_equal(mask, decode(counts)):
                        raise ValueError("Encoded mask differs from the prediction")
                    writer.writerow({"filament_id":f"{path.stem}_{idx}", "segmentation_rle":counts})
                manifest.append(dict(image_id=path.stem, file_name=path.name, instances=n, shape=list(SHAPE)))
                if i % 30 == 0 or i == len(images):
                    print(f"Processed {i}/{len(images)} images", flush=True)
        rows = validate_csv(temporary, manifest)
        digest = file_sha256(temporary)
        os.replace(temporary, destination)
    zeros = [r["image_id"] for r in manifest if r["instances"] == 0]
    return dict(csv_path=str(destination.resolve()), csv_sha256=digest, rows=rows, unique_ids=rows,
                expected_images=expected_images, processed_images=len(images), columns=COLUMNS,
                images_represented_in_csv=len(images)-len(zeros), zero_prediction_images=zeros,
                all_rles_decoded=True, all_rles_roundtrip=True, disjoint_instances=True, manifest=manifest)
