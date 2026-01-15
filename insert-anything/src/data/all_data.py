import os, csv
import cv2
import numpy as np
from .base import BaseDataset

def _resolve(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    p = p[2:] if p.startswith("./") else p
    return os.path.join(base_dir, p)

def _letterbox_rgb(img, out_hw=(768, 768), pad_value=0):
    """RGB 이미지: 비율 유지 resize + padding -> (H,W,3)"""
    oh, ow = out_hw
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image shape: {img.shape}")

    scale = min(ow / w, oh / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    # 안전장치
    nw = max(1, min(ow, nw))
    nh = max(1, min(oh, nh))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((oh, ow, 3), pad_value, dtype=resized.dtype)
    top = (oh - nh) // 2
    left = (ow - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def _letterbox_mask(mask, out_hw=(768, 768), pad_value=0):
    """마스크(0/1 또는 0~255): 비율 유지 resize(nearest) + padding(0) -> (H,W)"""
    oh, ow = out_hw
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid mask shape: {mask.shape}")

    scale = min(ow / w, oh / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    nw = max(1, min(ow, nw))
    nh = max(1, min(oh, nh))

    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((oh, ow), pad_value, dtype=resized.dtype)
    top = (oh - nh) // 2
    left = (ow - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

class AllDatasetCSV(BaseDataset):
    def __init__(self, csv_path=None, base_dir="", image_dir=None, data_type="noperson"):
        super().__init__()
        self.size = (768, 768)  # (H,W)
        self.data_type = data_type

        # train.py가 image_dir=...로 넘기는 구버전 호환
        if image_dir is not None and (base_dir == "" or base_dir is None):
            base_dir = image_dir
        self.base_dir = base_dir

        if csv_path is None:
            raise ValueError("csv_path is required for AllDatasetCSV")

        self.rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            # BOM/공백 강제 제거(재발 방지)
            r.fieldnames = [h.lstrip("\ufeff").strip() for h in (r.fieldnames or [])]

            need = {"ref_image", "ref_masked", "tar_image", "tar_masked"}
            if not need.issubset(set(r.fieldnames)):
                raise ValueError(f"CSV must contain columns {sorted(list(need))}, got {r.fieldnames}")

            for row in r:
                row = {k.lstrip("\ufeff").strip(): v for k, v in row.items()}
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def get_sample(self, idx):
        row = self.rows[idx]

        ref_image_path = _resolve(self.base_dir, row["ref_image"])
        tar_image_path = _resolve(self.base_dir, row["tar_image"])
        ref_mask_path  = _resolve(self.base_dir, row["ref_masked"])
        tar_mask_path  = _resolve(self.base_dir, row["tar_masked"])

        # Read Image (BGR -> RGB)
        ref_bgr = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
        if ref_bgr is None:
            raise FileNotFoundError(f"ref_image not found: {ref_image_path}")
        ref_image = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

        tar_bgr = cv2.imread(tar_image_path, cv2.IMREAD_COLOR)
        if tar_bgr is None:
            raise FileNotFoundError(f"tar_image not found: {tar_image_path}")
        tar_image = cv2.cvtColor(tar_bgr, cv2.COLOR_BGR2RGB)

        # Read Mask (grayscale)
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_mask is None:
            raise FileNotFoundError(f"ref_mask not found: {ref_mask_path}")
        tar_mask = cv2.imread(tar_mask_path, cv2.IMREAD_GRAYSCALE)
        if tar_mask is None:
            raise FileNotFoundError(f"tar_mask not found: {tar_mask_path}")

        # Binarize BEFORE resize (둘 중 어느 쪽이든 상관은 없지만 일관성을 위해)
        ref_mask = (ref_mask > 128).astype(np.uint8)
        tar_mask = (tar_mask > 128).astype(np.uint8)

        # ✅ Resize + Padding (black)
        out_hw = self.size  # (H,W)
        ref_image = _letterbox_rgb(ref_image, out_hw=out_hw, pad_value=0)
        tar_image = _letterbox_rgb(tar_image, out_hw=out_hw, pad_value=0)
        ref_mask  = _letterbox_mask(ref_mask,  out_hw=out_hw, pad_value=0)
        tar_mask  = _letterbox_mask(tar_mask,  out_hw=out_hw, pad_value=0)

        item = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        return item
