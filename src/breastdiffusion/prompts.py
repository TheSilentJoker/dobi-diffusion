from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


ATTRIBUTE_COLUMNS = [
    "cyst",
    "lump",
    "location",
    "size",
    "form",
    "direction",
    "border",
    "edge",
    "in_echo",
    "rear_echo",
    "calcifications",
    "blood",
    "bi_rads",
]

NEGATIVE_PROMPT = (
    "color photo, RGB medical photo, X-ray, CT, MRI, ultrasound screenshot, "
    "text, arrows, labels, watermark, ruler marks, cropped breast, blank image, "
    "overexposed image, severe noise, checkerboard artifacts, duplicated anatomy"
)


_PREFIX_PATTERNS = [
    r"^时钟法[:：]",
    r"^部位[:：]",
    r"^最大径[:：]",
    r"^形态[:：]",
    r"^方向[:：]",
    r"^边界[:：]",
    r"^边缘[:：]",
    r"^内部回声[:：]",
    r"^后方回声[:：]",
    r"^钙化灶[:：]",
    r"^血流[:：]",
]

_PHRASE_MAP = {
    "有": "lesion present",
    "无": "absent",
    "单发": "single lesion",
    "多发": "multiple lesions",
    "椭圆形": "oval shape",
    "圆形": "round shape",
    "不规则": "irregular shape",
    "纵横比>=1": "aspect ratio greater than or equal to 1",
    "纵横比<1": "aspect ratio less than 1",
    "光整": "smooth boundary",
    "成角": "angular boundary",
    "清晰": "clear margin",
    "不清晰": "unclear margin",
    "低": "low internal echo",
    "等": "iso internal echo",
    "不均匀": "heterogeneous internal echo",
    "无变化": "no posterior echo change",
    "细小": "microcalcifications",
    "少许": "sparse blood flow",
    "丰富": "rich blood flow",
    "定期检查": "routine follow-up",
}


@dataclass(frozen=True)
class PromptRecord:
    filename: str
    prompt: str
    negative_prompt: str


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "***"}


def _clean_token(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    for pattern in _PREFIX_PATTERNS:
        text = re.sub(pattern, "", text)
    text = text.replace("；", ";").replace("，", ",").replace("：", ":")
    text = text.replace("级", "")
    text = text.strip(" ;,")
    for cn, en in sorted(_PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(cn, en)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;,")


def _attribute_phrase(column: str, value: object) -> str:
    token = _clean_token(value)
    if not token:
        return ""
    if column == "cyst":
        return f"cyst status {token}"
    if column == "lump":
        return f"lesion multiplicity {token}"
    if column == "location":
        return f"lesion location {token}"
    if column == "size":
        return f"lesion maximum diameter {token} millimeters"
    if column == "form":
        return f"lesion morphology {token}"
    if column == "direction":
        return f"lesion orientation {token}"
    if column == "border":
        return f"lesion boundary {token}"
    if column == "edge":
        return f"lesion margin {token}"
    if column == "in_echo":
        return f"internal echo pattern {token}"
    if column == "rear_echo":
        return f"posterior echo pattern {token}"
    if column == "calcifications":
        return f"calcification status {token}"
    if column == "blood":
        return f"blood flow status {token}"
    return token


def _metric_phrase(row: pd.Series, column: str, display_name: str) -> str:
    value = row.get(column)
    if _is_missing(value):
        return ""
    try:
        return f"{display_name} {float(value):.2f} percent"
    except (TypeError, ValueError):
        return ""


def build_prompt(row: pd.Series) -> str:
    """Build a Stable-Diffusion-friendly English prompt from one label row."""
    label = int(row.get("label", 0))
    side = "left breast" if str(row.get("side", "")).upper() == "L" else "right breast"
    tumor_state = "tumor-positive breast NIR image" if label == 1 else "tumor-negative breast NIR image"
    birads = _clean_token(row.get("bi_rads"))

    parts: list[str] = [
        "near-infrared dynamic optical breast imaging",
        "DOBI NIR grayscale medical image",
        "low-resolution 128 by 102 pixels",
        side,
        tumor_state,
    ]

    if birads:
        parts.append(f"BI-RADS {birads}")

    for column in ATTRIBUTE_COLUMNS:
        if column == "bi_rads":
            continue
        token = _attribute_phrase(column, row.get(column))
        if token:
            parts.append(token)

    metric_parts = [
        _metric_phrase(row, "Breast_Ratio_Global(%)", "global breast ratio"),
        _metric_phrase(row, "Illum_Coverage_Local(%)", "local illumination coverage"),
        _metric_phrase(row, "Leak_Ratio_Local(%)", "local leakage ratio"),
    ]
    parts.extend(part for part in metric_parts if part)

    if label == 1:
        parts.append("subtle asymmetric vascular and absorption pattern")
    else:
        parts.append("regular symmetric illumination pattern without tumor signature")

    return ", ".join(dict.fromkeys(parts))


def build_records(rows: Iterable[pd.Series]) -> list[PromptRecord]:
    return [
        PromptRecord(
            filename=str(row["filename"]),
            prompt=build_prompt(row),
            negative_prompt=NEGATIVE_PROMPT,
        )
        for _, row in rows
    ]
