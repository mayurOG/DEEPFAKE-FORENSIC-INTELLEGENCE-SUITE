import io
import os
from PIL import Image, ExifTags

# Editing tools: worth noting, but editing is not manipulation by itself.
EDITING_SOFTWARE = ["photoshop", "gimp", "canva", "lightroom", "snapseed"]
# Names of AI generators / face-swap tools. Finding one in the file's own
# metadata is explicit evidence, unlike a *missing* EXIF block.
AI_GENERATOR_MARKERS = [
    "stable diffusion", "midjourney", "dall-e", "dalle", "novelai",
    "comfyui", "automatic1111", "sdxl", "firefly", "faceapp", "deepface",
    "faceswap", "reface", "deepfacelab", "roop", "insightface",
]
# Text-chunk keys written by popular generators (PNG tEXt / iTXt).
AI_TEXT_KEYS = ["parameters", "prompt", "workflow", "invokeai_metadata"]


def _find_ai_signature(img, software):
    """Return the first AI-generator marker found in the file metadata."""
    candidates = []
    if software:
        candidates.append(str(software))
    for key, value in (img.info or {}).items():
        k = str(key).lower()
        if k in AI_TEXT_KEYS:
            # A1111-style "parameters" blocks always contain Steps/Sampler.
            v = str(value).lower()
            if k == "parameters" and "steps:" in v and "sampler:" in v:
                return "stable-diffusion style generation parameters"
            if k in ("prompt", "workflow"):
                return f"generator '{k}' metadata chunk"
        if isinstance(value, (str, bytes)):
            v = value.decode("utf-8", "ignore") if isinstance(value, bytes) else value
            candidates.append(v)
    blob = " ".join(candidates).lower()
    for marker in AI_GENERATOR_MARKERS:
        if marker in blob:
            return marker
    return None


def analyze_metadata(image_input) -> dict:
    """
    Extracts EXIF / header metadata. Returns a dict that includes
    ``ai_generator_signature`` when the file's own metadata names an AI
    generator (explicit evidence). A missing EXIF block is reported as a
    neutral note: social media and messengers strip EXIF from genuine photos.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
        file_size = os.path.getsize(image_input) if os.path.exists(image_input) else None
    elif hasattr(image_input, "read"):
        data = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        file_size = len(data)
        img = Image.open(io.BytesIO(data))
    elif isinstance(image_input, Image.Image):
        img = image_input
        file_size = None
    else:
        return {"error": "Unsupported image format"}

    width, height = img.size
    exif_data = {}
    software_found = camera_make = camera_model = date_taken = None
    warnings = []
    notes = []

    try:
        exif = img.getexif()
        merged = dict(exif)
        try:
            merged.update(exif.get_ifd(0x8769))  # Exif sub-IFD (dates, lens)
        except Exception:
            pass
        for tag_id, value in merged.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            sval = str(value)
            if len(sval) > 300:
                sval = sval[:300] + "..."
            exif_data[str(tag_name)] = sval
            if tag_name == "Software":
                software_found = str(value)
            elif tag_name == "Make":
                camera_make = str(value)
            elif tag_name == "Model":
                camera_model = str(value)
            elif tag_name in ("DateTimeOriginal", "DateTime") and not date_taken:
                date_taken = str(value)
    except Exception:
        pass

    if not software_found and "software" in (img.info or {}):
        software_found = str(img.info["software"])

    ai_signature = _find_ai_signature(img, software_found)
    if ai_signature:
        warnings.append(f"AI generator signature found in metadata: '{ai_signature}'")

    if software_found:
        sw = software_found.lower()
        if any(e in sw for e in EDITING_SOFTWARE):
            warnings.append(f"Edited with software: '{software_found}'")

    if not exif_data:
        notes.append(
            "No EXIF metadata present (common for web / messenger re-shares; "
            "not evidence of manipulation by itself)."
        )

    dpi = img.info.get("dpi")
    return {
        "dimensions": f"{width} x {height}",
        "width": width,
        "height": height,
        "format": img.format,
        "mode": img.mode,
        "file_size_bytes": file_size,
        "file_size_kb": f"{file_size / 1024:.1f} KB" if file_size else "N/A",
        "camera_make": camera_make or "Unknown",
        "camera_model": camera_model or "Unknown",
        "software": software_found or "None Detected",
        "date_taken": date_taken or "Unknown",
        "dpi": str(dpi) if dpi else "Unknown",
        "has_exif": len(exif_data) > 0,
        "ai_generator_signature": ai_signature,
        "warnings": warnings,
        "notes": notes,
        "exif_data": exif_data,
    }
