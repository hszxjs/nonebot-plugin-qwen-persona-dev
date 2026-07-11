from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any


DEFAULT_VISION_MODEL = "Salesforce/blip-image-captioning-base"
IMAGE_CAPTION_TASKS = ("image-to-text", "image-text-to-text")
IMAGE_CAPTION_PROMPT = "describe this image"
CHINESE_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
OCR_MIN_CONFIDENCE = 0.45
CLIP_CANDIDATE_LABELS: tuple[tuple[str, str], ...] = (
    ("ordinary photo", "图片"),
    ("sticker meme", "表情包"),
    ("chat screenshot", "聊天截图"),
    ("document or text screenshot", "文字截图"),
    ("anime character", "动漫角色"),
    ("game screenshot", "游戏画面"),
    ("food photo", "食物"),
    ("landscape or city scene", "风景场景"),
    ("product or object photo", "物品"),
)
_PIPELINE: Any | None = None
_PIPELINE_MODEL = ""
_PIPELINE_TASK = ""
_OCR_ENGINE: Any | None = None
_OCR_ENGINE_UNAVAILABLE = False


def caption_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
    offline: bool = False,
) -> dict[str, Any]:
    media_type = str(payload.get("type") or "media")
    if offline:
        return {"caption": _fallback_caption(media_type), "source": "fallback"}

    try:
        image = _load_image(payload)
        ocr_caption = _extract_chinese_text(image)
        if ocr_caption:
            return {
                "caption": _safe_caption(ocr_caption),
                "source": "ocr",
            }
        caption = _caption_image_with_transformers(image, model_name=model_name)
        if caption:
            return {
                "caption": _safe_caption(caption),
                "source": "clip" if _is_clip_model(model_name) else "transformers",
            }
    except Exception as exc:
        return {
            "caption": _fallback_caption(media_type),
            "source": "fallback",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"caption": _fallback_caption(media_type), "source": "fallback"}


def _caption_with_transformers(payload: dict[str, Any], *, model_name: str) -> str:
    image = _load_image(payload)
    return _caption_image_with_transformers(image, model_name=model_name)


def _caption_image_with_transformers(image: Any, *, model_name: str) -> str:
    model_name = model_name or DEFAULT_VISION_MODEL
    if _is_clip_model(model_name):
        return _caption_image_with_clip(image, model_name=model_name)
    last_task_error: KeyError | None = None
    for task in IMAGE_CAPTION_TASKS:
        try:
            pipeline = _get_pipeline(task, model_name)
        except KeyError as exc:
            last_task_error = exc
            continue
        result = _run_caption_pipeline(task, pipeline, image)
        return _caption_result_text(result)
    if last_task_error is not None:
        raise last_task_error
    return ""


def _run_caption_pipeline(task: str, pipeline: Any, image: Any) -> Any:
    if task == "image-text-to-text":
        return pipeline(images=image, text=IMAGE_CAPTION_PROMPT)
    return pipeline(image)


def _caption_result_text(result: Any) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("generated_text") or first.get("caption") or "")
        return str(first)
    if isinstance(result, dict):
        return str(result.get("generated_text") or result.get("caption") or "")
    return str(result or "")


def _caption_with_clip(payload: dict[str, Any], *, model_name: str) -> str:
    image = _load_image(payload)
    return _caption_image_with_clip(image, model_name=model_name)


def _caption_image_with_clip(image: Any, *, model_name: str) -> str:
    pipeline = _get_pipeline("zero-shot-image-classification", model_name)
    label_map = dict(CLIP_CANDIDATE_LABELS)
    candidate_labels = [label for label, _summary in CLIP_CANDIDATE_LABELS]
    result = pipeline(image, candidate_labels=candidate_labels)
    if not isinstance(result, list) or not result:
        return ""
    best = max(
        (item for item in result if isinstance(item, dict)),
        key=lambda item: float(item.get("score") or 0),
        default={},
    )
    label = str(best.get("label") or "").strip()
    summary = label_map.get(label, label)
    score = best.get("score")
    if isinstance(score, int | float):
        return f"{summary}，置信度 {float(score):.2f}"
    return summary


def _get_pipeline(task: str, model_name: str) -> Any:
    global _PIPELINE, _PIPELINE_MODEL, _PIPELINE_TASK
    if _PIPELINE is None or _PIPELINE_MODEL != model_name or _PIPELINE_TASK != task:
        _PIPELINE = _load_pipeline(task, model_name)
        _PIPELINE_MODEL = model_name
        _PIPELINE_TASK = task
    return _PIPELINE


def _load_pipeline(task: str, model_name: str) -> Any:
    from transformers import pipeline

    return pipeline(task, model=model_name)


def _is_clip_model(model_name: str) -> bool:
    return "clip" in str(model_name or DEFAULT_VISION_MODEL).lower()


def _extract_chinese_text(image: Any) -> str:
    engine = _get_ocr_engine()
    if engine is None:
        return ""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        result = engine(buffer.getvalue())
    except Exception:
        return ""

    lines: list[str] = []
    seen: set[str] = set()
    for text, confidence in _iter_ocr_text(result):
        if confidence is not None and confidence < OCR_MIN_CONFIDENCE:
            continue
        cleaned = _normalize_ocr_line(text)
        if not cleaned or not CHINESE_TEXT_RE.search(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        lines.append(cleaned)
    return "，".join(lines)


def _get_ocr_engine() -> Any | None:
    global _OCR_ENGINE, _OCR_ENGINE_UNAVAILABLE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_ENGINE_UNAVAILABLE:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
    except Exception:
        _OCR_ENGINE_UNAVAILABLE = True
        return None
    return _OCR_ENGINE


def _iter_ocr_text(result: Any) -> list[tuple[str, float | None]]:
    payload = result[0] if isinstance(result, tuple) and result else result
    if payload is None:
        return []
    items: list[tuple[str, float | None]] = []
    for item in payload if isinstance(payload, list | tuple) else []:
        if isinstance(item, dict):
            text = item.get("text") or item.get("txt") or ""
            confidence = item.get("score") or item.get("confidence")
        elif isinstance(item, list | tuple) and len(item) >= 2:
            text = item[1]
            confidence = item[2] if len(item) >= 3 else None
        else:
            continue
        try:
            score = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            score = None
        items.append((str(text or ""), score))
    return items


def _normalize_ocr_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(
        r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]) "
        r"(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])",
        "",
        cleaned,
    )
    return re.sub(r"\s*([，。！？；：、])\s*", r"\1", cleaned)


def _load_image(payload: dict[str, Any]) -> Any:
    from PIL import Image

    data = payload.get("data") if isinstance(payload, dict) else {}
    data = data if isinstance(data, dict) else {}
    url = str(payload.get("url") or data.get("url") or "")
    if not url or not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise ValueError("only http image URLs are supported by builtin vision service")
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = response.read(16 * 1024 * 1024)
    return Image.open(BytesIO(raw)).convert("RGB")


def _fallback_caption(media_type: str) -> str:
    if media_type == "image":
        return "图片"
    if media_type in {"face", "mface", "emoji", "sticker"}:
        return "表情"
    return "媒体"


def _safe_caption(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"(?:https?://|file://|base64://)\S+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:120] or "媒体"


class VisionRequestHandler(BaseHTTPRequestHandler):
    model_name = DEFAULT_VISION_MODEL

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"ok": True, "model": self.model_name})
            return
        self._send_text("not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/caption":
            self._send_text("not found", status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > 2_000_000:
                raise ValueError("request body too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            self._send_json(caption_payload(payload, model_name=self.model_name))
        except Exception as exc:
            self._send_json({"caption": "媒体", "source": "fallback", "error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: object) -> None:
        print("vision service: " + fmt % args, file=sys.stderr, flush=True)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL)
    args = parser.parse_args(argv)
    VisionRequestHandler.model_name = args.model
    server = ThreadingHTTPServer((args.host, args.port), VisionRequestHandler)
    print(f"vision service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
