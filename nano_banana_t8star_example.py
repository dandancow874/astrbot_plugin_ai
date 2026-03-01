import base64
import json
import os
from pathlib import Path

import requests


BASE_URL = "https://ai.t8star.cn"
API_KEY = os.getenv("T8STAR_API_KEY", "YOUR_API_KEY")
MODEL = "gemini-3-pro-image-preview"


def normalize_gemini_models_base(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1beta") or url.endswith("/v1"):
        return f"{url}/models"
    if "/models" in url:
        return url
    if "/v1beta/" not in url and "/v1/" not in url and url.count("/") <= 2:
        return f"{url}/v1beta/models"
    return url


def load_image_b64(path: str) -> tuple[str, str]:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    return mime, b64


def build_payload(prompt: str, image_paths: list[str]) -> dict:
    parts: list[dict] = [{"text": prompt}]
    for p in image_paths:
        mime, b64 = load_image_b64(p)
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    return {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        ],
    }


def main() -> None:
    prompt = "戴上帽子"
    image_paths = [
        r"C:\path\to\input.png",
    ]
    models_base = normalize_gemini_models_base(BASE_URL)
    url = f"{models_base}/{MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
    payload = build_payload(prompt, image_paths)

    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    print(resp.status_code)
    print(resp.text[:1000])
    resp.raise_for_status()
    data = resp.json()

    out_dir = Path("nano_banana_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for cand in data.get("candidates", []):
        if cand.get("finishReason") != "STOP":
            continue
        parts = cand.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData") if isinstance(part, dict) else None
            if not isinstance(inline, dict):
                continue
            mime = inline.get("mimeType", "image/png")
            b64 = inline.get("data")
            if not isinstance(b64, str) or not b64.strip():
                continue
            raw = base64.b64decode(b64)
            ext = "png"
            if mime == "image/jpeg":
                ext = "jpg"
            elif mime == "image/webp":
                ext = "webp"
            out_path = out_dir / f"out_{idx}.{ext}"
            out_path.write_bytes(raw)
            idx += 1

    meta_path = out_dir / "response.json"
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {idx} images to {out_dir}")


if __name__ == "__main__":
    main()

