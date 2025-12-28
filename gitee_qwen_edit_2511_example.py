import contextlib
import json
import mimetypes
import os
import time

import requests
from requests_toolbelt import MultipartEncoder


API_URL = "https://ai.gitee.com/v1/async/images/edits"
API_TOKEN = "YOUR_API_TOKEN"


def query(payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    fields: list[tuple[str, object]] = [
        ("prompt", payload["prompt"]),
        ("model", payload.get("model", "Qwen-Image-Edit-2511")),
        ("num_inference_steps", str(payload.get("num_inference_steps", 4))),
        ("guidance_scale", str(payload.get("guidance_scale", 1))),
    ]
    with contextlib.ExitStack() as stack:
        for item in payload.get("task_types", []):
            fields.append(
                ("task_types", item if isinstance(item, str) else json.dumps(item))
            )
        for filepath in payload.get("image", []):
            name = os.path.basename(filepath)
            if filepath.startswith(("http://", "https://")):
                resp = requests.get(filepath, timeout=10)
                resp.raise_for_status()
                fields.append(
                    (
                        "image",
                        (
                            name,
                            resp.content,
                            resp.headers.get("Content-Type", "application/octet-stream"),
                        ),
                    )
                )
            else:
                mime_type, _ = mimetypes.guess_type(filepath)
                fields.append(
                    (
                        "image",
                        (
                            name,
                            stack.enter_context(open(filepath, "rb")),
                            mime_type or "application/octet-stream",
                        ),
                    )
                )
        encoder = MultipartEncoder(fields)
        headers["Content-Type"] = encoder.content_type
        resp = requests.post(API_URL, headers=headers, data=encoder, timeout=60)
        resp.raise_for_status()
        return resp.json()


def poll_task(task_id: str) -> dict:
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    status_url = f"https://ai.gitee.com/v1/task/{task_id}"
    timeout_seconds = 30 * 60
    retry_interval = 2
    started = time.time()
    while True:
        if time.time() - started > timeout_seconds:
            return {"status": "timeout", "message": "maximum wait time exceeded"}
        resp = requests.get(status_url, headers=headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            raise ValueError(f"{result['error']}: {result.get('message', 'Unknown error')}")
        status = result.get("status", "unknown")
        if status == "success":
            return result
        if status in ("failed", "cancelled"):
            return result
        time.sleep(retry_interval)


if __name__ == "__main__":
    result = query(
        {
            "prompt": "戴上帽子",
            "image": [
                "/path/to/image-id.png",
                "/path/to/image-style.png",
            ],
            "task_types": ["id", "style"],
            "model": "Qwen-Image-Edit-2511",
            "num_inference_steps": 4,
            "guidance_scale": 1,
        }
    )
    task_id = result.get("task_id")
    if not task_id:
        raise ValueError("Task ID not found in the response")
    task = poll_task(task_id)
    print(json.dumps(task, ensure_ascii=False, indent=2))
