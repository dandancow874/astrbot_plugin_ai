import asyncio
import base64
import copy
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from aiohttp import FormData
from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class ComfyUIProvider(BaseProvider):
    """本地/局域网 ComfyUI API Format 工作流提供商"""

    api_type: str = "ComfyUI"

    async def generate_images(
        self,
        provider_config: ProviderConfig,
        params: dict,
        image_b64_list: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        images_result, _, err = await self._call_api(
            provider_config=provider_config,
            api_key="",
            params=params,
            image_b64_list=image_b64_list,
        )
        return images_result, err

    @staticmethod
    def _base_url(api_url: str) -> str:
        return (api_url or "http://127.0.0.1:8188").strip().rstrip("/")

    @staticmethod
    def _disabled_message(base_url: str) -> str:
        return (
            f"ComfyUI 环境没有启用或无法连接（{base_url}）。"
            "请确认 ComfyUI 已启动，并且插件设置里的 ComfyUI 地址可从 AstrBot 机器访问。"
        )

    @staticmethod
    def _workflow_dirs() -> list[Path]:
        root = Path(__file__).resolve().parent.parent
        return [root / "workflow", root / "workflows"]

    @classmethod
    def _load_workflow(cls, workflow_name: str) -> tuple[dict | None, str | None]:
        name = (workflow_name or "").strip().strip("/\\")
        if not name:
            return None, "缺少 ComfyUI 工作流名"
        candidates = [name]
        if not name.lower().endswith(".json"):
            candidates.append(f"{name}.json")
        for base_dir in cls._workflow_dirs():
            for filename in candidates:
                path = (base_dir / filename).resolve()
                try:
                    path.relative_to(base_dir.resolve())
                except ValueError:
                    continue
                if not path.is_file():
                    continue
                try:
                    return json.loads(path.read_text(encoding="utf-8")), None
                except Exception as e:
                    return None, f"读取 ComfyUI 工作流失败：{path.name}，{e}"
        return None, f"未找到 ComfyUI 工作流：{workflow_name}.json"

    @staticmethod
    def _split_workflow_prompt(prompt: str) -> tuple[str, str]:
        text = (prompt or "").strip()
        if not text:
            return "", ""
        workflow, _, user_prompt = text.partition(" ")
        return workflow.strip(), user_prompt.strip()

    @staticmethod
    def _replace_placeholders(obj, values: dict[str, str]):
        if isinstance(obj, dict):
            return {
                key: ComfyUIProvider._replace_placeholders(value, values)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [ComfyUIProvider._replace_placeholders(item, values) for item in obj]
        if isinstance(obj, str):
            result = obj
            for key, value in values.items():
                result = result.replace(key, value)
            return result
        return obj

    @staticmethod
    def _workflow_text(workflow: dict) -> str:
        try:
            return json.dumps(workflow, ensure_ascii=False)
        except Exception:
            return ""

    @staticmethod
    def _decode_image(b64: str) -> bytes | None:
        try:
            return base64.b64decode(b64, validate=False)
        except Exception:
            return None

    async def _upload_image(
        self,
        base_url: str,
        index: int,
        mime: str,
        b64: str,
    ) -> tuple[str | None, str | None]:
        if self.aiohttp_session is None:
            return None, "ComfyUI 图片上传失败：aiohttp session 未初始化"
        raw = self._decode_image(b64)
        if not raw:
            return None, f"ComfyUI 图片上传失败：第 {index} 张图片不是有效 base64"
        ext = mimetypes.guess_extension(mime or "") or ".png"
        filename = f"astrbot_{int(time.time())}_{uuid.uuid4().hex[:8]}_{index}{ext}"
        form = FormData()
        form.add_field(
            "image",
            raw,
            filename=filename,
            content_type=mime or "image/png",
        )
        form.add_field("type", "input")
        form.add_field("overwrite", "true")
        try:
            async with self.aiohttp_session.post(
                f"{base_url}/upload/image",
                data=form,
                proxy=self.def_common_config.proxy,
                timeout=self.def_common_config.timeout,
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    return None, f"ComfyUI 图片上传失败：HTTP {response.status} {text[:200]}"
                try:
                    payload = json.loads(text)
                except Exception:
                    payload = {}
                return payload.get("name") or filename, None
        except Exception as e:
            logger.error(f"[ComfyUI] 图片上传连接失败: {e}")
            return None, self._disabled_message(base_url)

    async def _build_values(
        self,
        base_url: str,
        workflow: dict,
        user_prompt: str,
        image_b64_list: list[tuple[str, str]],
    ) -> tuple[dict[str, str] | None, str | None]:
        values = {
            "{{prompt}}": user_prompt,
            "{{seed}}": str(int(time.time() * 1000) % 1_000_000_000_000),
        }
        workflow_text = self._workflow_text(workflow)
        for idx, item in enumerate(image_b64_list[:2], start=1):
            mime, b64 = item
            values[f"{{{{image{idx}_base64}}}}"] = b64
            values[f"{{{{image{idx}_data_url}}}}"] = f"data:{mime or 'image/png'};base64,{b64}"
            if f"{{{{image{idx}}}}}" in workflow_text:
                filename, err = await self._upload_image(base_url, idx, mime, b64)
                if err:
                    return None, err
                values[f"{{{{image{idx}}}}}"] = filename or ""
        return values, None

    @staticmethod
    def _find_unresolved_placeholders(workflow: dict) -> list[str]:
        text = ComfyUIProvider._workflow_text(workflow)
        found = []
        for name in [
            "{{prompt}}",
            "{{image1}}",
            "{{image2}}",
            "{{image1_base64}}",
            "{{image2_base64}}",
            "{{image1_data_url}}",
            "{{image2_data_url}}",
        ]:
            if name in text:
                found.append(name)
        return found

    @staticmethod
    def _extract_output_images(history: dict) -> list[dict]:
        outputs = history.get("outputs") if isinstance(history, dict) else None
        if not isinstance(outputs, dict):
            return []
        images: list[dict] = []
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            node_images = node_output.get("images")
            if isinstance(node_images, list):
                for item in node_images:
                    if isinstance(item, dict):
                        images.append(item)
        return images

    async def _fetch_output_images(
        self, base_url: str, image_items: list[dict]
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        results: list[tuple[str, str]] = []
        for item in image_items:
            filename = item.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                continue
            query = urlencode(
                {
                    "filename": filename,
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                }
            )
            url = f"{base_url}/view?{query}"
            try:
                response = await self.session.get(
                    url,
                    proxy=self.def_common_config.proxy,
                    timeout=self.def_common_config.timeout,
                )
                if response.status_code >= 400:
                    return None, f"ComfyUI 下载结果失败：HTTP {response.status_code}"
                mime = "image/png"
                try:
                    content_type = response.headers.get("Content-Type") or ""
                    if isinstance(content_type, str) and content_type.startswith("image/"):
                        mime = content_type.split(";", 1)[0].strip()
                except Exception:
                    pass
                b64 = base64.b64encode(response.content).decode("utf-8")
                results.append((mime, b64))
            except Exception as e:
                return None, f"ComfyUI 下载结果失败：{e}"
        if not results:
            return None, "ComfyUI 任务完成但没有输出图片"
        return results, None

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        base_url = self._base_url(provider_config.api_url)
        workflow_name, user_prompt = self._split_workflow_prompt(params.get("prompt", ""))
        workflow, err = self._load_workflow(workflow_name)
        if err:
            return None, 400, err
        assert workflow is not None

        workflow = copy.deepcopy(workflow)
        values, err = await self._build_values(
            base_url, workflow, user_prompt, image_b64_list
        )
        if err:
            return None, 400, err
        assert values is not None
        workflow = self._replace_placeholders(workflow, values)
        unresolved = self._find_unresolved_placeholders(workflow)
        if unresolved:
            return None, 400, f"ComfyUI 工作流缺少变量：{', '.join(unresolved)}"

        client_id = f"astrbot-{uuid.uuid4().hex}"
        try:
            response = await self.session.post(
                f"{base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                proxy=self.def_common_config.proxy,
                timeout=self.def_common_config.timeout,
            )
            if response.status_code >= 400:
                return None, response.status_code, f"ComfyUI 提交失败：{response.text[:500]}"
            payload = response.json()
            prompt_id = payload.get("prompt_id")
            if not prompt_id:
                return None, 502, "ComfyUI 未返回 prompt_id"

            deadline = time.time() + float(self.def_common_config.timeout or 300)
            while time.time() < deadline:
                await asyncio.sleep(2)
                history_resp = await self.session.get(
                    f"{base_url}/history/{prompt_id}",
                    proxy=self.def_common_config.proxy,
                    timeout=30,
                )
                if history_resp.status_code >= 400:
                    return None, history_resp.status_code, "ComfyUI 查询任务结果失败"
                history_payload = history_resp.json()
                history = history_payload.get(prompt_id)
                if isinstance(history, dict):
                    output_images = self._extract_output_images(history)
                    if output_images:
                        images, fetch_err = await self._fetch_output_images(
                            base_url, output_images
                        )
                        if images:
                            return images, 200, None
                        return None, 502, fetch_err
            return None, 408, "ComfyUI 任务超时"
        except Timeout as e:
            logger.error(f"[ComfyUI] 请求超时: {e}")
            return None, 408, "ComfyUI 请求超时"
        except Exception as e:
            logger.error(f"[ComfyUI] 请求错误: {e}", exc_info=True)
            return None, 503, self._disabled_message(base_url)

    async def _call_stream_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        return await self._call_api(
            provider_config=provider_config,
            api_key=api_key,
            image_b64_list=image_b64_list,
            params=params,
        )
