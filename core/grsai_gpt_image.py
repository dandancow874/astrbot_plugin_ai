import json
import asyncio
import time

from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class GrsaiGPTImageProvider(BaseProvider):
    """Grsai GPT Image 提供商"""

    api_type: str = "Grsai_GPT_Image"

    @staticmethod
    def _resolve_draw_url(api_url: str) -> str:
        raw = (api_url or "").strip().rstrip("/")
        lowered = raw.lower()
        root = raw
        if "/v1/" in lowered:
            root = raw.split("/v1/", 1)[0]
        elif lowered.endswith("/v1"):
            root = raw[: -len("/v1")]
        if "/chat/completions" in root.lower():
            root = root.split("/chat/completions", 1)[0]
        return root.rstrip("/") + "/v1/draw/completions"

    @staticmethod
    def _resolve_result_url(api_url: str) -> str:
        raw = (api_url or "").strip().rstrip("/")
        lowered = raw.lower()
        root = raw
        if "/v1/" in lowered:
            root = raw.split("/v1/", 1)[0]
        elif lowered.endswith("/v1"):
            root = raw[: -len("/v1")]
        if "/chat/completions" in root.lower():
            root = root.split("/chat/completions", 1)[0]
        return root.rstrip("/") + "/v1/draw/result"

    @staticmethod
    def _extract_error_message(payload_text: str) -> str | None:
        if not isinstance(payload_text, str) or not payload_text.strip():
            return None
        try:
            data = json.loads(payload_text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if isinstance(data.get("msg"), str) and data["msg"].strip():
            return data["msg"].strip()
        if isinstance(data.get("message"), str) and data["message"].strip():
            return data["message"].strip()
        err = data.get("error")
        if (
            isinstance(err, dict)
            and isinstance(err.get("message"), str)
            and err["message"].strip()
        ):
            return err["message"].strip()
        failure = data.get("failure_reason")
        if isinstance(failure, str) and failure.strip():
            return failure.strip()
        return None

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        model = (params.get("model") or provider_config.model or "").strip()
        prompt = params.get("prompt", "anything")
        urls = params.get("__source_image_urls__")
        if not isinstance(urls, list):
            urls = []
        urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "size": "1:1",
        }
        if urls:
            payload["urls"] = urls

        aspect_ratio = params.get("aspect_ratio")
        if isinstance(aspect_ratio, str) and aspect_ratio.strip():
            ar = aspect_ratio.strip()
            if ar and ar.lower() != "default":
                size_map = {
                    "1:1": "1:1",
                    "3:2": "3:2",
                    "2:3": "2:3",
                    "16:9": "16:9",
                    "9:16": "9:16",
                }
                payload["size"] = size_map.get(ar, ar)

        draw_url = self._resolve_draw_url(provider_config.api_url)
        try:
            impersonate = (
                provider_config.impersonate.strip()
                if isinstance(provider_config.impersonate, str)
                and provider_config.impersonate.strip()
                else None
            )
            verify = (
                provider_config.tls_verify
                if isinstance(provider_config.tls_verify, bool)
                else True
            )
            req_kwargs = {
                "timeout": self.def_common_config.timeout,
                "proxy": self.def_common_config.proxy,
                "verify": verify,
            }
            if impersonate:
                req_kwargs["impersonate"] = impersonate

            response = await self.session.post(
                url=draw_url,
                headers=headers,
                json=payload,
                **req_kwargs,
            )
            resp_text = getattr(response, "text", "")

            if isinstance(resp_text, str):
                stripped = resp_text.lstrip()
                lowered = stripped.lower()
                if (
                    lowered.startswith("<!doctype html")
                    or lowered.startswith("<html")
                    or "<html" in lowered
                ):
                    return (
                        None,
                        502,
                        f"上游返回HTML，可能是鉴权失败或接口路径错误（{draw_url}）",
                    )

            if response.status_code != 200:
                detail = self._extract_error_message(
                    resp_text if isinstance(resp_text, str) else ""
                )
                return (
                    None,
                    response.status_code,
                    f"图片生成失败: {detail}"
                    if detail
                    else f"图片生成失败: 状态码 {response.status_code}",
                )

            result_url_list: list[str] = []
            if isinstance(resp_text, str):
                for line in resp_text.splitlines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:].strip()
                    if not json_str:
                        continue
                    try:
                        data = json.loads(json_str)
                    except Exception:
                        continue

                    if not isinstance(data, dict):
                        continue

                    status = data.get("status")
                    progress = data.get("progress")
                    if status == "succeeded" or progress == 100:
                        url = data.get("url")
                        if isinstance(url, str) and url.strip():
                            result_url_list.append(url.strip())
                        results = data.get("results")
                        if isinstance(results, list):
                            for item in results:
                                if isinstance(item, dict):
                                    u = item.get("url")
                                    if isinstance(u, str) and u.strip():
                                        result_url_list.append(u.strip())

                    if status == "failed":
                        reason = data.get("failure_reason") or data.get("message") or "生成失败"
                        return None, 400, f"图片生成失败: {reason}"

            if not result_url_list:
                try:
                    data = json.loads(resp_text) if isinstance(resp_text, str) else {}
                except Exception:
                    data = {}
                if isinstance(data, dict):
                    detail = self._extract_error_message(resp_text)
                    if detail:
                        return None, 400, f"图片生成失败: {detail}"
                    # 尝试轮询模式：检查是否有 task_id
                    task_id = None
                    data_obj = data.get("data")
                    if isinstance(data_obj, dict):
                        tid = data_obj.get("id")
                        if isinstance(tid, str) and tid.strip():
                            task_id = tid.strip()
                    if task_id:
                        return await self._poll_result(
                            provider_config, headers, task_id, verify, impersonate
                        )

                return None, 200, "响应中未包含图片数据"

            result_url_list = list(dict.fromkeys(result_url_list))
            b64_images = await self.downloader.fetch_media(result_url_list)
            b64_images = [(mime, b64) for mime, b64 in b64_images if b64]
            if not b64_images:
                return None, 200, "图片下载失败"
            return b64_images, 200, None

        except Timeout as e:
            logger.error(f"[GPT Image] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except Exception as e:
            logger.error(f"[GPT Image] 请求错误: {e}, url={draw_url}")
            return None, None, "图片生成失败：程序错误"

    async def _poll_result(
        self,
        provider_config: ProviderConfig,
        headers: dict,
        task_id: str,
        verify: bool,
        impersonate: str | None,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        result_url = self._resolve_result_url(provider_config.api_url)
        poll_kwargs: dict = {
            "timeout": 30,
            "proxy": self.def_common_config.proxy,
            "verify": verify,
        }
        if impersonate:
            poll_kwargs["impersonate"] = impersonate

        started = time.time()
        max_wait = self.def_common_config.timeout
        while True:
            elapsed = time.time() - started
            if elapsed > max_wait:
                return None, 408, "图片生成失败：任务轮询超时"

            try:
                resp = await self.session.post(
                    result_url,
                    headers=headers,
                    json={"id": task_id},
                    **poll_kwargs,
                )
                resp_text = getattr(resp, "text", "")
                result = json.loads(resp_text) if isinstance(resp_text, str) else {}
            except Exception:
                await asyncio.sleep(3)
                continue

            if not isinstance(result, dict):
                await asyncio.sleep(3)
                continue

            code = result.get("code")
            data = result.get("data")
            if not isinstance(data, dict):
                await asyncio.sleep(3)
                continue

            status = data.get("status")
            if status == "succeeded":
                media_urls: list[str] = []
                results = data.get("results")
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict):
                            u = item.get("url")
                            if isinstance(u, str) and u.strip():
                                media_urls.append(u.strip())
                url = data.get("url")
                if isinstance(url, str) and url.strip():
                    media_urls.append(url.strip())

                if not media_urls:
                    return None, 200, "任务成功但未返回图片"
                media_urls = list(dict.fromkeys(media_urls))
                b64_images = await self.downloader.fetch_media(media_urls)
                b64_images = [(m, b) for m, b in b64_images if b]
                if not b64_images:
                    return None, 200, "图片下载失败"
                return b64_images, 200, None

            if status in ("failed", "cancelled"):
                reason = data.get("failure_reason") or data.get("message") or status
                return None, 400, f"图片生成失败：任务{reason}"

            await asyncio.sleep(3)

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
