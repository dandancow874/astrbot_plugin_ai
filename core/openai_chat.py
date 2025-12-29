import json
import re
import base64
import asyncio
import time
from urllib.parse import urlparse

from aiohttp import ClientTimeout
from curl_cffi import CurlMime
from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class OpenAIChatProvider(BaseProvider):
    """OpenAI Chat 提供商"""

    api_type: str = "OpenAI_Chat"

    def _extract_media_sources(
        self, content: str
    ) -> tuple[list[tuple[str, str]], list[str]]:
        b64_items: list[tuple[str, str]] = []
        urls: list[str] = []

        if not isinstance(content, str) or not content.strip():
            return b64_items, urls

        candidates: list[str] = []
        candidates.extend(re.findall(r"!\[[^\]]*?\]\((.*?)\)", content))
        candidates.extend(re.findall(r"\[[^\]]*?\]\((https?://.*?)\)", content))
        candidates.extend(re.findall(r"(https?://[^\s\)\]]+)", content))

        for raw in candidates:
            src = (raw or "").strip().strip("<>").strip()
            src = src.rstrip(").,;\"'")
            if not src:
                continue
            if src.startswith("data:"):
                try:
                    header, base64_data = src.split(",", 1)
                    mime = header.split(";")[0].replace("data:", "").strip()
                    if mime:
                        b64_items.append((mime, base64_data))
                except Exception:
                    continue
                continue
            parsed = urlparse(src)
            if parsed.scheme in {"http", "https"}:
                urls.append(src)

        urls = list(dict.fromkeys(urls))
        return b64_items, urls

    def _extract_error_message(self, payload_text: str) -> str | None:
        if not isinstance(payload_text, str) or not payload_text.strip():
            return None
        try:
            data = json.loads(payload_text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if isinstance(data.get("message"), str) and data["message"].strip():
            return data["message"].strip()
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str) and err["message"].strip():
            return err["message"].strip()
        detail = data.get("detail")
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and isinstance(first.get("msg"), str) and first["msg"].strip():
                return first["msg"].strip()
        return None

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """发起 OpenAI 图片生成请求
        返回值: 元组(图片 base64 列表, 状态码, 人类可读的错误信息)
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # 构建请求上下文
        openai_context = self._build_openai_chat_context(
            params.get("model", provider_config.model), image_b64_list, params
        )
        openai_context["stream"] = False
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
            # 发送请求
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                json=openai_context,
                **req_kwargs,
            )
            # 响应反序列化
            result = response.json()
            if response.status_code == 200:
                b64_images: list[tuple[str, str]] = []
                media_urls: list[str] = []
                for item in result.get("choices", []):
                    # 检查 finish_reason 状态
                    finish_reason = item.get("finish_reason", "")
                    if finish_reason == "stop":
                        content = item.get("message", {}).get("content", "")
                        b64_part, url_part = self._extract_media_sources(content)
                        if b64_part:
                            b64_images.extend(b64_part)
                        if url_part:
                            media_urls.extend(url_part)
                    else:
                        logger.warning(
                            f"[BIG BANANA] 图片生成失败, 响应内容: {response.text[:1024]}"
                        )
                        return None, 200, f"图片生成失败: {finish_reason}"
                # 最后再检查是否有图片数据
                if not media_urls and not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                # 下载媒体并转换为 base64
                if media_urls:
                    b64_images += await self.downloader.fetch_media(media_urls)
                if not b64_images:
                    return None, 200, "媒体下载失败"
                return b64_images, 200, None
            else:
                logger.error(
                    f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {response.text[:1024]}"
                )
                detail = None
                if isinstance(result, dict):
                    if isinstance(result.get("message"), str):
                        detail = result.get("message")
                    err_obj = result.get("error")
                    if not detail and isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                        detail = err_obj.get("message")
                return (
                    None,
                    response.status_code,
                    f"图片生成失败: {detail}" if detail else f"图片生成失败: 状态码 {response.status_code}",
                )
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except json.JSONDecodeError as e:
            logger.error(
                f"[BIG BANANA] JSON反序列化错误: {e}，状态码：{response.status_code}，响应内容：{response.text[:1024]}"
            )
            return None, response.status_code, "图片生成失败：响应内容格式错误"
        except Exception as e:
            logger.error(f"[BIG BANANA] 请求错误: {e}")
            return None, None, "图片生成失败：程序错误"

    async def _call_stream_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """发起 OpenAI 图片生成流式请求
        返回值: 元组(图片 base64 列表, 状态码, 人类可读的错误信息)
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        # 构建请求上下文
        openai_context = self._build_openai_chat_context(
            params.get("model", provider_config.model), image_b64_list, params
        )
        openai_context["stream"] = True
        try:
            use_aiohttp = (
                self.aiohttp_session is not None
                and isinstance(provider_config.api_url, str)
                and "/v1/chat/completions" in provider_config.api_url
                and provider_config.stream
            )
            if use_aiohttp:
                timeout = ClientTimeout(total=float(self.def_common_config.timeout))
                proxy = (
                    self.def_common_config.proxy
                    if isinstance(self.def_common_config.proxy, str)
                    and self.def_common_config.proxy.strip()
                    else None
                )
                ssl = None
                if provider_config.tls_verify is False:
                    ssl = False
                async with self.aiohttp_session.post(
                    provider_config.api_url,
                    headers=headers,
                    json=openai_context,
                    timeout=timeout,
                    proxy=proxy,
                    ssl=ssl,
                ) as resp:
                    result = await resp.text()
                    if resp.status == 200:
                        reasoning_content = ""
                        content_buf_parts: list[str] = []
                        reasoning_buf_parts: list[str] = []
                        for line in result.splitlines():
                            if line.startswith("data: "):
                                line_data = line[len("data: ") :].strip()
                                if line_data == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(line_data)
                                    for item in json_data.get("choices", []):
                                        content = item.get("delta", {}).get("content", "")
                                        if isinstance(content, str) and content:
                                            content_buf_parts.append(content)
                                        rc = item.get("delta", {}).get("reasoning_content", "")
                                        if isinstance(rc, str) and rc:
                                            reasoning_buf_parts.append(rc)
                                except json.JSONDecodeError:
                                    continue
                        reasoning_content = "".join(reasoning_buf_parts)
                        full_text = "".join(content_buf_parts) + reasoning_content
                        b64_images, media_urls = self._extract_media_sources(full_text)
                        if media_urls:
                            b64_images += await self.downloader.fetch_media(media_urls)
                        if not b64_images:
                            if reasoning_content.strip():
                                logger.warning(
                                    f"[BIG BANANA] 请求成功，但未返回媒体数据, 响应内容: {result[:1024]}"
                                )
                                return None, 200, reasoning_content.strip()
                            logger.warning(
                                f"[BIG BANANA] 请求成功，但未返回媒体数据, 响应内容: {result[:1024]}"
                            )
                            return None, 200, "响应中未包含媒体数据"
                        return b64_images, 200, None
                    logger.error(
                        f"[BIG BANANA] 图片生成失败，状态码: {resp.status}, 响应内容: {result[:1024]}"
                    )
                    detail = self._extract_error_message(result)
                    return (
                        None,
                        int(resp.status),
                        f"图片生成失败: {detail}"
                        if detail
                        else f"图片生成失败: 状态码 {resp.status}",
                    )
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
                "proxy": self.def_common_config.proxy,
                "timeout": self.def_common_config.timeout,
                "verify": verify,
            }
            if impersonate:
                req_kwargs["impersonate"] = impersonate
            async def post_request(use_data: bool):
                if use_data:
                    payload = json.dumps(openai_context, ensure_ascii=False)
                    return await self.session.post(
                        url=provider_config.api_url,
                        headers=headers,
                        data=payload,
                        **req_kwargs,
                    )
                return await self.session.post(
                    url=provider_config.api_url,
                    headers=headers,
                    json=openai_context,
                    **req_kwargs,
                )

            async def read_response_text(resp) -> str:
                text = getattr(resp, "text", None)
                if isinstance(text, str):
                    return text
                data = b""
                async for chunk in resp.aiter_content(chunk_size=1024):
                    data += chunk
                return data.decode("utf-8", errors="replace")

            response = await post_request(use_data=False)
            result = await read_response_text(response)
            if response.status_code == 422:
                detail = self._extract_error_message(result)
                if detail == "Field required":
                    response = await post_request(use_data=True)
                    result = await read_response_text(response)
            if response.status_code == 200:
                reasoning_content = ""
                content_buf_parts: list[str] = []
                reasoning_buf_parts: list[str] = []
                for line in result.splitlines():
                    if line.startswith("data: "):
                        line_data = line[len("data: ") :].strip()
                        if line_data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(line_data)
                            # 遍历 json_data，检查是否有图片
                            for item in json_data.get("choices", []):
                                content = item.get("delta", {}).get("content", "")
                                if isinstance(content, str) and content:
                                    content_buf_parts.append(content)
                                rc = item.get("delta", {}).get("reasoning_content", "")
                                if isinstance(rc, str) and rc:
                                    reasoning_buf_parts.append(rc)
                        except json.JSONDecodeError:
                            continue
                reasoning_content = "".join(reasoning_buf_parts)
                full_text = "".join(content_buf_parts) + reasoning_content
                b64_images, media_urls = self._extract_media_sources(full_text)
                if media_urls:
                    b64_images += await self.downloader.fetch_media(media_urls)
                if not b64_images:
                    if reasoning_content.strip():
                        logger.warning(
                            f"[BIG BANANA] 请求成功，但未返回媒体数据, 响应内容: {result[:1024]}"
                        )
                        return None, 200, reasoning_content.strip()
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回媒体数据, 响应内容: {result[:1024]}"
                    )
                    return None, 200, "响应中未包含媒体数据"
                return b64_images, 200, None
            else:
                logger.error(
                    f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {result[:1024]}"
                )
                detail = self._extract_error_message(result)
                return (
                    None,
                    response.status_code,
                    f"图片生成失败: {detail}"
                    if detail
                    else f"图片生成失败: 状态码 {response.status_code}",
                )
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except Exception as e:
            logger.error(f"[BIG BANANA] 请求错误: {e}")
            return None, None, "图片生成失败：程序错误"

    def _build_openai_chat_context(
        self,
        model: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> dict:
        images_content = []
        for mime, b64 in image_b64_list:
            images_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        context = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": params.get("prompt", "anything")},
                        *images_content,
                    ],
                }
            ],
            "stream": params.get("stream", False),
        }
        return context


class OpenAIImagesProvider(BaseProvider):
    api_type: str = "OpenAI_Images"

    @staticmethod
    def _is_edits_url(url: object) -> bool:
        if not isinstance(url, str):
            return False
        u = url.lower()
        return "/images/edits" in u or u.endswith("/edits")

    @staticmethod
    def _resolve_images_url(api_url: str, prefer_edits: bool) -> str:
        base = (api_url or "").strip().rstrip("/")
        lower = base.lower()
        if "/images/" in lower or lower.endswith("/edits") or lower.endswith("/generations"):
            return base

        if base.endswith("/v1/async"):
            return f"{base}/images/edits" if prefer_edits else f"{base}/images/generations"
        if base.endswith("/v1"):
            return f"{base}/images/edits" if prefer_edits else f"{base}/images/generations"

        if prefer_edits:
            return f"{base}/v1/images/edits"
        return f"{base}/images/generations"

    @staticmethod
    def _guess_mime_from_b64(b64_data: str) -> str:
        head = (b64_data or "")[:128]
        if not head:
            return "image/png"
        pad_len = (-len(head)) % 4
        head_padded = head + ("=" * pad_len)
        try:
            raw = base64.b64decode(head_padded, validate=False)
        except Exception:
            return "image/png"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
            return "image/webp"
        return "image/png"

    @staticmethod
    def _map_image_size(image_size: object) -> str | None:
        if not isinstance(image_size, str):
            return None
        size = image_size.strip().upper()
        if size == "1K":
            return "1024x1024"
        if size == "2K":
            return "2048x2048"
        if size == "4K":
            return "4096x4096"
        if "X" in size and all(p.isdigit() for p in size.split("X", 1)):
            return size.lower().replace("x", "x")
        return None

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        prefer_edits = bool(image_b64_list)
        resolved_url = self._resolve_images_url(provider_config.api_url, prefer_edits)
        if self._is_edits_url(resolved_url):
            cfg = ProviderConfig(
                name=provider_config.name,
                enabled=provider_config.enabled,
                api_type=provider_config.api_type,
                keys=provider_config.keys,
                api_url=resolved_url,
                model=provider_config.model,
                stream=provider_config.stream,
                tls_verify=provider_config.tls_verify,
                impersonate=provider_config.impersonate,
            )
            return await self._call_edits_api(
                provider_config=cfg,
                api_key=api_key,
                image_b64_list=image_b64_list,
                params=params,
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        body: dict = {
            "prompt": params.get("prompt", "anything"),
            "model": params.get("model", provider_config.model),
            "response_format": "b64_json",
        }
        mapped_size = self._map_image_size(params.get("image_size"))
        if mapped_size:
            body["size"] = mapped_size

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
                url=resolved_url,
                headers=headers,
                json=body,
                **req_kwargs,
            )
            result = response.json()
            if response.status_code == 200:
                b64_images: list[tuple[str, str]] = []
                images_url: list[str] = []
                for item in (result.get("data") or []):
                    if not isinstance(item, dict):
                        continue
                    b64 = item.get("b64_json")
                    if isinstance(b64, str) and b64.strip():
                        mime = self._guess_mime_from_b64(b64)
                        b64_images.append((mime, b64))
                        continue
                    url = item.get("url")
                    if isinstance(url, str) and url.strip():
                        images_url.append(url)

                if images_url:
                    b64_images += await self.downloader.fetch_images(images_url)

                b64_images = [(mime, b64) for mime, b64 in b64_images if b64]
                if not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                return b64_images, 200, None

            logger.error(
                f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {response.text[:1024]}"
            )
            detail = None
            if isinstance(result, dict):
                if isinstance(result.get("message"), str):
                    detail = result.get("message")
                err_obj = result.get("error")
                if not detail and isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                    detail = err_obj.get("message")
            return None, response.status_code, f"图片生成失败: {detail}" if detail else f"图片生成失败: 状态码 {response.status_code}"
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except json.JSONDecodeError as e:
            logger.error(
                f"[BIG BANANA] JSON反序列化错误: {e}，状态码：{response.status_code}，响应内容：{response.text[:1024]}"
            )
            return None, response.status_code, "图片生成失败：响应内容格式错误"
        except Exception as e:
            logger.error(f"[BIG BANANA] 请求错误: {e}")
            return None, None, "图片生成失败：程序错误"

    @staticmethod
    def _mime_to_ext(mime: str) -> str:
        m = (mime or "").lower()
        if m == "image/jpeg":
            return "jpg"
        if m == "image/webp":
            return "webp"
        if m == "image/gif":
            return "gif"
        return "png"

    async def _call_edits_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        if not image_b64_list:
            return None, 400, "图片编辑失败：缺少输入图片"

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        form: dict = {
            "prompt": params.get("prompt", "anything"),
            "model": params.get("model", provider_config.model),
            "response_format": "b64_json",
        }
        mapped_size = self._map_image_size(params.get("image_size"))
        if mapped_size:
            form["size"] = mapped_size

        mp = CurlMime()
        has_any_image = False
        for idx, (mime, b64) in enumerate(image_b64_list):
            if not b64:
                continue
            try:
                raw = base64.b64decode(b64, validate=False)
            except Exception:
                continue
            ext = self._mime_to_ext(mime)
            mp.addpart(
                name="image",
                content_type=mime or "image/png",
                filename=f"image_{idx}.{ext}",
                data=raw,
            )
            has_any_image = True
            break

        if not has_any_image:
            return None, 400, "图片编辑失败：输入图片格式错误"

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
                url=provider_config.api_url,
                headers=headers,
                data=form,
                multipart=mp,
                **req_kwargs,
            )
            result = response.json()
            if (
                response.status_code == 200
                and isinstance(result, dict)
                and isinstance(result.get("task_id"), str)
                and result.get("task_id").strip()
            ):
                task_id = result.get("task_id").strip()
                try:
                    parsed = urlparse(provider_config.api_url)
                    task_url = f"{parsed.scheme}://{parsed.netloc}/v1/task/{task_id}"
                except Exception:
                    task_url = ""
                if not task_url:
                    return None, 200, "图片编辑失败：无法解析任务查询地址"

                started = time.time()
                poll_kwargs = {
                    "timeout": 30,
                    "proxy": self.def_common_config.proxy,
                    "verify": verify,
                }
                if impersonate:
                    poll_kwargs["impersonate"] = impersonate

                while True:
                    if time.time() - started > 1800:
                        return None, 408, "图片编辑失败：任务超时"

                    task_resp = await self.session.get(
                        task_url,
                        headers=headers,
                        **poll_kwargs,
                    )
                    try:
                        task_json = task_resp.json()
                    except Exception:
                        task_json = None

                    if not isinstance(task_json, dict):
                        await asyncio.sleep(2)
                        continue
                    if task_json.get("error"):
                        msg = (
                            task_json.get("message")
                            if isinstance(task_json.get("message"), str)
                            else None
                        )
                        return None, 200, f"图片编辑失败：{msg or str(task_json.get('error'))}"
                    status = task_json.get("status")
                    if status == "success":
                        output = task_json.get("output")
                        if (
                            isinstance(output, dict)
                            and isinstance(output.get("file_url"), str)
                            and output.get("file_url").strip()
                        ):
                            file_url = output.get("file_url").strip()
                            b64_images = await self.downloader.fetch_images([file_url])
                            b64_images = [(m, b) for m, b in b64_images if b]
                            if not b64_images:
                                return None, 200, "图片编辑失败：结果图片下载失败"
                            return b64_images, 200, None
                        return None, 200, "图片编辑失败：任务成功但未返回图片"
                    if status in ("failed", "cancelled"):
                        return None, 200, f"图片编辑失败：任务{status}"

                    await asyncio.sleep(2)
            if response.status_code == 200:
                b64_images: list[tuple[str, str]] = []
                images_url: list[str] = []
                data = result.get("data")
                if isinstance(data, dict):
                    data = [data]
                for item in (data or []):
                    if not isinstance(item, dict):
                        continue
                    b64_out = item.get("b64_json")
                    if isinstance(b64_out, str) and b64_out.strip():
                        mime_out = self._guess_mime_from_b64(b64_out)
                        b64_images.append((mime_out, b64_out))
                        continue
                    url = item.get("url")
                    if isinstance(url, str) and url.strip():
                        images_url.append(url)

                if images_url:
                    b64_images += await self.downloader.fetch_images(images_url)

                b64_images = [(mime, b64) for mime, b64 in b64_images if b64]
                if not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                return b64_images, 200, None

            logger.error(
                f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {response.text[:1024]}"
            )
            detail = None
            if isinstance(result, dict):
                if isinstance(result.get("message"), str):
                    detail = result.get("message")
                err_obj = result.get("error")
                if not detail and isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                    detail = err_obj.get("message")
            return None, response.status_code, f"图片生成失败: {detail}" if detail else f"图片生成失败: 状态码 {response.status_code}"
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except json.JSONDecodeError as e:
            logger.error(
                f"[BIG BANANA] JSON反序列化错误: {e}，状态码：{response.status_code}，响应内容：{response.text[:1024]}"
            )
            return None, response.status_code, "图片生成失败：响应内容格式错误"
        except Exception as e:
            logger.error(f"[BIG BANANA] 请求错误: {e}")
            return None, None, "图片生成失败：程序错误"

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
