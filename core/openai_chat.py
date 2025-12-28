import json
import re
import base64

from curl_cffi import CurlMime
from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class OpenAIChatProvider(BaseProvider):
    """OpenAI Chat 提供商"""

    api_type: str = "OpenAI_Chat"

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
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # 构建请求上下文
        openai_context = self._build_openai_chat_context(
            params.get("model", provider_config.model), image_b64_list, params
        )
        try:
            # 发送请求
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                json=openai_context,
                timeout=self.def_common_config.timeout,
                proxy=self.def_common_config.proxy,
            )
            # 响应反序列化
            result = response.json()
            if response.status_code == 200:
                b64_images = []
                images_url = []
                for item in result.get("choices", []):
                    # 检查 finish_reason 状态
                    finish_reason = item.get("finish_reason", "")
                    if finish_reason == "stop":
                        content = item.get("message", {}).get("content", "")
                        match = re.search(r"!\[.*?\]\((.*?)\)", content)
                        if match:
                            img_src = match.group(1)
                            if img_src.startswith("data:image/"):  # base64
                                header, base64_data = img_src.split(",", 1)
                                mime = header.split(";")[0].replace("data:", "")
                                b64_images.append((mime, base64_data))
                            else:  # URL
                                images_url.append(img_src)
                    else:
                        logger.warning(
                            f"[BIG BANANA] 图片生成失败, 响应内容: {response.text[:1024]}"
                        )
                        return None, 200, f"图片生成失败: {finish_reason}"
                # 最后再检查是否有图片数据
                if not images_url and not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                # 下载图片并转换为 base64
                b64_images += await self.downloader.fetch_images(images_url)
                if not b64_images:
                    return None, 200, "图片下载失败"
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
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # 构建请求上下文
        openai_context = self._build_openai_chat_context(
            params.get("model", provider_config.model), image_b64_list, params
        )
        try:
            # 发送请求
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                json=openai_context,
                proxy=self.def_common_config.proxy,
                stream=True,
            )
            # 处理流式响应
            streams = response.aiter_content(chunk_size=1024)
            # 读取完整内容
            data = b""
            async for chunk in streams:
                data += chunk
            result = data.decode("utf-8")
            if response.status_code == 200:
                b64_images = []
                images_url = []
                reasoning_content = ""
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
                                match = re.search(r"!\[.*?\]\((.*?)\)", content)
                                if match:
                                    img_src = match.group(1)
                                    if img_src.startswith("data:image/"):  # base64
                                        header, base64_data = img_src.split(",", 1)
                                        mime = header.split(";")[0].replace("data:", "")
                                        b64_images.append((mime, base64_data))
                                    else:  # URL
                                        images_url.append(img_src)
                                else:  # 尝试查找失败的原因或者纯文本返回结果
                                    reasoning_content += item.get("delta", {}).get(
                                        "reasoning_content", ""
                                    )
                        except json.JSONDecodeError:
                            continue
                if not images_url and not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {result[:1024]}"
                    )
                    return None, 200, reasoning_content or "响应中未包含图片数据"
                # 下载图片并转换为 base64（有时会出现连接被重置的错误，不知道什么原因，国外服务器也一样）
                b64_images += await self.downloader.fetch_images(images_url)
                if not b64_images:
                    return None, 200, "图片下载失败"
                return b64_images, 200, None
            else:
                logger.error(
                    f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {result[:1024]}"
                )
                return None, response.status_code, "响应中未包含图片数据"
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
        if self._is_edits_url(provider_config.api_url):
            return await self._call_edits_api(
                provider_config=provider_config,
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
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                json=body,
                timeout=self.def_common_config.timeout,
                proxy=self.def_common_config.proxy,
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
                name="image[]",
                content_type=mime or "image/png",
                filename=f"image_{idx}.{ext}",
                data=raw,
            )
            has_any_image = True

        if not has_any_image:
            return None, 400, "图片编辑失败：输入图片格式错误"

        try:
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                data=form,
                multipart=mp,
                timeout=self.def_common_config.timeout,
                proxy=self.def_common_config.proxy,
            )
            result = response.json()
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
