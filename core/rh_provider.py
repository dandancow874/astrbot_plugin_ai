import json
from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class RHProvider(BaseProvider):
    """RH 供应商 - 用于 nano-banana 模型"""

    api_type: str = "RH_Provider"

    # RH 基础 URL
    RH_BASE_URL = "https://www.runninghub.cn"

    # 接口路径映射
    ENDPOINT_MAP = {
        # nano-banana-2 模型
        "nano-banana-2": {
            "text_to_image": "/rhart-image-n-g31-flash/text-to-image",
            "image_to_image": "/rhart-image-n-g31-flash/image-to-image",
        },
        # nano-banana-pro 模型
        "nano-banana-pro": {
            "text_to_image": "/rhart-image-n-pro/text-to-image",
            "image_to_image": "/rhart-image-n-pro/edit",
        },
    }

    def _get_endpoint(self, model: str, has_images: bool) -> str:
        """获取对应的接口路径"""
        model_key = model if model in self.ENDPOINT_MAP else "nano-banana-2"
        endpoint_type = "image_to_image" if has_images else "text_to_image"
        path = self.ENDPOINT_MAP[model_key][endpoint_type]
        return f"{self.RH_BASE_URL}{path}"

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """发起 RH 图片生成请求
        返回值: 元组(图片 base64 列表, 状态码, 人类可读的错误信息)
        """
        model = params.get("model", provider_config.model or "nano-banana-2")
        has_images = bool(image_b64_list)
        url = self._get_endpoint(model, has_images)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 构建请求体
        body: dict = {
            "prompt": params.get("prompt", "anything"),
        }

        # 处理图片输入
        if has_images and image_b64_list:
            # 取第一张图片作为参考
            mime, b64 = image_b64_list[0]
            body["image"] = f"data:{mime};base64,{b64}"

        # 处理尺寸参数
        image_size = params.get("image_size")
        if isinstance(image_size, str) and image_size.strip() in {"1K", "2K", "4K"}:
            body["image_size"] = image_size.strip()

        # 处理宽高比
        aspect_ratio = params.get("aspect_ratio")
        if isinstance(aspect_ratio, str) and aspect_ratio.strip():
            ar = aspect_ratio.strip()
            if ar.lower() != "default":
                body["aspect_ratio"] = ar

        try:
            req_kwargs = {
                "timeout": self.def_common_config.timeout,
                "proxy": self.def_common_config.proxy,
                "verify": True,
            }
            if (
                isinstance(provider_config.impersonate, str)
                and provider_config.impersonate.strip()
            ):
                req_kwargs["impersonate"] = provider_config.impersonate.strip()

            response = await self.session.post(
                url=url,
                headers=headers,
                json=body,
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
                        f"上游返回HTML，可能是鉴权失败或接口路径错误（{url}）",
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

            result = response.json()
            if isinstance(result, dict):
                # 检查是否有错误
                if result.get("error"):
                    error_msg = result.get("message", str(result.get("error")))
                    return None, 400, f"图片生成失败: {error_msg}"

                # 提取图片 URL
                image_urls: list[str] = []
                # 可能的响应格式
                if "url" in result and isinstance(result["url"], str):
                    image_urls.append(result["url"])
                elif "image_url" in result and isinstance(result["image_url"], str):
                    image_urls.append(result["image_url"])
                elif "data" in result:
                    data = result["data"]
                    if isinstance(data, dict):
                        if "url" in data and isinstance(data["url"], str):
                            image_urls.append(data["url"])
                        elif "image_url" in data and isinstance(data["image_url"], str):
                            image_urls.append(data["image_url"])
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                if "url" in item and isinstance(item["url"], str):
                                    image_urls.append(item["url"])
                                elif "image_url" in item and isinstance(
                                    item["image_url"], str
                                ):
                                    image_urls.append(item["image_url"])

                if image_urls:
                    b64_images = await self.downloader.fetch_media(image_urls)
                    b64_images = [(mime, b64) for mime, b64 in b64_images if b64]
                    if b64_images:
                        return b64_images, 200, None
                    return None, 200, "媒体下载失败"

            return None, 200, "响应中未包含图片数据"

        except Exception as e:
            logger.error(f"[BIG BANANA] RH 请求错误: {e}, url={url}")
            return None, None, f"图片生成失败：{str(e)}"

    async def _call_stream_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """RH 不支持流式响应"""
        return await self._call_api(
            provider_config=provider_config,
            api_key=api_key,
            image_b64_list=image_b64_list,
            params=params,
        )

    def _extract_error_message(self, payload_text: str) -> str | None:
        """从响应中提取错误信息"""
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
        if isinstance(data.get("msg"), str) and data["msg"].strip():
            return data["msg"].strip()
        if isinstance(data.get("error"), str) and data["error"].strip():
            return data["error"].strip()
        err = data.get("error")
        if (
            isinstance(err, dict)
            and isinstance(err.get("message"), str)
            and err["message"].strip()
        ):
            return err["message"].strip()
        return None
