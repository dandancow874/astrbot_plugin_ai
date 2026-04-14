import asyncio
import json
from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class RHProvider(BaseProvider):
    """RH 供应商 - 用于 nano-banana 模型"""

    api_type: str = "RH_Provider"

    # RH 基础 URL (正确的 openapi/v2 端点)
    RH_BASE_URL = "https://www.runninghub.cn/openapi/v2"

    # 接口路径映射
    ENDPOINT_MAP = {
        # nano-banana-2 模型
        "nano-banana-2": {
            "text_to_image": "rhart-image-n-g31-flash/text-to-image",
            "image_to_image": "rhart-image-n-g31-flash/image-to-image",
        },
        # nano-banana-pro 模型
        "nano-banana-pro": {
            "text_to_image": "rhart-image-n-pro/text-to-image",
            "image_to_image": "rhart-image-n-pro/edit",
        },
    }

    # 轮询配置
    POLLING_INTERVAL = 5  # 轮询间隔（秒）
    MAX_POLLING_TIME = 180  # 最大轮询时间（秒）

    def _get_endpoint(self, model: str, has_images: bool) -> str:
        """获取对应的接口路径"""
        model_key = model if model in self.ENDPOINT_MAP else "nano-banana-2"
        endpoint_type = "image_to_image" if has_images else "text_to_image"
        path = self.ENDPOINT_MAP[model_key][endpoint_type]
        return f"{self.RH_BASE_URL}/{path}"

    def _normalize_resolution(self, image_size: str | None) -> str:
        """将 image_size 参数转换为 RH API 需要的 resolution 格式"""
        if not image_size:
            return "2k"
        size = image_size.strip().upper()
        if size in {"1K", "2K", "4K"}:
            return size.lower()
        return "2k"

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """发起 RH 图片生成请求（两步流程：提交任务 → 轮询结果）
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
            "apiKey": api_key,
            "prompt": params.get("prompt", "anything"),
        }

        # 处理图片输入（图生图模式）
        if has_images and image_b64_list:
            mime, b64 = image_b64_list[0]
            body["imageUrl"] = f"data:{mime};base64,{b64}"

        # 处理分辨率参数
        image_size = params.get("image_size")
        body["resolution"] = self._normalize_resolution(image_size)

        # 处理宽高比
        aspect_ratio = params.get("aspect_ratio")
        if isinstance(aspect_ratio, str) and aspect_ratio.strip():
            ar = aspect_ratio.strip()
            if ar.lower() != "default":
                body["aspectRatio"] = ar

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

            # ========== 步骤 1: 提交任务 ==========
            logger.info(f"[BIG BANANA] RH 提交任务: {url}")
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
            if not isinstance(result, dict):
                return None, 502, "图片生成失败: 响应格式错误"

            # 检查业务错误
            error_code = result.get("errorCode", "")
            error_msg = result.get("errorMessage", "")
            if error_code or error_msg:
                return None, 400, f"图片生成失败: {error_msg or f'错误码 {error_code}'}"

            # 获取任务 ID
            task_id = result.get("taskId", "")
            if not task_id:
                return None, 502, "图片生成失败: 未获取到任务 ID"

            logger.info(
                f"[BIG BANANA] RH 任务已提交: {task_id}, 状态: {result.get('status', 'UNKNOWN')}"
            )

            # ========== 步骤 2: 轮询任务结果 ==========
            poll_url = f"{self.RH_BASE_URL}/query"
            poll_body = {"taskId": task_id}
            elapsed = 0

            while elapsed < self.MAX_POLLING_TIME:
                await asyncio.sleep(self.POLLING_INTERVAL)
                elapsed += self.POLLING_INTERVAL

                try:
                    poll_response = await self.session.post(
                        url=poll_url,
                        headers=headers,
                        json=poll_body,
                        **req_kwargs,
                    )

                    if poll_response.status_code != 200:
                        logger.warning(
                            f"[BIG BANANA] RH 轮询 HTTP {poll_response.status_code}"
                        )
                        continue

                    poll_result = poll_response.json()
                    if not isinstance(poll_result, dict):
                        continue

                    # 检查轮询错误
                    poll_err_code = poll_result.get("errorCode", "")
                    poll_err_msg = poll_result.get("errorMessage", "")
                    if poll_err_code or poll_err_msg:
                        return (
                            None,
                            400,
                            f"图片生成失败: {poll_err_msg or f'错误码 {poll_err_code}'}",
                        )

                    status = (poll_result.get("status") or "").strip().upper()
                    logger.info(f"[BIG BANANA] RH 轮询 {elapsed}s - 状态: {status}")

                    if status == "SUCCESS":
                        # 提取图片 URL
                        results = poll_result.get("results", [])
                        if not results:
                            return None, 200, "图片生成失败: 响应中未包含结果"

                        image_urls: list[str] = []
                        for item in results:
                            if isinstance(item, dict):
                                url_val = item.get("url") or item.get("outputUrl")
                                if url_val and isinstance(url_val, str):
                                    image_urls.append(url_val)

                        if image_urls:
                            logger.info(
                                f"[BIG BANANA] RH 任务完成: {task_id}, 图片数: {len(image_urls)}"
                            )
                            b64_images = await self.downloader.fetch_media(image_urls)
                            b64_images = [
                                (mime, b64) for mime, b64 in b64_images if b64
                            ]
                            if b64_images:
                                return b64_images, 200, None
                            return None, 200, "媒体下载失败"

                        return None, 200, "响应中未包含图片数据"

                    elif status == "FAILED":
                        return None, 400, f"图片生成失败: 任务执行失败 ({task_id})"

                    elif status == "CANCEL":
                        return None, 400, f"图片生成失败: 任务已取消 ({task_id})"

                    # 继续轮询: CREATE, QUEUED, RUNNING
                except Exception as poll_err:
                    logger.warning(f"[BIG BANANA] RH 轮询异常: {poll_err}")
                    continue

            # 超时
            return None, 408, f"图片生成失败: 任务超时 ({self.MAX_POLLING_TIME}秒)"

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

        # RH API v2 错误格式
        if isinstance(data.get("errorMessage"), str) and data["errorMessage"].strip():
            return data["errorMessage"].strip()
        if isinstance(data.get("errorCode"), str) and data["errorCode"].strip():
            return f"错误码: {data['errorCode'].strip()}"

        # 兼容旧格式
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
