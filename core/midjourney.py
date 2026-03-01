"""
Midjourney Provider for astrbot_plugin_ai
支持 mj v7 和 nj v7 模型
API 文档参考: https://ai.t8star.cn
"""

import json
import time
import asyncio
from urllib.parse import urlparse

from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class MidjourneyProvider(BaseProvider):
    """Midjourney 提供商 - 支持 mj v7 和 niji v7"""

    api_type: str = "Midjourney_Provider"

    # 任务状态轮询间隔（秒）
    POLL_INTERVAL = 3
    # 任务超时时间（秒）
    TASK_TIMEOUT = 600

    @staticmethod
    def _split_quarter_image(mime: str, b64_data: str) -> list[tuple[str, str]]:
        """
        将四宫格图片裁切成四张单独的图片
        
        参数:
            mime: 图片 MIME 类型
            b64_data: base64 编码的图片数据
            
        返回:
            四张图片的 [(mime, b64), ...] 列表
        """
        import base64
        from io import BytesIO
        
        try:
            # 解码 base64
            img_data = base64.b64decode(b64_data, validate=False)
            
            # 尝试使用 PIL 裁切
            try:
                from PIL import Image
            except ImportError:
                logger.warning("[Midjourney] PIL 未安装，跳过四宫格裁切")
                return [(mime, b64_data)]
            
            # 打开图片
            img = Image.open(BytesIO(img_data))
            width, height = img.size
            
            # 确保是正方形或接近正方形（四宫格通常是 2x2）
            # 计算每个小图的尺寸
            half_w = width // 2
            half_h = height // 2
            
            # 定义四个区域的坐标 (left, upper, right, lower)
            regions = [
                (0, 0, half_w, half_h),           # 左上
                (half_w, 0, width, half_h),       # 右上
                (0, half_h, half_w, height),      # 左下
                (half_w, half_h, width, height),  # 右下
            ]
            
            result: list[tuple[str, str]] = []
            
            for i, region in enumerate(regions):
                try:
                    # 裁切图片
                    cropped = img.crop(region)
                    
                    # 转换为 base64
                    buffer = BytesIO()
                    # 保持原始格式
                    save_format = "PNG" if "png" in mime.lower() else "JPEG"
                    if save_format == "JPEG" and cropped.mode in ("RGBA", "P"):
                        cropped = cropped.convert("RGB")
                    cropped.save(buffer, format=save_format, quality=95)
                    cropped_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    cropped_mime = "image/png" if save_format == "PNG" else "image/jpeg"
                    result.append((cropped_mime, cropped_b64))
                    
                except Exception as e:
                    logger.warning(f"[Midjourney] 裁切第 {i+1} 张图片失败: {e}")
                    continue
            
            if result:
                logger.info(f"[Midjourney] 四宫格裁切完成，共 {len(result)} 张图片")
                return result
            
            # 裁切失败，返回原图
            return [(mime, b64_data)]
            
        except Exception as e:
            logger.error(f"[Midjourney] 图片裁切失败: {e}")
            return [(mime, b64_data)]

    @staticmethod
    def _is_midjourney_url(api_url: object) -> bool:
        """检查是否为 midjourney 相关的 URL"""
        if not isinstance(api_url, str):
            return False
        u = api_url.lower()
        return "t8star" in u or "midjourney" in u or "mj" in u

    @staticmethod
    def _resolve_mj_api_url(api_url: str, model: str) -> str:
        """
        解析并构建 Midjourney API URL

        T8star API 格式:
        - mj: https://ai.t8star.cn/mj/submit/imagine
        - niji: https://ai.t8star.cn/mj/submit/imagine (使用 --niji 参数)
        - 查询任务: https://ai.t8star.cn/mj/task/{task_id}/fetch
        """
        raw = (api_url or "").strip().rstrip("/")
        lowered = raw.lower()

        # 如果已经有完整的路径，直接使用
        if "/mj/submit" in lowered or "/submit/imagine" in lowered:
            return raw

        # 否则构建默认路径
        base = raw
        if base.endswith("/v1"):
            base = base[:-3]
        if base.endswith("/mj"):
            base = base[:-3]

        # 默认使用 imagine 端点
        return f"{base}/mj/submit/imagine"

    @staticmethod
    def _get_fetch_url(api_url: str, task_id: str) -> str:
        """构建任务查询 URL"""
        raw = (api_url or "").strip().rstrip("/")
        lowered = raw.lower()

        # 提取基础 URL
        if "/mj/submit" in lowered:
            base = raw.split("/mj/submit", 1)[0]
        elif "/submit" in lowered:
            base = raw.split("/submit", 1)[0]
        elif "/mj" in lowered:
            base = raw.split("/mj", 1)[0]
        else:
            base = raw

        return f"{base}/mj/task/{task_id}/fetch"

    def _build_mj_payload(self, prompt: str, model_type: str, params: dict) -> dict:
        """
        构建 Midjourney 请求 payload

        model_type: "mj" 或 "nj" (niji)
        """
        # 基础 prompt
        final_prompt = prompt.strip() if prompt else "a beautiful image"

        # 根据模型类型添加参数
        # mj 使用 "V 7"，nj 使用 "niji 7"
        if model_type == "nj":
            # niji 模型
            if "--niji" not in final_prompt.lower():
                final_prompt = f"{final_prompt} --niji 7"
            else:
                # 替换 --niji 版本
                import re
                final_prompt = re.sub(
                    r"--niji\s*\d*", "--niji 7", final_prompt, flags=re.I
                )
        else:
            # mj v7 模型 (默认)
            if "--v" not in final_prompt.lower():
                final_prompt = f"{final_prompt} --v 7"
            else:
                # 替换版本号
                import re
                final_prompt = re.sub(r"--v\s*\d*", "--v 7", final_prompt, flags=re.I)

        # 处理宽高比参数
        aspect_ratio = params.get("aspect_ratio")
        if (
            isinstance(aspect_ratio, str)
            and aspect_ratio.strip()
            and aspect_ratio.lower() != "default"
        ):
            ar = (
                aspect_ratio.strip()
                .replace("：", ":")
                .replace("／", ":")
                .replace("/", ":")
                .replace("\\", ":")
            )
            if "--ar" not in final_prompt.lower():
                final_prompt = f"{final_prompt} --ar {ar}"

        payload = {
            "prompt": final_prompt,
            "base64Array": [],  # 垫图支持
            "notifyHook": "",
            "state": "",
        }

        return payload

    async def _poll_task_result(
        self,
        task_id: str,
        api_url: str,
        api_key: str,
        provider_config: ProviderConfig,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        """
        轮询任务结果

        返回: (图片列表, 错误信息)
        """
        fetch_url = self._get_fetch_url(api_url, task_id)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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
                "timeout": 30,  # 单次查询超时
                "proxy": self.def_common_config.proxy,
                "verify": verify,
            }
            if impersonate:
                req_kwargs["impersonate"] = impersonate

            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.TASK_TIMEOUT:
                    return None, f"任务超时（{self.TASK_TIMEOUT}秒）"

                try:
                    response = await self.session.get(
                        url=fetch_url,
                        headers=headers,
                        **req_kwargs,
                    )

                    if response.status_code != 200:
                        text = getattr(response, "text", "")[:1024]
                        return (
                            None,
                            f"查询任务失败: 状态码 {response.status_code}, {text}",
                        )

                    result = response.json()

                    if not isinstance(result, dict):
                        await asyncio.sleep(self.POLL_INTERVAL)
                        continue

                    status = result.get("status", "").lower()

                    # 任务成功
                    if status == "success":
                        images: list[tuple[str, str]] = []

                        # 获取图片 URL
                        image_url = result.get("imageUrl") or result.get("image_url")
                        if isinstance(image_url, str) and image_url.strip():
                            # 下载图片并转换为 base64
                            downloaded = await self.downloader.fetch_images(
                                [image_url.strip()]
                            )
                            images.extend(downloaded)

                        # 处理多图情况（如四宫格）
                        buttons = result.get("buttons", [])
                        if isinstance(buttons, list):
                            for btn in buttons:
                                if isinstance(btn, dict):
                                    btn_url = btn.get("url") or btn.get("imageUrl")
                                    if isinstance(btn_url, str) and btn_url.strip():
                                        downloaded = await self.downloader.fetch_images(
                                            [btn_url.strip()]
                                        )
                                        images.extend(downloaded)

                        # 检查是否有图片数据
                        images = [(mime, b64) for mime, b64 in images if b64]
                        if images:
                            # 将四宫格图片裁切成四张单独的图片
                            split_images: list[tuple[str, str]] = []
                            for mime, b64 in images:
                                split_result = self._split_quarter_image(mime, b64)
                                split_images.extend(split_result)
                            if split_images:
                                logger.info(f"[Midjourney] 返回 {len(split_images)} 张图片")
                                return split_images, None
                            return images, None

                        return None, "任务成功但未获取到图片数据"

                    # 任务失败
                    if status in ("failure", "failed", "error"):
                        fail_reason = (
                            result.get("failReason")
                            or result.get("error")
                            or result.get("message")
                            or "未知原因"
                        )
                        return None, f"任务失败: {fail_reason}"

                    # 任务取消
                    if status == "cancelled":
                        return None, "任务已取消"

                    # 任务进行中，继续轮询
                    if status in ("submitted", "in_progress", "pending", "not_start"):
                        progress = result.get("progress", "")
                        if progress:
                            logger.debug(
                                f"[Midjourney] 任务 {task_id} 进度: {progress}"
                            )
                        await asyncio.sleep(self.POLL_INTERVAL)
                        continue

                    # 未知状态，继续轮询
                    logger.debug(f"[Midjourney] 任务 {task_id} 状态: {status}")
                    await asyncio.sleep(self.POLL_INTERVAL)

                except Exception as e:
                    logger.warning(f"[Midjourney] 轮询任务异常: {e}")
                    await asyncio.sleep(self.POLL_INTERVAL)

        except Exception as e:
            logger.error(f"[Midjourney] 轮询任务错误: {e}")
            return None, f"轮询任务错误: {e}"

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """
        发起 Midjourney 图片生成请求

        返回值: 元组(图片 base64 列表, 状态码, 人类可读的错误信息)
        """
        # 确定模型类型：mj 或 nj
        model = str(params.get("model", provider_config.model or "")).strip().lower()
        model_type = "mj"  # 默认为 mj

        if (
            model.startswith("nj")
            or model.startswith("niji")
            or "niji" in model
            or model == "niji 7"
        ):
            model_type = "nj"

        # 构建 API URL
        resolved_url = self._resolve_mj_api_url(provider_config.api_url, model)

        # 构建 prompt
        prompt = params.get("prompt", "a beautiful image")

        # 构建请求 payload
        payload = self._build_mj_payload(prompt, model_type, params)

        # 如果有输入图片，添加到 base64Array（垫图功能）
        if image_b64_list:
            for mime, b64 in image_b64_list[:5]:  # 最多支持 5 张垫图
                if b64:
                    payload["base64Array"].append(f"data:{mime};base64,{b64}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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
                "timeout": 60,  # 提交任务超时
                "proxy": self.def_common_config.proxy,
                "verify": verify,
            }
            if impersonate:
                req_kwargs["impersonate"] = impersonate

            # 提交任务
            logger.info(f"[Midjourney] 提交任务: {model_type} - {prompt[:100]}...")
            response = await self.session.post(
                url=resolved_url,
                headers=headers,
                json=payload,
                **req_kwargs,
            )

            resp_text = getattr(response, "text", "")

            if response.status_code != 200 and response.status_code != 201:
                logger.error(
                    f"[Midjourney] 提交任务失败，状态码: {response.status_code}, 响应: {resp_text[:1024]}"
                )
                # 尝试解析错误信息
                try:
                    err_data = (
                        json.loads(resp_text) if isinstance(resp_text, str) else {}
                    )
                    err_msg = (
                        err_data.get("error")
                        or err_data.get("message")
                        or err_data.get("msg")
                        or resp_text[:200]
                    )
                except Exception:
                    err_msg = (
                        resp_text[:200] if isinstance(resp_text, str) else "未知错误"
                    )
                return None, response.status_code, f"提交任务失败: {err_msg}"

            # 解析响应获取任务 ID
            try:
                result = response.json()
            except Exception as e:
                logger.error(
                    f"[Midjourney] 解析响应失败: {e}, 响应: {resp_text[:1024]}"
                )
                return None, 500, f"解析响应失败: {resp_text[:200]}"

            if not isinstance(result, dict):
                return None, 500, f"响应格式错误: {resp_text[:200]}"

            # 获取任务 ID
            task_id = (
                result.get("taskId")
                or result.get("task_id")
                or result.get("id")
                or result.get("data", {}).get("taskId")
            )

            if not task_id or not isinstance(task_id, str):
                # 可能直接返回了结果（同步模式）
                image_url = (
                    result.get("imageUrl")
                    or result.get("image_url")
                    or result.get("url")
                )
                if image_url:
                    downloaded = await self.downloader.fetch_images([image_url])
                    downloaded = [(m, b) for m, b in downloaded if b]
                    if downloaded:
                        return downloaded, 200, None

                err_msg = (
                    result.get("error") or result.get("message") or result.get("msg")
                )
                if err_msg:
                    return None, 400, f"提交任务失败: {err_msg}"

                return None, 500, f"未获取到任务 ID: {resp_text[:200]}"

            logger.info(f"[Midjourney] 任务已提交: {task_id}")

            # 轮询任务结果
            images, err = await self._poll_task_result(
                task_id=task_id,
                api_url=resolved_url,
                api_key=api_key,
                provider_config=provider_config,
            )

            if images:
                return images, 200, None

            return None, 500, err or "任务执行失败"

        except Timeout as e:
            logger.error(f"[Midjourney] 网络请求超时: {e}")
            return None, 408, f"网络请求超时: {e}"
        except Exception as e:
            logger.error(f"[Midjourney] 请求错误: {e}")
            return None, 500, f"请求错误: {e}"

    async def _call_stream_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """
        Midjourney 不支持流式 API，直接调用同步方法
        """
        return await self._call_api(
            provider_config=provider_config,
            api_key=api_key,
            image_b64_list=image_b64_list,
            params=params,
        )
