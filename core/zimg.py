
from .openai_chat import OpenAIImagesProvider
from .data import ProviderConfig
from astrbot.api import logger
from curl_cffi.requests.exceptions import Timeout

class ZImageProvider(OpenAIImagesProvider):
    api_type: str = "ZImage_Provider"

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """
        Z-Image-Turbo 专用调用逻辑
        不发送 aspect_ratio 字段，完全依赖 size 字段
        """
        # Z-Image 不支持图生图 (edits)，忽略 image_b64_list
        # 清理 URL: 去除首尾空格、反引号、逗号、单双引号
        raw_url = (provider_config.api_url or "").strip(" \t\n\r`'\",")
        resolved_url = raw_url.rstrip("/")
        
        if not resolved_url:
            resolved_url = "https://ai.gitee.com/v1/images/generations"
        elif "/images/generations" not in resolved_url:
            # 如果是 /v1 结尾，追加 /images/generations
            if resolved_url.endswith("/v1"):
                resolved_url += "/images/generations"
            # 如果包含 /chat/completions (用户可能复制了对话接口)，替换为 /images/generations
            elif "/chat/completions" in resolved_url:
                resolved_url = resolved_url.replace("/chat/completions", "/images/generations")
            # 否则假设用户只填了 Base URL (如 https://ai.gitee.com/v1)，尝试追加
            else:
                resolved_url += "/images/generations"

        prompt = params.get("prompt", "anything")
        
        # 使用自定义的 _map_image_size 计算最终分辨率
        size = self._map_image_size(params.get("image_size"), params.get("aspect_ratio"))
        if not size:
            size = "1024x1024"
        
        logger.info(f"[Z-Image] Calculated resolution: {size} (from input: size={params.get('image_size')}, ar={params.get('aspect_ratio')})")

        body = {
            "model": params.get("model", provider_config.model),
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        
        # 允许用户通过 params 传递 extra_body (如 num_inference_steps)
        # 但过滤掉 aspect_ratio 和其他已处理字段
        for k, v in params.items():
            if k not in ["image_size", "aspect_ratio", "prompt", "n", "size", "response_format", "model", "__source_image_urls__"]:
                body[k] = v

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Failover-Enabled": "true", # 用户建议添加
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
                        f"[Z-Image] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                return b64_images, 200, None

            logger.error(
                f"[Z-Image] 图片生成失败，状态码: {response.status_code}, 响应内容: {response.text[:1024]}"
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
            logger.error(f"[Z-Image] 网络请求超时: {e}")
            return None, None, "网络请求超时"
        except Exception as e:
            logger.error(f"[Z-Image] 未知错误: {e}")
            return None, None, f"未知错误: {e}"

    @staticmethod
    def _map_image_size(image_size: object, aspect_ratio: str | None = None) -> str | None:
        """
        Gitee AI / Z-Image-Turbo 专用分辨率映射
        """
        # 如果没有指定 image_size，默认为 "1024x1024" (1K) 以便进行 AR 计算
        if not image_size or not isinstance(image_size, str):
            image_size = "1024x1024"
            
        size = image_size.strip().upper()
        
        # 基础分辨率映射 (1:1)
        base_w, base_h = 1024, 1024
        if size == "1K":
            base_w, base_h = 1024, 1024
        elif size == "2K":
            base_w, base_h = 2048, 2048
        elif size == "4K":
            base_w, base_h = 4096, 4096
        elif "X" in size and all(p.isdigit() for p in size.split("X", 1)):
            parts = size.split("X", 1)
            base_w, base_h = int(parts[0]), int(parts[1])
        else:
            base_w, base_h = 1024, 1024
            
        if not aspect_ratio or aspect_ratio.lower() == "default":
            return f"{base_w}x{base_h}"

        # 解析宽高比
        try:
            ar_w, ar_h = map(float, aspect_ratio.replace(":", " ").split())
            target_ratio = ar_w / ar_h
        except Exception:
            return f"{base_w}x{base_h}"

        # Gitee AI / Z-Image-Turbo 专用常用分辨率 (基于 SDXL 最佳实践)
        STANDARD_SIZES = [
            # 1:1
            (1024, 1024), (768, 768), (512, 512),
            (1440, 1440), # 2MP 1:1

            # 16:9 / 9:16
            (1344, 768), (768, 1344),   # SDXL 16:9 标准
            (1920, 1080), (1080, 1920), # FHD (部分支持)
            
            # 4:3 / 3:4
            (1152, 896), (896, 1152),   # SDXL 4:3 标准
            
            # 3:2 / 2:3
            (1216, 832), (832, 1216),   # SDXL 3:2 标准
            
            # 21:9 / 9:21
            (1536, 640), (640, 1536),   # SDXL 21:9 标准
        ]

        target_area = base_w * base_h
        best_size = None
        min_score = float("inf")

        for w, h in STANDARD_SIZES:
            ratio = w / h
            # 允许 5% 的宽高比误差
            if abs(ratio - target_ratio) / target_ratio > 0.05:
                continue
            
            # 评分标准：面积差异 + 宽高比差异 (加权)
            area_diff = abs((w * h) - target_area) / target_area
            ratio_diff = abs(ratio - target_ratio)
            
            score = area_diff + ratio_diff * 2
            if score < min_score:
                min_score = score
                best_size = (w, h)

        if best_size:
            return f"{best_size[0]}x{best_size[1]}"

        # 兜底计算 (确保 64 对齐)
        import math
        new_h = math.sqrt(target_area / target_ratio)
        new_w = new_h * target_ratio
        
        final_w = int(round(new_w / 64) * 64)
        final_h = int(round(new_h / 64) * 64)
        
        # 再次检查是否过大 (Z-Image 限制 4MP，建议 2048 以下)
        if final_w > 2048: final_w = 2048
        if final_h > 2048: final_h = 2048
        
        return f"{final_w}x{final_h}"
