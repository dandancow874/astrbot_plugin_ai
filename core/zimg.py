
import base64

from .openai_chat import OpenAIImagesProvider
from .data import ProviderConfig
from astrbot.api import logger
from curl_cffi.requests.exceptions import Timeout

class ZImageProvider(OpenAIImagesProvider):
    api_type: str = "ZImage_Provider"

    @staticmethod
    def _extract_size_wh(size: str) -> tuple[int, int]:
        raw = (size or "").strip().lower()
        if "x" not in raw:
            return 1024, 1024
        a, b = raw.split("x", 1)
        try:
            w = int(a.strip())
            h = int(b.strip())
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return 1024, 1024

    @staticmethod
    def _parse_png_dimensions(data: bytes) -> tuple[int, int] | None:
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        if data[12:16] != b"IHDR":
            return None
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        if w > 0 and h > 0:
            return w, h
        return None

    @staticmethod
    def _parse_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
        if len(data) < 4 or data[0:2] != b"\xFF\xD8":
            return None
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]
            i += 1
            if marker in (0xD8, 0xD9):
                continue
            if i + 1 >= len(data):
                break
            seg_len = int.from_bytes(data[i : i + 2], "big")
            if seg_len < 2 or i + seg_len > len(data):
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if i + 7 >= len(data):
                    break
                h = int.from_bytes(data[i + 3 : i + 5], "big")
                w = int.from_bytes(data[i + 5 : i + 7], "big")
                if w > 0 and h > 0:
                    return w, h
                return None
            i += seg_len
        return None

    @classmethod
    def _infer_b64_dimensions(cls, mime: str, b64: str) -> tuple[int, int] | None:
        if not isinstance(b64, str) or not b64.strip():
            return None
        try:
            raw = base64.b64decode(b64[:200000], validate=False)
        except Exception:
            return None
        mt = (mime or "").lower()
        if "png" in mt:
            return cls._parse_png_dimensions(raw)
        if "jpeg" in mt or "jpg" in mt:
            return cls._parse_jpeg_dimensions(raw)
        dim = cls._parse_png_dimensions(raw)
        if dim:
            return dim
        return cls._parse_jpeg_dimensions(raw)

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
        w, h = self._extract_size_wh(size)
        aspect_ratio = params.get("aspect_ratio")
        want_ar = isinstance(aspect_ratio, str) and aspect_ratio.strip() and aspect_ratio.strip().lower() != "default"

        extra: dict = {}
        for k, v in params.items():
            if k not in [
                "image_size",
                "aspect_ratio",
                "prompt",
                "n",
                "size",
                "width",
                "height",
                "response_format",
                "model",
                "__source_image_urls__",
            ]:
                extra[k] = v

        body_base = {
            "model": params.get("model", provider_config.model),
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        body_wh = {**body_base, "width": w, "height": h, **extra}
        body_size = {**body_base, "size": size, **extra}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Failover-Enabled": "true",
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
            if want_ar:
                logger.info(f"[Z-Image] Request params: ar={aspect_ratio.strip()} size={size} width={w} height={h}")

            async def _send(body: dict):
                r = await self.session.post(
                    url=resolved_url,
                    headers=headers,
                    json=body,
                    **req_kwargs,
                )
                return r, r.json()

            async def _collect_images(res_obj: dict, resp_obj):
                b64_images: list[tuple[str, str]] = []
                images_url: list[str] = []
                for item in (res_obj.get("data") or []):
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
                    return None, f"响应中未包含图片数据: {resp_obj.text[:1024]}"
                return b64_images, None

            response, result = await _send(body_wh)
            
            if response.status_code == 200:
                b64_images, err = await _collect_images(result, response)
                if not b64_images:
                    logger.warning(
                        f"[Z-Image] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                if want_ar:
                    dim = self._infer_b64_dimensions(b64_images[0][0], b64_images[0][1])
                    if dim and dim[0] == dim[1]:
                        response2, result2 = await _send(body_size)
                        if response2.status_code == 200:
                            b64_images2, _ = await _collect_images(result2, response2)
                            if b64_images2:
                                return b64_images2, 200, None
                return b64_images, 200, None

            if (
                response.status_code == 400
                and isinstance(result, dict)
                and isinstance(result.get("error"), dict)
                and isinstance(result["error"].get("message"), str)
                and "参数无效 'size'" in result["error"]["message"]
                and size != "1024x1024"
            ):
                base_hint = params.get("image_size")
                reduced_base = (
                    "1536x1536"
                    if isinstance(base_hint, str) and base_hint.strip().upper() in {"2K", "4K"}
                    else "1024x1024"
                )
                candidates: list[str] = []
                if want_ar and isinstance(aspect_ratio, str) and aspect_ratio.strip():
                    reduced = self._map_image_size(reduced_base, aspect_ratio)
                    if reduced and reduced != size:
                        candidates.append(reduced)
                candidates.append("1024x1024")

                for cand in candidates:
                    w2, h2 = self._extract_size_wh(cand)
                    body_wh2 = {**body_base, "width": w2, "height": h2, **extra}
                    response2, result2 = await _send(body_wh2)
                    if response2.status_code == 200:
                        b64_images2, _ = await _collect_images(result2, response2)
                        if b64_images2:
                            return b64_images2, 200, None

            if (
                response.status_code == 400
                and isinstance(result, dict)
                and isinstance(result.get("error"), dict)
                and isinstance(result["error"].get("message"), str)
                and ("参数无效 'width'" in result["error"]["message"] or "参数无效 'height'" in result["error"]["message"])
            ):
                response2, result2 = await _send(body_size)
                if response2.status_code == 200:
                    b64_images2, _ = await _collect_images(result2, response2)
                    if b64_images2:
                        return b64_images2, 200, None

                base_hint = params.get("image_size")
                reduced_base = (
                    "1536x1536"
                    if isinstance(base_hint, str) and base_hint.strip().upper() in {"2K", "4K"}
                    else "1024x1024"
                )
                if want_ar and isinstance(aspect_ratio, str) and aspect_ratio.strip():
                    reduced = self._map_image_size(reduced_base, aspect_ratio)
                else:
                    reduced = reduced_base
                if reduced and reduced != size:
                    w3, h3 = self._extract_size_wh(reduced)
                    response3, result3 = await _send({**body_base, "width": w3, "height": h3, **extra})
                    if response3.status_code == 200:
                        b64_images3, _ = await _collect_images(result3, response3)
                        if b64_images3:
                            return b64_images3, 200, None
                    response4, result4 = await _send({**body_base, "size": reduced, **extra})
                    if response4.status_code == 200:
                        b64_images4, _ = await _collect_images(result4, response4)
                        if b64_images4:
                            return b64_images4, 200, None

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
            
        if isinstance(aspect_ratio, str):
            aspect_ratio = aspect_ratio.strip("`'\" \t\r\n,，;；。.!！?？)）]】}、")
        if not aspect_ratio or aspect_ratio.lower() == "default":
            return f"{base_w}x{base_h}"

        # 解析宽高比
        try:
            normalized_ar = (
                aspect_ratio.replace("：", ":")
                .replace("／", ":")
                .replace("/", ":")
                .replace("\\", ":")
            )
            ar_w, ar_h = map(float, normalized_ar.replace(":", " ").split())
            target_ratio = ar_w / ar_h
        except Exception:
            return f"{base_w}x{base_h}"

        STANDARD_SIZES = [
            (1024, 1024),
            (1536, 1024),
            (1024, 1536),
            (1536, 1536),
            (1024, 768),
            (768, 1024),
            (1536, 1152),
            (1152, 1536),
            (2048, 2048),
            (2048, 1536),
            (1536, 2048),
            (2048, 1152),
            (1152, 2048),
            (2048, 1024),
            (1024, 2048),
            (2048, 1344),
            (1344, 2048),
            (2048, 1280),
            (1280, 2048),
            (2048, 1664),
            (1664, 2048),
            (2048, 896),
            (896, 2048),
            (4096, 4096),
            (4096, 3072),
            (3072, 4096),
            (4096, 2688),
            (2688, 4096),
            (4096, 2560),
            (2560, 4096),
            (4096, 1792),
            (1792, 4096),
            (4096, 2304),
            (2304, 4096),
        ]

        target_area = base_w * base_h
        best_size = None
        min_score = float("inf")

        for w, h in STANDARD_SIZES:
            ratio = w / h
            # 评分标准：面积差异 + 宽高比差异 (加权)
            area_diff = abs((w * h) - target_area) / target_area
            ratio_diff = abs(ratio - target_ratio)
            
            score = area_diff + ratio_diff * 2
            if score < min_score:
                min_score = score
                best_size = (w, h)

        if best_size:
            return f"{best_size[0]}x{best_size[1]}"

        return "1024x1024"
