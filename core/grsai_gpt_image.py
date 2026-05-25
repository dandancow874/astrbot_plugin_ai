import json
import asyncio
import time
import base64
import math
from io import BytesIO

try:
    from curl_cffi import CurlHttpVersion
except Exception:
    CurlHttpVersion = None
from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class GrsaiGPTImageProvider(BaseProvider):
    """Grsai GPT Image 提供商"""

    api_type: str = "Grsai_GPT_Image"

    SUPPORTED_RATIOS: tuple[tuple[str, float], ...] = (
        ("9:16", 9 / 16),
        ("2:3", 2 / 3),
        ("3:4", 3 / 4),
        ("1:1", 1),
        ("3:2", 3 / 2),
        ("8:5", 8 / 5),
        ("5:3", 5 / 3),
        ("16:9", 16 / 9),
        ("2:1", 2),
        ("2.44:1", 2.44),
        ("3:1", 3),
        ("4:1", 4),
    )

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

    @staticmethod
    def _normalize_ratio(value: str) -> str:
        return (
            value.strip("`'\" \t\r\n,，;；。.!！?？)）]】}、")
            .replace("：", ":")
            .replace("／", ":")
            .replace("/", ":")
            .replace("\\", ":")
        )

    @classmethod
    def _nearest_supported_ratio(cls, width: int, height: int) -> str | None:
        if width <= 0 or height <= 0:
            return None
        actual = width / height
        # 相对误差比绝对误差更适合比较横图、竖图和超宽比例。
        return min(
            cls.SUPPORTED_RATIOS,
            key=lambda item: abs(math.log(actual / item[1])),
        )[0]

    @staticmethod
    def _infer_b64_dimensions(b64: str) -> tuple[int, int] | None:
        if not isinstance(b64, str) or not b64.strip():
            return None
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            return None
        try:
            from PIL import Image

            with Image.open(BytesIO(raw)) as img:
                width, height = img.size
        except Exception:
            return None
        if width > 0 and height > 0:
            return width, height
        return None

    @classmethod
    def _resolve_payload_size(
        cls,
        aspect_ratio: object,
        image_b64_list: list[tuple[str, str]],
    ) -> str:
        if isinstance(aspect_ratio, str) and aspect_ratio.strip():
            ar = cls._normalize_ratio(aspect_ratio)
            if ar.lower() == "default":
                return "1:1" if image_b64_list else "9:16"
            if ar.lower() == "auto":
                for _, b64 in image_b64_list:
                    dim = cls._infer_b64_dimensions(b64)
                    if dim:
                        resolved = cls._nearest_supported_ratio(dim[0], dim[1])
                        if resolved:
                            logger.info(
                                f"[GPT Image] auto size: source={dim[0]}x{dim[1]} -> {resolved}"
                            )
                            return resolved
                return "1:1" if image_b64_list else "9:16"
            return ar
        return "1:1" if image_b64_list else "9:16"

    @staticmethod
    def _build_data_urls(image_b64_list: list[tuple[str, str]]) -> list[str]:
        data_urls: list[str] = []
        for mime, b64 in image_b64_list:
            if not b64:
                continue
            mime_type = (mime or "image/png").strip() or "image/png"
            data_urls.append(f"data:{mime_type};base64,{b64}")
        return data_urls

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
        image_data_urls = self._build_data_urls(image_b64_list)

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "size": self._resolve_payload_size(
                params.get("aspect_ratio"), image_b64_list
            ),
        }
        quality = params.get("quality")
        if isinstance(quality, str) and quality.strip().lower() in {
            "auto",
            "high",
            "medium",
            "low",
        }:
            payload["quality"] = quality.strip().lower()
        logger.info(
            f"[GPT Image] request size={payload['size']}, quality={payload.get('quality', 'auto')}, aspect_ratio={params.get('aspect_ratio')}, reference_images={len(image_data_urls)}"
        )
        if image_data_urls:
            payload["urls"] = image_data_urls
        elif urls:
            payload["urls"] = urls

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
            if CurlHttpVersion is not None:
                req_kwargs["http_version"] = CurlHttpVersion.V1_1

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
            err_text = str(e)
            logger.error(f"[GPT Image] 请求错误: {err_text}, url={draw_url}")
            if (
                "curl: (35)" in err_text
                or "connection reset" in err_text.lower()
                or "recv failure" in err_text.lower()
                or "tls connect error" in err_text.lower()
            ):
                return None, 503, "图片生成失败：上游连接被重置，请稍后重试"
            return None, 500, f"图片生成失败：请求错误: {err_text}"

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


class GrsaiGPTImageVIPProvider(GrsaiGPTImageProvider):
    """Grsai GPT Image VIP 提供商（/v1/api/generate）。"""

    api_type: str = "Grsai_GPT_Image_VIP"

    @staticmethod
    def _resolve_generate_url(api_url: str) -> str:
        raw = (api_url or "").strip().rstrip("/")
        lowered = raw.lower()
        root = raw
        if "/v1/" in lowered:
            root = raw.split("/v1/", 1)[0]
        elif lowered.endswith("/v1"):
            root = raw[: -len("/v1")]
        return root.rstrip("/") + "/v1/api/generate"

    VIP_RATIO_SIZE_MAP: dict[str, dict[str, str]] = {
        "1:1": {"1K": "1024x1024", "2K": "2048x2048", "4K": "2880x2880"},
        "16:9": {"1K": "1280x720", "2K": "2048x1152", "4K": "3840x2160"},
        "9:16": {"1K": "720x1280", "2K": "1152x2048", "4K": "2160x3840"},
        "4:3": {"1K": "1152x864", "2K": "2304x1728", "4K": "3264x2448"},
        "3:4": {"1K": "864x1152", "2K": "1728x2304", "4K": "2448x3264"},
        "3:2": {"1K": "1536x1024", "2K": "2048x1360", "4K": "3504x2336"},
        "2:3": {"1K": "1024x1536", "2K": "1360x2048", "4K": "2336x3504"},
        "5:4": {"1K": "1120x896", "2K": "2240x1792", "4K": "3200x2560"},
        "4:5": {"1K": "896x1120", "2K": "1792x2240", "4K": "2560x3200"},
        "21:9": {"1K": "1456x624", "2K": "2912x1248", "4K": "3840x1648"},
        "9:21": {"1K": "624x1456", "2K": "1248x2912", "4K": "1648x3840"},
        "1:3": {"2K": "688x2048", "4K": "1280x3840"},
        "3:1": {"2K": "2048x688", "4K": "3840x1280"},
        "2:1": {"1K": "1536x768", "2K": "3072x1536", "4K": "3840x1920"},
        "1:2": {"1K": "768x1536", "2K": "1536x3072", "4K": "1920x3840"},
    }

    @staticmethod
    def _parse_ratio_value(ratio_text: str) -> float | None:
        try:
            left, right = ratio_text.split(":", 1)
            w = float(left)
            h = float(right)
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return w / h

    @staticmethod
    def _round_to_16(value: float) -> int:
        return max(16, int(round(value / 16) * 16))

    @classmethod
    def _fit_vip_constraints(cls, ratio: float, target_area: int) -> str:
        ratio = max(1 / 3, min(3, ratio))
        max_area = 8_294_400
        min_area = 655_360
        area = min(max(target_area, min_area), max_area)
        width = math.sqrt(area * ratio)
        height = math.sqrt(area / ratio)

        if width > 3840:
            width = 3840
            height = width / ratio
        if height > 3840:
            height = 3840
            width = height * ratio

        w = cls._round_to_16(width)
        h = cls._round_to_16(height)
        w = min(3840, max(16, w - (w % 16)))
        h = min(3840, max(16, h - (h % 16)))

        while w * h > max_area and w > 16 and h > 16:
            if w >= h:
                w -= 16
                h = cls._round_to_16(w / ratio)
            else:
                h -= 16
                w = cls._round_to_16(h * ratio)
            w = min(3840, max(16, w - (w % 16)))
            h = min(3840, max(16, h - (h % 16)))

        return f"{w}x{h}"

    @classmethod
    def _resolve_vip_aspect_ratio(
        cls,
        aspect_ratio: object,
        image_size: object,
        image_b64_list: list[tuple[str, str]],
    ) -> str:
        ratio_text = GrsaiGPTImageProvider._resolve_payload_size(
            aspect_ratio, image_b64_list
        )
        size_text = str(image_size or "1K").strip().upper()
        if size_text not in {"1K", "2K", "4K"}:
            size_text = "1K"

        fixed = cls.VIP_RATIO_SIZE_MAP.get(ratio_text, {}).get(size_text)
        if fixed:
            return fixed

        ratio = cls._parse_ratio_value(ratio_text)
        if ratio is None:
            return cls.VIP_RATIO_SIZE_MAP["1:1"][size_text]
        target_area = {
            "1K": 1_048_576,
            "2K": 4_194_304,
            "4K": 8_294_400,
        }[size_text]
        return cls._fit_vip_constraints(ratio, target_area)

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
        model = (params.get("model") or provider_config.model or "gpt-image-2-vip").strip()
        prompt = params.get("prompt", "anything")
        urls = params.get("__source_image_urls__")
        if not isinstance(urls, list):
            urls = []
        urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        image_data_urls = self._build_data_urls(image_b64_list)
        images = image_data_urls or urls

        aspect_ratio = self._resolve_vip_aspect_ratio(
            params.get("aspect_ratio"), params.get("image_size"), image_b64_list
        )
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "images": images,
            "aspectRatio": aspect_ratio,
            "replyType": "json",
        }
        logger.info(
            f"[GPT Image VIP] request aspectRatio={aspect_ratio}, image_size={params.get('image_size')}, reference_images={len(images)}"
        )

        generate_url = self._resolve_generate_url(provider_config.api_url)
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
            if CurlHttpVersion is not None:
                req_kwargs["http_version"] = CurlHttpVersion.V1_1

            response = await self.session.post(
                url=generate_url,
                headers=headers,
                json=payload,
                **req_kwargs,
            )
            resp_text = getattr(response, "text", "")
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

            try:
                data = json.loads(resp_text) if isinstance(resp_text, str) else {}
            except Exception:
                data = {}
            if not isinstance(data, dict):
                return None, 200, "响应内容格式错误"

            if data.get("status") not in (None, "succeeded"):
                reason = data.get("failure_reason") or data.get("message") or data.get("status")
                return None, 400, f"图片生成失败: {reason}"

            result_url_list: list[str] = []
            results = data.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        u = item.get("url")
                        if isinstance(u, str) and u.strip():
                            result_url_list.append(u.strip())
            url = data.get("url")
            if isinstance(url, str) and url.strip():
                result_url_list.append(url.strip())

            if not result_url_list:
                return None, 200, "响应中未包含图片数据"

            result_url_list = list(dict.fromkeys(result_url_list))
            b64_images = await self.downloader.fetch_media(result_url_list)
            b64_images = [(mime, b64) for mime, b64 in b64_images if b64]
            if not b64_images:
                return None, 200, "图片下载失败"
            return b64_images, 200, None
        except Timeout as e:
            logger.error(f"[GPT Image VIP] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except Exception as e:
            err_text = str(e)
            logger.error(f"[GPT Image VIP] 请求错误: {err_text}, url={generate_url}")
            return None, 500, f"图片生成失败：请求错误: {err_text}"
