import base64
import mimetypes
from io import BytesIO

from curl_cffi import AsyncSession
from curl_cffi.requests.exceptions import (
    CertificateVerifyError,
    SSLError,
    Timeout,
)
from PIL import Image

from astrbot.api import logger

from .data import CommonConfig


class Downloader:
    def __init__(self, session: AsyncSession, common_config: CommonConfig):
        self.session = session
        self.def_common_config = common_config

    async def fetch_image(self, url: str) -> tuple[str, str] | None:
        """下载单张图片并转换为 (mime, base64)"""
        # 重试逻辑
        for _ in range(3):
            content = await self._download_image(url)
            if content is not None:
                return content

    async def fetch_images(self, image_urls: list[str]) -> list[tuple[str, str] | None]:
        """下载多张图片并转换为 (mime, base64) 列表"""
        image_b64_list = []
        for url in image_urls:
            # 重试逻辑
            for _ in range(3):
                content = await self._download_image(url)
                if content is not None:
                    image_b64_list.append(content)
                    break  # 成功就跳出重试
        return image_b64_list

    async def fetch_media(self, urls: list[str]) -> list[tuple[str, str]]:
        media_b64_list: list[tuple[str, str]] = []
        for url in urls:
            for _ in range(3):
                content = await self._download_media(url)
                if content is not None:
                    media_b64_list.append(content)
                    break
        return media_b64_list

    def _handle_image(self, image_bytes: bytes) -> tuple[str, str]:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                fmt = (img.format or "").upper()
                # 如果不是 GIF，直接返回原图
                if fmt != "GIF":
                    mime = f"image/{fmt.lower()}"
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    return (mime, b64)
                # 处理 GIF
                buf = BytesIO()
                # 取第一帧
                img.seek(0)
                img = img.convert("RGBA")
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return ("image/png", b64)
        except Exception as e:
            logger.warning(f"[BIG BANANA] GIF 处理失败，返回原图: {e}")
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            return ("image/gif", b64)

    async def _download_image(self, url: str) -> tuple[str, str] | None:
        try:
            response = await self.session.get(
                url,
                impersonate="chrome131",
                proxy=self.def_common_config.proxy,
                timeout=30,
            )
            if self._looks_like_html(response):
                logger.warning(f"[BIG BANANA] 下载图片疑似返回HTML，已跳过: {url}")
                return None
            content = self._handle_image(response.content)
            return content
        except (SSLError, CertificateVerifyError):
            response = await self.session.get(
                url, impersonate="chrome131", timeout=30, verify=False
            )
            if self._looks_like_html(response):
                logger.warning(f"[BIG BANANA] 下载图片疑似返回HTML，已跳过: {url}")
                return None
            content = self._handle_image(response.content)
            return content
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {url}，错误信息：{e}")
            return None
        except Exception as e:
            logger.error(f"[BIG BANANA] 下载图片失败: {url}，错误信息：{e}")
            return None

    @staticmethod
    def _looks_like_html(response) -> bool:
        try:
            headers = getattr(response, "headers", None) or {}
            ct = headers.get("Content-Type") or headers.get("content-type") or ""
            if isinstance(ct, str) and ";" in ct:
                ct = ct.split(";", 1)[0].strip()
            ct_lower = ct.lower() if isinstance(ct, str) else ""
            if ct_lower in {"text/html", "application/xhtml+xml"}:
                return True

            content = getattr(response, "content", b"") or b""
            if isinstance(content, (bytes, bytearray)):
                head = bytes(content[:512]).lstrip()
                head_lower = head.lower()
                if head_lower.startswith(b"<!doctype html") or head_lower.startswith(b"<html"):
                    return True
        except Exception:
            return False
        return False

    async def _download_media(self, url: str) -> tuple[str, str] | None:
        try:
            response = await self.session.get(
                url,
                impersonate="chrome131",
                proxy=self.def_common_config.proxy,
                timeout=30,
            )
        except (SSLError, CertificateVerifyError):
            response = await self.session.get(
                url,
                impersonate="chrome131",
                timeout=30,
                proxy=self.def_common_config.proxy,
                verify=False,
            )
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {url}，错误信息：{e}")
            return None
        except Exception as e:
            logger.error(f"[BIG BANANA] 下载媒体失败: {url}，错误信息：{e}")
            return None

        if self._looks_like_html(response):
            logger.warning(f"[BIG BANANA] 下载媒体疑似返回HTML，已跳过: {url}")
            return None

        mime = None
        try:
            headers = getattr(response, "headers", None)
            if headers:
                mime = headers.get("Content-Type") or headers.get("content-type")
        except Exception:
            mime = None
        if isinstance(mime, str) and ";" in mime:
            mime = mime.split(";", 1)[0].strip()
        if not isinstance(mime, str) or not mime.strip():
            guessed, _ = mimetypes.guess_type(url)
            mime = guessed or "application/octet-stream"

        if isinstance(mime, str) and mime.startswith("image/"):
            handled = self._handle_image(response.content)
            return handled

        b64 = base64.b64encode(response.content).decode("utf-8")
        return (mime, b64)
