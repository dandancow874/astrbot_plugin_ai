import random
from abc import ABC, abstractmethod
from typing import ClassVar

from aiohttp import ClientSession
from curl_cffi import AsyncSession

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig

from .data import CommonConfig, PromptConfig, ProviderConfig
from .downloader import Downloader
from .utils import get_key_index


class BaseProvider(ABC):
    """提供商抽象基类"""

    api_type: str
    """提供商 API 类型标识符"""

    _registry: ClassVar[dict[str, type["BaseProvider"]]] = {}
    """提供商类注册表"""

    # 全局Key轮询索引记录
    _rotation_index_map: ClassVar[dict[str, int]] = {}

    session: AsyncSession
    aiohttp_session: ClientSession | None
    def_common_config: CommonConfig
    def_prompt_config: PromptConfig
    downloader: Downloader

    # 可重试状态码
    RETRY_STATUS_CODES = frozenset({408, 500, 502, 503, 504})
    # 不可重试状态码
    NO_RETRY_STATUS_CODES = frozenset({401, 402, 403, 422, 429})
    # 内容审核拦截关键词（检测到则跳过重试）
    CONTENT_BLOCK_KEYWORDS = frozenset(
        {"blocked", "content_policy", "moderation", "safety"}
    )

    def __init__(
        self,
        config: AstrBotConfig,
        common_config: CommonConfig,
        prompt_config: PromptConfig,
        session: AsyncSession,
        downloader: Downloader,
        aiohttp_session: ClientSession | None = None,
    ):
        self.conf = config
        self.def_prompt_config = prompt_config
        self.def_common_config = common_config
        self.session = session
        self.downloader = downloader
        self.aiohttp_session = aiohttp_session

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "api_type")
            and cls.api_type
            and not cls._registry.get(cls.api_type)
        ):
            cls._registry[cls.api_type] = cls
            logger.debug(f"已注册提供商类: {cls.api_type}")

    @classmethod
    def get_provider_class(cls, api_type: str) -> type["BaseProvider"] | None:
        return cls._registry.get(api_type, None)

    @classmethod
    def get_next_rotation_index(cls, provider_name: str, key_count: int) -> int:
        """获取下一个轮询索引，实现真正的交替使用"""
        if provider_name not in cls._rotation_index_map:
            cls._rotation_index_map[provider_name] = 0

        current_index = cls._rotation_index_map[provider_name]
        next_index = (current_index + 1) % key_count
        cls._rotation_index_map[provider_name] = next_index

        return current_index

    async def generate_images(
        self,
        provider_config: ProviderConfig,
        params: dict,
        image_b64_list: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        """图片生成调度方法 - 实现真正的交替轮询"""
        key_list_len = len(provider_config.keys)
        if key_list_len == 0:
            return None, "图片生成失败：未配置 API Key"

        # 获取本次应该使用的Key索引（真正的轮询交替）
        current_index = self.get_next_rotation_index(provider_config.name, key_list_len)
        api_key = provider_config.keys[current_index]

        logger.info(
            f"[{self.api_type}] 轮询使用 {provider_config.name} 的第 {current_index + 1}/{key_list_len} 个API Key"
        )

        # 只重试当前Key，不切换到其他Key（实现真正的交替）
        for i in range(self.def_common_config.max_retry):
            if provider_config.stream:
                images_result, status, err = await self._call_stream_api(
                    provider_config=provider_config,
                    api_key=api_key,
                    params=params,
                    image_b64_list=image_b64_list,
                )
            else:
                images_result, status, err = await self._call_api(
                    provider_config=provider_config,
                    api_key=api_key,
                    params=params,
                    image_b64_list=image_b64_list,
                )

            if images_result:
                return images_result, None

            # 内容审核拦截，跳过重试
            if self.is_content_blocked(err):
                logger.warning(f"[{self.api_type}] 内容审核拦截，跳过重试: {err}")
                return None, err

            if status == 404 and isinstance(err, str) and err.strip():
                api_url = (provider_config.api_url or "").strip()
                if api_url and api_url not in err:
                    err = f"{err.strip()}（{api_url}）"

            if self.def_common_config.smart_retry and not self.should_retry(status):
                break

            logger.warning(
                f"[{self.api_type}] API Key ({current_index + 1}/{key_list_len}) 重试 ({i + 1}/{self.def_common_config.max_retry}): {err}"
            )

        return (
            None,
            err
            or f"图片生成失败：API Key ({current_index + 1}/{key_list_len}) 重试 {self.def_common_config.max_retry} 次后仍失败",
        )

    def should_retry(self, status) -> bool:
        if status in self.RETRY_STATUS_CODES:
            return True
        return False

    def is_content_blocked(self, err: str | None) -> bool:
        """检测错误消息是否为内容审核拦截"""
        if not isinstance(err, str) or not err.strip():
            return False
        err_lower = err.lower()
        return any(keyword in err_lower for keyword in self.CONTENT_BLOCK_KEYWORDS)

    @abstractmethod
    async def _call_api(
        self, **kwargs
    ) -> tuple[list[tuple[str, str]], int | None, str | None]:
        """调用同步 API 方法"""
        pass

    @abstractmethod
    async def _call_stream_api(
        self, **kwargs
    ) -> tuple[list[tuple[str, str]], int | None, str | None]:
        """调用流式 API 方法"""
        pass
