"""
图片生成服务 - 负责图片生成调度与管理
"""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger

from .core import Downloader
from .core.data import CommonConfig, ModelConfig, PromptConfig, ProviderConfig
from .core.utils import clear_cache, read_file, save_images

if TYPE_CHECKING:
    from .main import BigBanana

SUPPORTED_FILE_FORMATS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
)
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_SIZE_B64_LEN = int(MAX_SIZE_BYTES * 4 / 3)


class GenerationService:
    """图片生成服务类"""

    def __init__(self, plugin: "BigBanana"):
        self.plugin = plugin
        self.downloader: Downloader = plugin.downloader
        self.prompt_config: PromptConfig = plugin.prompt_config
        self.common_config: CommonConfig = plugin.common_config
        self.provider_map = plugin.provider_map
        self.models = plugin.models
        self.temp_dir = plugin.temp_dir
        self.save_dir = plugin.save_dir
        self.refer_images_dir = plugin.refer_images_dir

    async def generate_images(
        self,
        event,
        params: dict,
        image_urls: list[str] | None = None,
        referer_id: list[str] | None = None,
        is_llm_tool: bool = False,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        """
        图片生成主方法

        负责参数处理、调度提供商、保存图片等逻辑
        返回：(图片 b64 列表，错误信息)
        """
        if image_urls is None:
            image_urls = []
        if referer_id is None:
            referer_id = []

        # 收集图片 URL
        image_urls = await self._collect_images(
            event, image_urls, referer_id, is_llm_tool
        )

        # 检查图片数量
        min_required_images = params.get("min_images", self.prompt_config.min_images)
        if len(image_urls) < min_required_images:
            warn_msg = f"图片数量不足，最少需要 {min_required_images} 张图片，当前仅 {len(image_urls)} 张"
            logger.warning(warn_msg)
            return None, warn_msg

        # 下载并转换图片为 Base64
        image_b64_list = await self._download_images(image_urls, params)

        # 调用提供商生成图片
        trigger_cmd = str(params.get("__trigger_cmd__") or "").strip()
        result, err_msg = await self._call_provider(
            event, params, image_b64_list, trigger_cmd
        )

        return result, err_msg

    async def _collect_images(
        self,
        event,
        image_urls: list[str],
        referer_id: list[str],
        is_llm_tool: bool,
    ) -> list[str]:
        """收集所有参考图片 URL"""
        import astrbot.api.message_components as Comp

        skipped_at_qq = False
        reply_sender_id = ""

        for comp in event.get_messages():
            if isinstance(comp, Comp.Reply):
                reply_urls, reply_sender_id = await self._collect_reply_image_urls(
                    event, comp
                )
                if reply_urls:
                    image_urls.extend(reply_urls)

            elif (
                isinstance(comp, Comp.At)
                and comp.qq
                and event.platform_meta.name == "aiocqhttp"
            ):
                qq = str(comp.qq)
                self_id = event.get_self_id()
                if not self._should_skip_at(
                    qq,
                    self_id,
                    reply_sender_id,
                    skipped_at_qq,
                    event.is_at_or_wake_command,
                    is_llm_tool,
                ):
                    image_urls.append(f"https://q.qlogo.cn/g?b=qq&s=0&nk={comp.qq}")

            elif isinstance(comp, Comp.Image) and comp.url:
                image_urls.append(comp.url)

            elif (
                isinstance(comp, Comp.File)
                and comp.url
                and comp.url.startswith("http")
                and comp.url.lower().endswith(SUPPORTED_FILE_FORMATS)
            ):
                image_urls.append(comp.url)

        # 处理 q 参数（指定 QQ 号）
        q_value = params.get("q")
        if q_value and event.platform_meta.name == "aiocqhttp":
            import re

            for qq in re.findall(r"\d+", str(q_value)):
                build_url = f"https://q.qlogo.cn/g?b=qq&s=0&nk={qq}"
                if build_url not in image_urls:
                    image_urls.append(build_url)

        # 处理 referer_id 参数
        if is_llm_tool and referer_id and event.platform_meta.name == "aiocqhttp":
            for target_id in referer_id:
                target_id = target_id.strip()
                if target_id:
                    build_url = f"https://q.qlogo.cn/g?b=qq&s=0&nk={target_id}"
                    if build_url not in image_urls:
                        image_urls.append(build_url)

        # 图生图模式自动补充用户头像
        trigger_cmd = str(params.get("__trigger_cmd__") or "").strip()
        is_i2i_mode = trigger_cmd in {"bp1", "bp2", "edit"}
        if (
            is_i2i_mode
            and not image_urls
            and not params.get("refer_images")
            and event.platform_meta.name == "aiocqhttp"
        ):
            image_urls.append(
                f"https://q.qlogo.cn/g?b=qq&s=0&nk={event.get_sender_id()}"
            )

        # 图片去重
        image_urls = list(dict.fromkeys(image_urls))
        return image_urls

    def _should_skip_at(
        self,
        qq: str,
        self_id: str,
        reply_sender_id: str,
        skipped_at_qq: bool,
        is_at_or_wake_command: bool,
        is_llm_tool: bool,
    ) -> bool:
        """判断是否跳过 At 头像获取"""
        from .core.data import PreferenceConfig

        pref: PreferenceConfig = self.plugin.preference_config

        if qq == reply_sender_id and pref.skip_quote_first:
            return not skipped_at_qq
        if qq == self_id and is_at_or_wake_command and pref.skip_at_first:
            return not skipped_at_qq
        if qq == self_id and pref.skip_llm_at_first and is_llm_tool:
            return not skipped_at_qq
        return False

    async def _collect_reply_image_urls(self, event, reply_comp):
        """收集引用消息中的图片 URL"""
        urls = []
        sender_id = ""
        try:
            replied_event = await event.get_replied_event()
            if replied_event:
                sender_id = replied_event.get_sender_id()
                for comp in replied_event.get_messages():
                    if hasattr(comp, "url") and comp.url:
                        urls.append(comp.url)
        except Exception as e:
            logger.debug(f"获取引用消息失败：{e}")
        return urls, sender_id

    async def _download_images(
        self, image_urls: list[str], params: dict
    ) -> list[tuple[str, str]]:
        """下载图片并转换为 Base64"""
        image_b64_list = []
        max_allowed_images = params.get("max_images", self.prompt_config.max_images)

        # 处理 refer_images 参数（本地参考图片）
        refer_images = params.get("refer_images", self.prompt_config.refer_images)
        if refer_images:
            for filename in refer_images.split(","):
                if len(image_b64_list) >= max_allowed_images:
                    break
                filename = filename.strip()
                if filename:
                    path = self.refer_images_dir / filename
                    mime_type, b64_data = await asyncio.to_thread(read_file, path)
                    if b64_data:
                        image_b64_list.append((mime_type, b64_data))

        # 下载网络图片
        append_count = max_allowed_images - len(image_b64_list)
        if append_count > 0 and image_urls:
            download_results = await self.downloader.fetch_images(
                image_urls[:append_count]
            )
            for success, result in download_results:
                if success and isinstance(result, tuple):
                    image_b64_list.append(result)

        return image_b64_list

    async def _call_provider(
        self,
        event,
        params: dict,
        image_b64_list: list[tuple[str, str]],
        trigger_cmd: str,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        """调用提供商生成图片"""
        # 选择模型和提供商
        model_name = params.get("__model_name__")
        if not model_name:
            # 从触发词匹配模型
            for model in self.models:
                if trigger_cmd in model.triggers:
                    model_name = model.name
                    break

        if not model_name:
            return None, "未找到匹配的模型配置"

        # 查找模型配置
        model_config = None
        for m in self.models:
            if m.name == model_name:
                model_config = m
                break

        if not model_config:
            return None, f"未找到模型：{model_name}"

        # 选择提供商（按优先级）
        selected_provider = None
        for p in sorted(model_config.providers, key=lambda x: x.priority):
            if p.enabled and p.api_type in self.provider_map:
                selected_provider = p
                break

        if not selected_provider:
            return None, "没有可用的提供商"

        # 调用提供商生成
        provider_instance = self.provider_map[selected_provider.api_type]
        result, err_msg = await provider_instance.generate_images(
            provider_config=selected_provider,
            params=params,
            image_b64_list=image_b64_list,
        )

        return result, err_msg

    def build_message_chain(self, event, results: list, task_temp_dir: Path):
        """构建响应消息链"""
        import astrbot.api.message_components as Comp

        msg_chain = []
        for b64_data, filename in results:
            # 保存图片到临时目录
            file_path = task_temp_dir / filename
            save_images(b64_data, str(file_path))

            # 根据大小决定作为图片还是文件发送
            file_size = os.path.getsize(file_path)
            if file_size < MAX_SIZE_BYTES:
                with open(file_path, "rb") as f:
                    img_data = f.read()
                msg_chain.append(Comp.Image.frombytes(img_data))
            else:
                msg_chain.append(Comp.File(file=str(file_path)))

        return msg_chain
