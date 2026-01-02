import asyncio
import itertools
import json
import os
import re

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.session_waiter import SessionController, session_waiter

from .core import BaseProvider, Downloader, HttpManager
from .core.data import (
    CommonConfig,
    DEF_GEMINI_API_URL,
    DEF_OPENAI_API_URL,
    DEF_OPENAI_IMAGES_API_URL,
    PreferenceConfig,
    PromptConfig,
    ProviderConfig,
    ModelConfig,
)
from .core.llm_tools import BigBananaPromptTool, BigBananaTool, remove_tools
from .core.utils import clear_cache, read_file, save_images

# 提示词参数列表
PARAMS_LIST = [
    "min_images",
    "max_images",
    "refer_images",
    "image_size",
    "aspect_ratio",
    "google_search",
    "preset_append",
    "gather_mode",
    "model",
    "provider",
    "preset",
    "q",
]

# 参数别称映射
PARAMS_ALIAS_MAP = {
    "append_mode": "gather_mode",
    "ar": "aspect_ratio",
    "r": "aspect_ratio",
    "p": "preset",
}

# 支持的文件格式
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

# 提供商配置键列表
provider_keys = ["main_provider", "back_provider", "back_provider2"]

# 部分平台对单张图片大小有限制，超过限制需要作为文件发送
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
# 预计算 Base64 长度阈值 (向下取整)，base64编码约为原始数据的4/3倍
MAX_SIZE_B64_LEN = int(MAX_SIZE_BYTES * 4 / 3)


class BigBanana(Star):
    MAX_CONCURRENT_JOBS = 8

    @staticmethod
    def _normalize_api_url(api_type: str, api_url: str | None) -> str:
        url = (api_url or "").strip()
        if api_type == "Gemini":
            if not url:
                return DEF_GEMINI_API_URL
            url = url.rstrip("/")
            if url.endswith("/v1beta") or url.endswith("/v1"):
                return f"{url}/models"
            if "/models" in url:
                return url
            if "/v1beta/" not in url and "/v1/" not in url and url.count("/") <= 2:
                return f"{url}/v1beta/models"
            return url
        if api_type == "OpenAI_Chat":
            if not url:
                return DEF_OPENAI_API_URL
            url = url.rstrip("/")
            if "chat/completions" in url:
                return url
            if url.endswith("/v1"):
                return f"{url}/chat/completions"
            if url.endswith("/v1/async"):
                return f"{url}/chat/completions"
            if "/v1/" not in url and url.count("/") <= 2:
                return f"{url}/v1/chat/completions"
            return url
        if api_type == "OpenAI_Images":
            if not url:
                return DEF_OPENAI_IMAGES_API_URL
            url = url.rstrip("/")
            if "images/generations" in url:
                return url
            if "images/edits" in url or "images/variations" in url:
                return url
            return url
        return url.rstrip("/")

    @staticmethod
    def _parse_keys(raw: object) -> list[str]:
        if not isinstance(raw, str):
            return []
        parts = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        tokens: list[str] = []
        for part in parts:
            tokens.extend(part.split(","))
        return [t.strip() for t in tokens if t.strip()]

    def _collect_image_urls(self, event: AstrMessageEvent) -> list[str]:
        image_urls: list[str] = []
        for comp in event.get_messages():
            if isinstance(comp, Comp.Reply) and getattr(comp, "chain", None):
                quote_chain = comp.chain
                if not isinstance(quote_chain, list) and hasattr(quote_chain, "chain"):
                    quote_chain = getattr(quote_chain, "chain", None)
                for quote in quote_chain or []:
                    if isinstance(quote, Comp.Image) and quote.url:
                        image_urls.append(quote.url)
                    elif (
                        isinstance(quote, Comp.File)
                        and quote.url
                        and quote.url.startswith("http")
                        and quote.url.lower().endswith(SUPPORTED_FILE_FORMATS)
                    ):
                        image_urls.append(quote.url)
            elif isinstance(comp, Comp.Image) and comp.url:
                image_urls.append(comp.url)
            elif (
                isinstance(comp, Comp.File)
                and comp.url
                and comp.url.startswith("http")
                and comp.url.lower().endswith(SUPPORTED_FILE_FORMATS)
            ):
                image_urls.append(comp.url)
        return image_urls

    async def _try_call_platform_api(
        self, event: AstrMessageEvent, action: str, params: dict
    ) -> object | None:
        candidates: list[object] = [event]
        for attr in ("bot", "_bot", "platform", "client", "_client"):
            obj = getattr(event, attr, None)
            if obj is not None:
                candidates.append(obj)

        for target in candidates:
            for method_name in ("call_action", "call_api", "api_call", "request"):
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    return await method(action, **params)
                except TypeError:
                    try:
                        return await method(action, params)
                    except Exception:
                        continue
                except Exception:
                    continue
        return None

    async def _collect_reply_image_urls(
        self, event: AstrMessageEvent, reply_comp: object
    ) -> tuple[list[str], str]:
        urls: list[str] = []
        reply_sender_id = str(getattr(reply_comp, "sender_id", "") or "")

        quote_chain = getattr(reply_comp, "chain", None)
        if quote_chain is not None:
            if not isinstance(quote_chain, list) and hasattr(quote_chain, "chain"):
                quote_chain = getattr(quote_chain, "chain", None)
            for quote in quote_chain or []:
                if isinstance(quote, Comp.Image) and quote.url:
                    urls.append(quote.url)
                elif (
                    isinstance(quote, Comp.File)
                    and quote.url
                    and quote.url.startswith("http")
                    and quote.url.lower().endswith(SUPPORTED_FILE_FORMATS)
                ):
                    urls.append(quote.url)
            if urls:
                return urls, reply_sender_id

        reply_id = None
        for key in ("id", "message_id", "msg_id", "reply_id"):
            value = getattr(reply_comp, key, None)
            if value is None:
                continue
            if isinstance(value, (int, str)) and str(value).strip():
                reply_id = str(value).strip()
                break

        if not reply_id:
            return [], reply_sender_id

        resp = await self._try_call_platform_api(
            event, "get_msg", {"message_id": int(reply_id)}
        )
        if resp is None:
            return [], reply_sender_id

        data = resp
        if isinstance(resp, dict) and "data" in resp:
            data = resp.get("data")

        message = None
        if isinstance(data, dict):
            message = (
                data.get("message")
                or data.get("message_chain")
                or data.get("messageChain")
                or data.get("raw_message")
            )

        def append_if_media(url: object):
            if not isinstance(url, str):
                return
            u = url.strip()
            if not u:
                return
            if u.startswith("http") and u.lower().endswith(SUPPORTED_FILE_FORMATS):
                urls.append(u)
                return
            if u.startswith("http") and any(ext in u.lower() for ext in SUPPORTED_FILE_FORMATS):
                urls.append(u)

        if isinstance(message, list):
            for seg in message:
                if not isinstance(seg, dict):
                    continue
                t = seg.get("type")
                d = seg.get("data") if isinstance(seg.get("data"), dict) else {}
                if t == "image":
                    append_if_media(d.get("url") or d.get("file"))
                elif t == "file":
                    append_if_media(d.get("url") or d.get("name") or d.get("file"))
            return urls, reply_sender_id

        if isinstance(message, str) and message:
            for m in re.finditer(r"url=([^,\\]]+)", message):
                append_if_media(m.group(1))
            for m in re.finditer(r"file=([^,\\]]+)", message):
                append_if_media(m.group(1))
        return urls, reply_sender_id

    async def _image_to_prompt(
        self, event: AstrMessageEvent, prompt: str, min_required_images: int
    ) -> str:
        if not hasattr(self, "http_manager") or not hasattr(self, "downloader"):
            return "❌ 插件未初始化完成，请稍后再试"

        itp_conf = self.conf.get("Image-to-Prompt", {})
        if not isinstance(itp_conf, dict):
            itp_conf = {}

        if not bool(itp_conf.get("enabled", True)):
            return "❌ Image-to-Prompt 未启用"

        model = str(itp_conf.get("model", "") or "").strip()
        system_prompt = str(itp_conf.get("system_prompt", "") or "").strip()

        primary_conf = itp_conf.get("primary", {})
        if not isinstance(primary_conf, dict):
            primary_conf = {}

        api_url = primary_conf.get("api_url", itp_conf.get("api_url", ""))
        api_key_raw = primary_conf.get("api_key", itp_conf.get("api_key", ""))
        tls_verify = primary_conf.get("tls_verify", itp_conf.get("tls_verify", True))
        impersonate = primary_conf.get("impersonate", itp_conf.get("impersonate", ""))
        api_url = self._normalize_api_url("OpenAI_Chat", str(api_url or ""))
        keys = self._parse_keys(api_key_raw)

        if not keys:
            return "❌ Image-to-Prompt 未配置 API Key"

        image_urls = self._collect_image_urls(event)
        required = max(1, int(min_required_images or 1))
        if len(image_urls) < required:
            return f"❌ 需要至少 {required} 张图片"

        b64_images = await self.downloader.fetch_images(image_urls)
        if not b64_images:
            return "❌ 图片下载失败"

        images_content = []
        for mime, b64 in b64_images:
            images_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *images_content],
            }
        )

        body = {"messages": messages, "stream": False, "max_tokens": 1024}
        if model:
            body["model"] = model
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {keys[0]}",
        }

        try:
            req_kwargs = {
                "timeout": self.common_config.timeout,
                "proxy": self.common_config.proxy,
                "verify": bool(tls_verify),
            }
            if isinstance(impersonate, str) and impersonate.strip():
                req_kwargs["impersonate"] = impersonate.strip()
            response = await self.http_manager._get_curl_session().post(
                url=api_url,
                headers=headers,
                json=body,
                **req_kwargs,
            )
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                logger.error(
                    f"[BIG BANANA] Image-to-Prompt JSON反序列化错误: {e}，状态码：{response.status_code}，响应内容：{response.text[:1024]}"
                )
                return "❌ 反推失败：响应内容格式错误"
            if response.status_code != 200:
                msg = None
                if isinstance(result, dict):
                    if isinstance(result.get("message"), str):
                        msg = result.get("message")
                    err = result.get("error")
                    if not msg and isinstance(err, dict):
                        msg = err.get("message")
                logger.error(
                    f"[BIG BANANA] Image-to-Prompt 失败，状态码: {response.status_code}, 响应内容: {response.text[:1024]}"
                )
                return f"❌ 反推失败：{msg or f'状态码 {response.status_code}'}"

            choices = result.get("choices") if isinstance(result, dict) else None
            if not isinstance(choices, list) or not choices:
                return "❌ 反推失败：响应缺少 choices"
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                content = "\n".join(p for p in parts if p.strip())
            if not isinstance(content, str) or not content.strip():
                return "❌ 反推失败：响应缺少内容"
            return content.strip()
        except Exception as e:
            logger.error(f"[BIG BANANA] Image-to-Prompt 请求错误: {e}", exc_info=True)
            return "❌ 反推失败：请求错误"

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        # 初始化提示词配置
        self.init_prompts()
        self.user_selected_provider_model: dict[str, str] = {}
        # 白名单配置
        self.whitelist_config = self.conf.get("whitelist_config", {})
        # 群组白名单，列表是引用类型
        self.group_whitelist_enabled = self.whitelist_config.get("enabled", False)
        self.group_whitelist = self.whitelist_config.get("whitelist", [])
        # 用户白名单
        self.user_whitelist_enabled = self.whitelist_config.get("user_enabled", False)
        self.user_whitelist = self.whitelist_config.get("user_whitelist", [])

        nanobanana_conf = self.conf.get("nanobanana_config", {})
        if not isinstance(nanobanana_conf, dict):
            nanobanana_conf = {}
        nanobanana_whitelist = nanobanana_conf.get("whitelist", {})
        if not isinstance(nanobanana_whitelist, dict):
            nanobanana_whitelist = {}
        self.nanobanana_group_whitelist_enabled = bool(
            nanobanana_whitelist.get("enabled", False)
        )
        self.nanobanana_group_whitelist = nanobanana_whitelist.get("whitelist", [])
        self.nanobanana_user_whitelist_enabled = bool(
            nanobanana_whitelist.get("user_enabled", False)
        )
        self.nanobanana_user_whitelist = nanobanana_whitelist.get("user_whitelist", [])

        # 前缀配置
        prefix_config = self.conf.get("prefix_config", {})
        self.coexist_enabled = prefix_config.get("coexist_enabled", False)
        self.prefix_list = prefix_config.get("prefix_list", [])

        # 数据目录
        data_dir = StarTools.get_data_dir("astrbot_plugin_big_banana")
        self.refer_images_dir = data_dir / "refer_images"
        self.save_dir = data_dir / "save_images"
        # 临时文件目录
        self.temp_dir = data_dir / "temp_images"

        # 图片持久化
        self.save_images = self.conf.get("save_images", {}).get("local_save", False)

        # 正在运行的任务映射
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.job_semaphore: asyncio.Semaphore | None = None

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        # 初始化文件目录
        os.makedirs(self.refer_images_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)

        # 实例化类
        self.preference_config = PreferenceConfig(
            **self.conf.get("preference_config", {})
        )
        self.common_config = CommonConfig(**self.conf.get("common_config", {}))
        self.prompt_config = PromptConfig(**self.conf.get("prompt_config", {}))
        self.http_manager = HttpManager()
        curl_session = self.http_manager._get_curl_session()
        self.downloader = Downloader(curl_session, self.common_config)
        self.job_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_JOBS)

        # 注册提供商类型实例
        self.init_providers()
        self.init_prompts()

        # 检查配置是否启用函数调用工具
        if self.conf.get("llm_tool_settings", {}).get("llm_tool_enabled", False):
            self.context.add_llm_tools(BigBananaTool(plugin=self))
            logger.info("已注册函数调用工具: banana_image_generation")
            self.context.add_llm_tools(BigBananaPromptTool(plugin=self))
            logger.info("已注册函数调用工具: banana_preset_prompt")

    def init_providers(self):
        """解析提供商配置"""
        # 模型配置列表
        self.models: list[ModelConfig] = []
        # 提供商实例映射
        self.provider_map: dict[str, BaseProvider] = {}
        
        # 1. 获取模型配置列表
        models_data = self.conf.get("models", [])
        
        # 兼容旧配置：如果 models 为空，检查是否有 extra_models 或 default_model 并迁移
        # 这是一个临时迁移逻辑，防止用户更新后配置丢失
        if not models_data:
            if "extra_models" in self.conf:
                models_data.extend(self.conf.get("extra_models", []))
            
            if "default_model" in self.conf:
                default_model = self.conf.get("default_model")
                if default_model:
                     # 构造一个 Model 对象
                     models_data.insert(0, {
                         "name": "默认画图配置",
                         "triggers": default_model.get("triggers", []),
                         "providers": default_model.get("providers", []),
                         "enabled": default_model.get("enabled", True)
                     })
            
            # 如果从旧配置迁移了数据，保存回 models
            if models_data:
                self.conf["models"] = models_data
                # 清理旧键（可选，但为了保持配置整洁最好清理）
                self.conf.pop("extra_models", None)
                self.conf.pop("default_model", None)
                self.conf.save_config()

        # 如果仍然为空，且是首次运行，可能会读取默认配置（由 schema 定义）
        if not isinstance(models_data, list):
            models_data = []

        updated_models = False

        def parse_keys(raw: object) -> list[str]:
            if not isinstance(raw, str):
                return []
            parts = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            tokens: list[str] = []
            for part in parts:
                tokens.extend(part.split(","))
            return [t.strip() for t in tokens if t.strip()]

        def upsert_fixed_model(
            conf_key: str,
            name: str,
            default_triggers: list[str],
            default_provider_stub: dict,
            insert_index: int,
        ) -> None:
            nonlocal updated_models, models_data

            model_conf = self.conf.get(conf_key, {})
            if not isinstance(model_conf, dict):
                model_conf = {}

            enabled = bool(model_conf.get("enabled", True))

            providers = model_conf.get("providers", None)
            if not isinstance(providers, list) or not providers:
                provider_list: list[dict] = []

                primary_conf = model_conf.get("primary", {})
                if not isinstance(primary_conf, dict):
                    primary_conf = {}

                def build_provider_item(conf: dict, suffix: str) -> dict:
                    api_base_mapping = {
                        "t8star": "https://ai.t8star.cn",
                        "zhenzhen": "https://ai.t8star.cn",
                        "hk": "https://hk-api.gptbest.vip",
                        "us": "https://api.gptbest.vip",
                        "grsai": "https://grsaiapi.com",
                    }
                    api_type = conf.get("api_type", None)
                    if not isinstance(api_type, str) or not api_type.strip():
                        api_type = default_provider_stub.get("api_type")
                    api_type = str(api_type).strip()
                    if conf_key == "nanobanana_config":
                        if suffix == "主":
                            api_type = "OpenAI_Images"
                        elif suffix == "备" and not str(conf.get("api_type", "") or "").strip():
                            api_type = "OpenAI_Chat"

                    item = dict(default_provider_stub)
                    item["api_type"] = api_type

                    model_name = conf.get("model", None)
                    if isinstance(model_name, str) and model_name.strip():
                        item["model"] = model_name.strip()

                    url = conf.get("api_url", model_conf.get("api_url", ""))
                    api_base = conf.get("api_base", None)
                    if isinstance(api_base, str) and api_base.strip():
                        api_base = api_base.strip()
                        if api_base == "ip":
                            custom_ip = conf.get("custom_ip", None)
                            if isinstance(custom_ip, str) and custom_ip.strip():
                                url = custom_ip.strip()
                        elif api_base in api_base_mapping:
                            url = api_base_mapping[api_base]
                    item["api_url"] = url

                    tls_verify = conf.get("tls_verify", None)
                    if isinstance(tls_verify, bool):
                        item["tls_verify"] = tls_verify
                    impersonate = conf.get("impersonate", None)
                    if isinstance(impersonate, str) and impersonate.strip():
                        item["impersonate"] = impersonate.strip()

                    if (
                        item.get("api_type") == "OpenAI_Chat"
                        and isinstance(url, str)
                        and "/images/" in url.lower()
                    ):
                        item["api_type"] = "OpenAI_Images"

                    if api_type == "Vertex_AI_Anonymous":
                        item["keys"] = []
                        base_name = "Vertex匿名"
                    else:
                        key = conf.get("api_key", model_conf.get("api_key", ""))
                        item["keys"] = parse_keys(key)
                        base_name = str(default_provider_stub.get("name", name))

                    item["name"] = f"{base_name}_{suffix}"
                    return item

                primary_data = build_provider_item(primary_conf, "主")
                provider_list.append(primary_data)

                secondary_conf = model_conf.get("secondary", {})
                if not isinstance(secondary_conf, dict):
                    secondary_conf = {}
                secondary_url = secondary_conf.get("api_url", "")
                secondary_key = secondary_conf.get("api_key", "")
                secondary_api_type = secondary_conf.get("api_type", "")
                secondary_model = secondary_conf.get("model", "")
                if conf_key == "nanobanana_config" and not (
                    str(secondary_api_type).strip()
                    or str(secondary_url).strip()
                    or str(secondary_key).strip()
                    or str(secondary_model).strip()
                ):
                    secondary_url = "https://grsaiapi.com"
                    secondary_api_type = "OpenAI_Chat"
                    secondary_model = "nano-banana-pro"
                if (
                    str(secondary_api_type).strip() == "Vertex_AI_Anonymous"
                    or str(secondary_url).strip()
                    or str(secondary_key).strip()
                    or str(secondary_model).strip()
                ):
                    secondary_data = build_provider_item(
                        {
                            "api_type": secondary_api_type,
                            "api_url": secondary_url,
                            "api_key": secondary_key,
                            "model": secondary_model,
                        },
                        "备",
                    )
                    provider_list.append(secondary_data)

                providers = provider_list

            triggers = default_triggers

            new_data = {
                "name": name,
                "triggers": triggers,
                "providers": providers,
                "enabled": enabled,
            }

            target = next(
                (m for m in models_data if isinstance(m, dict) and m.get("name") == name),
                None,
            )
            if target is None:
                models_data.insert(insert_index, new_data)
                updated_models = True
                return

            if (
                target.get("triggers") != triggers
                or target.get("providers") != providers
                or target.get("enabled", True) != enabled
            ):
                target.update(new_data)
                updated_models = True

        upsert_fixed_model(
            conf_key="nanobanana_config",
            name="nano-banana",
            default_triggers=["bnn", "bnt", "bna"],
            default_provider_stub={
                "name": "nano-banana账号",
                "enabled": True,
                "api_type": "OpenAI_Images",
                "keys": [],
                "api_url": "https://ai.t8star.cn",
                "model": "nano-banana-2-2k",
                "stream": False,
            },
            insert_index=0,
        )
        upsert_fixed_model(
            conf_key="zimage_config",
            name="Z-Image-Turbo",
            default_triggers=["zimg"],
            default_provider_stub={
                "name": "Z-Image账号",
                "enabled": True,
                "api_type": "OpenAI_Images",
                "keys": [],
                "api_url": DEF_OPENAI_IMAGES_API_URL,
                "model": "Z-Image-Turbo",
                "stream": False,
            },
            insert_index=1,
        )
        upsert_fixed_model(
            conf_key="qwen_edit_2511_config",
            name="Qwen-Image-Edit-2511",
            default_triggers=["edit"],
            default_provider_stub={
                "name": "Qwen账号",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": DEF_OPENAI_API_URL,
                "model": "Qwen-Image-Edit-2511",
                "stream": False,
            },
            insert_index=2,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api bt1 文生图 横屏",
            default_triggers=["bt1"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "gemini-3.0-pro-image-landscape",
                "stream": True,
            },
            insert_index=3,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api bt2 文生图 竖屏",
            default_triggers=["bt2"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "gemini-3.0-pro-image-portrait",
                "stream": True,
            },
            insert_index=4,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api bp1 图生图 横屏",
            default_triggers=["bp1"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "gemini-3.0-pro-image-landscape",
                "stream": True,
            },
            insert_index=5,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api bp2 图生图 竖屏",
            default_triggers=["bp2"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "gemini-3.0-pro-image-portrait",
                "stream": True,
            },
            insert_index=6,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api tv1 文生视频 横屏",
            default_triggers=["tv1"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "veo_3_1_t2v_fast_landscape",
                "stream": True,
            },
            insert_index=7,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api tv2 文生视频 竖屏",
            default_triggers=["tv2"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "veo_3_1_t2v_fast_portrait",
                "stream": True,
            },
            insert_index=8,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api iv1 首尾帧图生视频 横屏",
            default_triggers=["iv1"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "veo_3_1_i2v_s_fast_fl_landscape",
                "stream": True,
            },
            insert_index=9,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api iv2 首尾帧图生视频 竖屏",
            default_triggers=["iv2"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "veo_3_1_i2v_s_fast_fl_portrait",
                "stream": True,
            },
            insert_index=10,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api rv1 图生视频 横屏",
            default_triggers=["rv1"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "veo_3_0_r2v_fast_landscape",
                "stream": True,
            },
            insert_index=11,
        )
        upsert_fixed_model(
            conf_key="flow2api_config",
            name="flow2api rv2 图生视频 竖屏",
            default_triggers=["rv2"],
            default_provider_stub={
                "name": "flow2api",
                "enabled": True,
                "api_type": "OpenAI_Chat",
                "keys": [],
                "api_url": "http://192.168.2.109:8300/v1/chat/completions",
                "model": "veo_3_0_r2v_fast_portrait",
                "stream": True,
            },
            insert_index=12,
        )

        if updated_models:
            self.conf["models"] = models_data
            self.conf.save_config()
        
        for model_data in models_data:
            # Parse ProviderConfig list
            providers_data = model_data.get("providers", [])
            providers = []
            for provider_data in providers_data:
                if provider_data.get("enabled", False):
                    # 过滤掉不在 ProviderConfig 中的字段
                    payload = {
                        k: v
                        for k, v in provider_data.items()
                        if k in ProviderConfig.__annotations__
                    }
                    payload.setdefault("keys", [])
                    payload["api_url"] = self._normalize_api_url(
                        payload.get("api_type", ""), payload.get("api_url", "")
                    )
                    p_config = ProviderConfig(**payload)
                    providers.append(p_config)
            
            # Create ModelConfig
            model_config = ModelConfig(
                name=model_data.get("name", "Unknown"),
                triggers=model_data.get("triggers", []),
                providers=providers,
                enabled=model_data.get("enabled", True)
            )
            if model_config.enabled:
                self.models.append(model_config)

        # 3. 收集所有需要的 API 类型并实例化
        needed_api_types = set()
        for model in self.models:
            for provider in model.providers:
                needed_api_types.add(provider.api_type)

        # 实例化提供商类
        for api_type in needed_api_types:
            provider_cls = BaseProvider.get_provider_class(api_type)
            if provider_cls is None:
                logger.warning(
                    f"未找到提供商类型对应的提供商类：{api_type}，跳过该提供商配置"
                )
                continue
            self.provider_map[api_type] = provider_cls(
                config=self.conf,
                common_config=self.common_config,
                prompt_config=self.prompt_config,
                session=self.http_manager._get_curl_session(),
                downloader=self.downloader,
                aiohttp_session=self.http_manager._get_aiohttp_session(),
            )

    def init_prompts(self):
        """初始化提示词配置"""
        # 预设提示词列表
        self.prompt_list = self.conf.get("prompt", [])
        self.prompt_dict = {}
        existing_cmds: set[str] = set()
        for item in self.prompt_list:
            cmd_list, params = self.parsing_prompt_params(item)
            for cmd in cmd_list:
                existing_cmds.add(cmd)
                self.prompt_dict[cmd] = params

        fixed_prompts: dict[str, str] = {
            "bt1": "bt1 {{user_text}} --min_images 0",
            "bt2": "bt2 {{user_text}} --min_images 0",
            "bp1": "bp1 {{user_text}} --min_images 1",
            "bp2": "bp2 {{user_text}} --min_images 1",
            "tv1": "tv1 {{user_text}} --min_images 0",
            "tv2": "tv2 {{user_text}} --min_images 0",
            "iv1": "iv1 {{user_text}} --min_images 2",
            "iv2": "iv2 {{user_text}} --min_images 2",
            "rv1": "rv1 {{user_text}} --min_images 1",
            "rv2": "rv2 {{user_text}} --min_images 1",
        }
        updated_prompts = False
        for trigger, prompt_line in fixed_prompts.items():
            if trigger in existing_cmds:
                continue
            cmd_list, params = self.parsing_prompt_params(prompt_line)
            self.prompt_list.append(prompt_line)
            updated_prompts = True
            for cmd in cmd_list:
                existing_cmds.add(cmd)
                self.prompt_dict[cmd] = params

        # 将模型触发词也加入到 prompt_dict 中，以便在 on_message 中能通过检查
        # 如果触发词已存在（即有预设提示词），则不做处理（预设提示词优先）
        # 如果触发词不存在，则添加一个默认的提示词配置
        if hasattr(self, "models"):
            for model in self.models:
                for trigger in model.triggers:
                    if trigger not in self.prompt_dict:
                        self.prompt_dict[trigger] = {
                            "prompt": "{{user_text}}",
                            "__model_name__": model.name
                        }
                    else:
                        # 如果已存在，标记该提示词属于哪个模型（如果未指定）
                        if "__model_name__" not in self.prompt_dict[trigger]:
                            self.prompt_dict[trigger]["__model_name__"] = model.name

        if updated_prompts:
            self.conf["prompt"] = self.prompt_list
            self.conf.save_config()

    def parsing_prompt_params(self, prompt: str) -> tuple[list[str], dict]:
        """解析提示词中的参数，若没有指定参数则使用默认值填充。必须是包括命令和参数的完整提示词"""

        # 以空格分割单词
        tokens = prompt.split()
        # 第一个单词作为命令或命令列表
        cmd_raw = tokens[0]

        # 解析多触发词
        if cmd_raw.startswith("[") and cmd_raw.endswith("]"):
            # 移除括号并按逗号分割
            cmd_list = cmd_raw[1:-1].split(",")
        else:
            cmd_list = [cmd_raw]

        # 迭代器跳过第一个单词
        tokens_iter = iter(tokens[1:])
        # 提示词传递参数列表
        params = {}
        # 过滤后的提示词单词列表
        filtered = []

        # 解析参数
        while True:
            token = next(tokens_iter, None)
            if token is None:
                break
            if token.startswith("--"):
                key = token[2:]
                # 处理参数别称映射
                if key in PARAMS_ALIAS_MAP:
                    key = PARAMS_ALIAS_MAP[key]
                # 仅处理已知参数
                if key in PARAMS_LIST:
                    value = next(tokens_iter, None)
                    if value is None:
                        params[key] = True
                        break
                    value = value.strip()
                    if value.startswith("--"):
                        params[key] = True
                        # 将被提前迭代的单词放回迭代流的最前端
                        tokens_iter = itertools.chain([value], tokens_iter)
                        continue
                    elif value.lower() == "true":
                        params[key] = True
                    elif value.lower() == "false":
                        params[key] = False
                    # 处理字符串数字类型
                    elif value.isdigit():
                        params[key] = int(value)
                    else:
                        params[key] = value
                    continue
            filtered.append(token)

        # 重新组合提示词
        prompt = " ".join(filtered)
        params["prompt"] = prompt
        return cmd_list, params

    # === 辅助功能：判断管理员，用于静默跳出 ===
    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        """检查发送者是否为全局管理员"""
        admin_ids = self.context.get_config().get("admins_id", [])
        # logger.info(f"全局管理员列表：{admin_ids}")
        return event.get_sender_id() in admin_ids

    @filter.command("使用模型切换")
    async def switch_provider_model_command(
        self, event: AstrMessageEvent, model_id: str = ""
    ):
        raw = (event.message_str or "").strip()
        if not model_id:
            tokens = raw.split()
            if len(tokens) >= 2:
                model_id = tokens[1].strip()

        model_map = {
            "1": "gemini-3.0-pro-image-portrait",
            "2": "gemini-3.0-pro-image-landscape",
            "3": "nano-banana-pro",
        }
        token = str(model_id).strip()
        key = token[:1] if token else ""
        if key not in model_map:
            m = re.search(r"([123])", token)
            key = m.group(1) if m else ""
        chosen = model_map.get(key)
        if not chosen:
            yield event.plain_result(
                "❌ 用法：使用模型切换 <1/2/3>\n1：gemini-3.0-pro-image-portrait\n2：gemini-3.0-pro-image-landscape\n3：nano-banana-pro"
            )
            return

        self.user_selected_provider_model[event.get_sender_id()] = chosen
        yield event.plain_result(f"✅ 已切换模型：{key}（{chosen}）")

    # === 管理指令：白名单管理 ===
    @filter.command("lm白名单添加", alias={"lmawl"})
    async def add_whitelist_command(
        self, event: AstrMessageEvent, cmd_type: str = "", target_id: str = ""
    ):
        """lm白名单添加 <用户/群组> <ID>"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lm白名单添加，权限不足"
            )
            return

        if not cmd_type or not target_id:
            yield event.plain_result(
                "❌ 格式错误。\n用法：lm白名单添加 (用户/群组) (ID)"
            )
            return

        msg_type = ""
        if cmd_type in ["用户", "user"] and target_id not in self.user_whitelist:
            msg_type = "用户"
            self.user_whitelist.append(target_id)
        elif cmd_type in ["群组", "group"] and target_id not in self.group_whitelist:
            msg_type = "群组"
            self.group_whitelist.append(target_id)
        elif cmd_type not in ["用户", "user", "群组", "group"]:
            yield event.plain_result("❌ 类型错误，请使用「用户」或「群组」。")
            return
        else:
            yield event.plain_result(f"⚠️ {target_id} 已在名单列表中。")
            return

        yield event.plain_result(f"✅ 已添加{msg_type}白名单：{target_id}")

    @filter.command("lm白名单删除", alias={"lmdwl"})
    async def del_whitelist_command(
        self, event: AstrMessageEvent, cmd_type: str = "", target_id: str = ""
    ):
        """lm白名单删除 <用户/群组> <ID>"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lm白名单删除，权限不足"
            )
            return

        if not cmd_type or not target_id:
            yield event.plain_result(
                "❌ 格式错误。\n用法：lm白名单删除 (用户/群组) (ID)"
            )
            return

        if cmd_type in ["用户", "user"] and target_id in self.user_whitelist:
            msg_type = "用户"
            self.user_whitelist.remove(target_id)
        elif cmd_type in ["群组", "group"] and target_id in self.group_whitelist:
            msg_type = "群组"
            self.group_whitelist.remove(target_id)
        elif cmd_type not in ["用户", "user", "群组", "group"]:
            yield event.plain_result("❌ 类型错误，请使用「用户」或「群组」。")
            return
        else:
            yield event.plain_result(f"⚠️ {target_id} 不在名单列表中。")
            return

        self.conf.save_config()
        yield event.plain_result(f"🗑️ 已删除{msg_type}白名单：{target_id}")

    @filter.command("lm白名单列表", alias={"lmwll"})
    async def list_whitelist_command(self, event: AstrMessageEvent):
        """lm白名单列表"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lm白名单列表，权限不足"
            )
            return

        msg = f"""📋 白名单配置状态：
=========
🏢 群组限制：{"✅ 开启" if self.group_whitelist_enabled else "⬜ 关闭"}
列表：{self.group_whitelist}
=========
👤 用户限制：{"✅ 开启" if self.user_whitelist_enabled else "⬜ 关闭"}
列表：{self.user_whitelist}"""

        yield event.plain_result(msg)

    # === 管理指令：模型管理 ===
    @filter.command("lm模型列表", alias={"lmml"})
    async def list_models_command(self, event: AstrMessageEvent):
        """lm模型列表 - 查看当前配置的模型和提供商"""
        if not self.is_global_admin(event):
            return

        msg = ["📋 当前模型配置："]
        if not self.models:
            msg.append("暂无模型配置。")
        
        for i, model in enumerate(self.models):
            msg.append(f"{i+1}. {model.name} [{'✅启用' if model.enabled else '❌禁用'}]")
            msg.append(f"   触发词: {', '.join(model.triggers)}")
            if not model.providers:
                msg.append("   提供商: 无")
            else:
                msg.append(f"   提供商 ({len(model.providers)}):")
                for j, provider in enumerate(model.providers):
                    msg.append(f"     {j+1}. [{provider.api_type}] {provider.name} {'(✅)' if provider.enabled else '(❌)'}")
        
        yield event.plain_result("\n".join(msg))


    @filter.command("lm触发词添加", alias={"lmtka"})
    async def add_model_trigger_command(self, event: AstrMessageEvent, model_name: str = "", trigger: str = ""):
        """lm触发词添加 <模型名称> <触发词>"""
        if not self.is_global_admin(event):
            return

        if not model_name or not trigger:
            yield event.plain_result("❌ 用法：lm触发词添加 <模型名称> <触发词>")
            return

        models_data = self.conf.get("models", [])
        if not isinstance(models_data, list):
            models_data = []

        target_model = None
        for m in models_data:
            if m.get("name") == model_name:
                target_model = m
                break
        
        if not target_model:
            yield event.plain_result(f"❌ 未找到模型：{model_name}")
            return
        
        current_triggers = target_model.get("triggers", [])
        if trigger in current_triggers:
            yield event.plain_result(f"⚠️ 触发词 {trigger} 已存在于模型 {model_name}。")
            return
            
        current_triggers.append(trigger)
        target_model["triggers"] = current_triggers

        self.conf["models"] = models_data
        
        self.conf.save_config()
        self.init_providers()
        self.init_prompts()
        
        yield event.plain_result(f"✅ 已为模型 {model_name} 添加触发词：{trigger}")



    @filter.command("lm提供商添加", alias={"lmpa"})
    async def add_provider_command(self, event: AstrMessageEvent, model_name: str = "", api_type: str = ""):
        """lm提供商添加 <模型名称> <类型: Gemini/OpenAI_Chat>"""
        if not self.is_global_admin(event):
            return

        if not model_name or not api_type:
            yield event.plain_result("❌ 用法：lm提供商添加 <模型名称> <类型>\n支持类型: Gemini, OpenAI_Chat, OpenAI_Images, Vertex_AI_Anonymous")
            return

        # Validate Model
        models_data = self.conf.get("models", [])
        target_model_data = None
        for m in models_data:
            if m.get("name") == model_name:
                target_model_data = m
                break
        
        if not target_model_data:
            yield event.plain_result(f"❌ 未找到模型：{model_name}")
            return

        # Validate Type
        # Note: Should match _API_Type literal
        valid_types = ["Gemini", "OpenAI_Chat", "OpenAI_Images", "Vertex_AI_Anonymous"]
        # Case insensitive match
        api_type_match = next((t for t in valid_types if t.lower() == api_type.lower()), None)
        
        if not api_type_match:
             yield event.plain_result(f"❌ 不支持的类型：{api_type}。\n可选：{', '.join(valid_types)}")
             return
        
        api_type = api_type_match

        # Prepare default config
        provider_name = f"{api_type}_{len(target_model_data.get('providers', [])) + 1}"
        provider_config = {
            "name": provider_name,
            "enabled": True,
            "api_type": api_type,
            "keys": [],
            "api_url": "",
            "model": "gemini-2.0-flash-exp" if "Gemini" in api_type else "gpt-4o",
            "stream": False
        }
        
        # If Vertex_AI_Anonymous, no key needed
        if api_type == "Vertex_AI_Anonymous":
             if "providers" not in target_model_data:
                target_model_data["providers"] = []
             target_model_data["providers"].append(provider_config)
             self.conf.save_config()
             self.init_providers()
             yield event.plain_result(f"✅ 已添加提供商 {provider_name} 到模型 {model_name}。")
             return

        # Interactive Setup
        yield event.plain_result(f"🍌 正在为模型 {model_name} 添加 {api_type} 提供商。\n请在60秒内输入 API Key (如果不需要请输入 'none' 或 'skip')：")
        
        operator_id = event.get_sender_id()
        
        @session_waiter(timeout=60, record_history_chains=False)
        async def waiter(controller: SessionController, ctx: AstrMessageEvent):
            if ctx.get_sender_id() != operator_id:
                return
            
            content = ctx.message_str.strip()
            if content == "取消":
                await ctx.send(ctx.plain_result("已取消。"))
                controller.stop()
                return

            if content.lower() not in ["none", "skip", "跳过"]:
                provider_config["keys"] = [content]
            
            # Add to config
            if "providers" not in target_model_data:
                target_model_data["providers"] = []
            target_model_data["providers"].append(provider_config)
            
            self.conf.save_config()
            self.init_providers()
            
            await ctx.send(ctx.plain_result(f"✅ 已添加提供商 {provider_name} 到模型 {model_name}。\n更多参数（如API地址）请通过配置文件或WebUI修改。"))
            controller.stop()

        try:
            await waiter(event)
        except TimeoutError:
             yield event.plain_result("❌ 超时，操作已取消。")

    @filter.command("lm提供商删除", alias={"lmpd"})
    async def del_provider_command(self, event: AstrMessageEvent, model_name: str = "", provider_index: str = ""):
        """lm提供商删除 <模型名称> <序号>"""
        if not self.is_global_admin(event):
            return

        if not model_name or not provider_index or not provider_index.isdigit():
            yield event.plain_result("❌ 用法：lm提供商删除 <模型名称> <序号(从1开始)>")
            return
        
        idx = int(provider_index) - 1
        
        models_data = self.conf.get("models", [])
        target_model_data = None
        for m in models_data:
            if m.get("name") == model_name:
                target_model_data = m
                break
        
        if not target_model_data:
            yield event.plain_result(f"❌ 未找到模型：{model_name}")
            return
            
        providers = target_model_data.get("providers", [])
        if idx < 0 or idx >= len(providers):
             yield event.plain_result(f"❌ 序号 {provider_index} 无效。当前有 {len(providers)} 个提供商。")
             return
             
        removed = providers.pop(idx)
        target_model_data["providers"] = providers
        
        self.conf.save_config()
        self.init_providers()
        
        yield event.plain_result(f"🗑️ 已从模型 {model_name} 删除提供商：{removed.get('name')}")

    # === 管理指令：添加/更新提示词 ===
    @filter.command("lm添加", alias={"lma"})
    async def add_prompt_command(self, event: AstrMessageEvent, trigger_word: str = ""):
        """lm添加 <触发词> <提示词内容>"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lm添加，权限不足"
            )
            return

        if not trigger_word:
            yield event.plain_result("❌ 格式错误：lm添加 (触发词)")
            return

        yield event.plain_result(
            f"🍌 正在为触发词 「{trigger_word}」 添加/更新提示词\n✦ 请在60秒内输入完整的提示词内容（不含触发词，包含参数）\n✦ 输入「取消」可取消操作。"
        )

        # 记录操作员账号
        operator_id = event.get_sender_id()

        @session_waiter(timeout=60, record_history_chains=False)  # type: ignore
        async def waiter(controller: SessionController, event: AstrMessageEvent):
            # 判断消息来源是否是同一用户（同一用户不需要鉴权了吧）
            if event.get_sender_id() != operator_id:
                return

            if event.message_str.strip() == "取消":
                await event.send(event.plain_result("🍌 操作已取消。"))
                controller.stop()
                return

            build_prompt = f"{trigger_word} {event.message_str.strip()}"

            action = "添加"
            # 直接从字典中查重
            if trigger_word in self.prompt_dict:
                action = "更新"
                # 从提示词列表中找出对应项进行更新
                for i, v in enumerate(self.prompt_list):
                    cmd, _, prompt_str = v.strip().partition(" ")
                    if cmd == trigger_word:
                        self.prompt_list[i] = build_prompt
                        break
                    # 处理多触发词
                    if cmd.startswith("[") and cmd.endswith("]"):
                        # 移除括号并按逗号分割
                        cmd_list = cmd[1:-1].split(",")
                        if trigger_word in cmd_list:
                            # 将这个提示词从多触发提示词中移除
                            cmd_list.remove(trigger_word)
                            # 重新构建提示词字符串
                            if len(cmd_list) == 1:
                                # 仅剩一个触发词，改为单触发词形式
                                new_config_item = f"{cmd_list[0]} {prompt_str}"
                            else:
                                new_cmd = "[" + ",".join(cmd_list) + "]"
                                new_config_item = f"{new_cmd} {prompt_str}"
                            self.prompt_list[i] = new_config_item
                            # 最后为新的提示词添加一项
                            self.prompt_list.append(build_prompt)
                            break
            # 新增提示词
            else:
                self.prompt_list.append(build_prompt)

            self.conf.save_config()
            self.init_prompts()
            await event.send(
                event.plain_result(f"✅ 已成功{action}提示词：「{trigger_word}」")
            )
            controller.stop()

        try:
            await waiter(event)
        except TimeoutError as _:
            yield event.plain_result("❌ 超时了，操作已取消！")
        except Exception as e:
            logger.error(f"大香蕉添加提示词出现错误: {e}", exc_info=True)
            yield event.plain_result("❌ 处理时发生了一个内部错误。")
        finally:
            event.stop_event()

    @filter.command("lmp")
    async def add_prompt_quick_command(
        self, event: AstrMessageEvent, trigger_word: str = "", prompt_str: str = ""
    ):
        """添加提示词预设"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lmp，权限不足"
            )
            return

        raw = (event.message_str or "").strip()
        if not trigger_word:
            tokens = raw.split()
            if len(tokens) >= 2:
                trigger_word = tokens[1]
            else:
                yield event.plain_result("❌ 用法：lmp <触发词> <提示词内容>")
                return

        if raw and trigger_word in raw:
            suffix = raw.split(trigger_word, 1)[1].strip()
            if suffix:
                prompt_str = suffix

        if not prompt_str.strip():
            yield event.plain_result(
                f"📝 请发送提示词内容（用于触发词「{trigger_word}」），30 秒内有效。"
            )

            @session_waiter(timeout=30, record_history_chains=False)  # type: ignore
            async def waiter(controller: SessionController, event: AstrMessageEvent):
                if not self.is_global_admin(event):
                    logger.info(
                        f"用户 {event.get_sender_id()} 试图执行管理员命令 lmp，权限不足"
                    )
                    return

                reply = (event.message_str or "").strip()
                if not reply:
                    await event.send(
                        event.plain_result("❌ 提示词内容不能为空，请重新发送。")
                    )
                    return

                build_prompt = f"{trigger_word} {reply}"
                action = "添加"

                if trigger_word in self.prompt_dict:
                    action = "更新"
                    for i, v in enumerate(self.prompt_list):
                        cmd, _, existing_prompt_str = v.strip().partition(" ")
                        if cmd == trigger_word:
                            self.prompt_list[i] = build_prompt
                            break
                        if cmd.startswith("[") and cmd.endswith("]"):
                            cmd_list = cmd[1:-1].split(",")
                            if trigger_word in cmd_list:
                                cmd_list.remove(trigger_word)
                                if len(cmd_list) == 1:
                                    new_config_item = (
                                        f"{cmd_list[0]} {existing_prompt_str}"
                                    )
                                else:
                                    new_cmd = "[" + ",".join(cmd_list) + "]"
                                    new_config_item = f"{new_cmd} {existing_prompt_str}"
                                self.prompt_list[i] = new_config_item
                                self.prompt_list.append(build_prompt)
                                break
                else:
                    self.prompt_list.append(build_prompt)

                self.conf.save_config()
                self.init_prompts()
                await event.send(
                    event.plain_result(
                        f"✅ 已成功{action}提示词：「{trigger_word}」"
                    )
                )
                controller.stop()

            try:
                await waiter(event)
            except TimeoutError:
                yield event.plain_result("❌ 超时了，操作已取消！")
            except Exception as e:
                logger.error(f"lmp 追加提示词出现错误: {e}", exc_info=True)
                yield event.plain_result("❌ 处理时发生了一个内部错误。")
            finally:
                event.stop_event()
            return

        build_prompt = f"{trigger_word} {prompt_str}"
        action = "添加"

        if trigger_word in self.prompt_dict:
            action = "更新"
            for i, v in enumerate(self.prompt_list):
                cmd, _, existing_prompt_str = v.strip().partition(" ")
                if cmd == trigger_word:
                    self.prompt_list[i] = build_prompt
                    break
                if cmd.startswith("[") and cmd.endswith("]"):
                    cmd_list = cmd[1:-1].split(",")
                    if trigger_word in cmd_list:
                        cmd_list.remove(trigger_word)
                        if len(cmd_list) == 1:
                            new_config_item = f"{cmd_list[0]} {existing_prompt_str}"
                        else:
                            new_cmd = "[" + ",".join(cmd_list) + "]"
                            new_config_item = f"{new_cmd} {existing_prompt_str}"
                        self.prompt_list[i] = new_config_item
                        self.prompt_list.append(build_prompt)
                        break
        else:
            self.prompt_list.append(build_prompt)

        self.conf.save_config()
        self.init_prompts()
        yield event.plain_result(f"✅ 已成功{action}提示词：「{trigger_word}」")

    @filter.command("lm列表", alias={"lml", "lmpl"})
    async def list_prompts_command(self, event: AstrMessageEvent):
        """lm列表"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lm列表，权限不足"
            )
            return

        prompts = list(self.prompt_dict.keys())
        if not prompts:
            yield event.plain_result("当前没有预设提示词。")
            return

        msg = "📜 当前预设提示词列表：\n" + "、".join(prompts)
        yield event.plain_result(msg)

    @filter.command("lm提示词", alias={"lmc", "lm详情", "lmps"})
    async def prompt_details(self, event: AstrMessageEvent, trigger_word: str):
        """获取提示词详情字符串"""
        if trigger_word not in self.prompt_dict:
            yield event.plain_result(f"❌ 未找到提示词：「{trigger_word}」")
            return

        params = self.prompt_dict[trigger_word]
        details = [f"📋 提示词详情：「{trigger_word}」"]
        details.append(params.get("prompt", ""))
        for key in PARAMS_LIST:
            if key in params:
                details.append(f"{key}: {params[key]}")
        if event.platform_meta.name == "aiocqhttp":
            from astrbot.api.message_components import Node, Nodes, Plain

            nodes = []
            for detail in details:
                nodes.append(
                    Node(
                        uin=event.get_sender_id(),
                        name=event.get_sender_name(),
                        content=[Plain(detail)],
                    )
                )
            yield event.chain_result([Nodes(nodes)])
        else:
            yield event.plain_result("\n".join(details))

    @filter.command("lm删除", alias={"lmd"})
    async def del_prompt_command(self, event: AstrMessageEvent, trigger_word: str = ""):
        """lm删除 <触发词>"""
        if not self.is_global_admin(event):
            logger.info(
                f"用户 {event.get_sender_id()} 试图执行管理员命令 lm删除，权限不足"
            )
            return

        if not trigger_word:
            yield event.plain_result("❌ 格式错误：lm删除 (触发词)")
            return

        if trigger_word not in self.prompt_dict:
            yield event.plain_result(f"❌ 未找到提示词：「{trigger_word}」")
            return

        # 从提示词列表中找出对应项进行更新
        for i, v in enumerate(self.prompt_list):
            cmd, _, prompt_str = v.strip().partition(" ")
            if cmd == trigger_word:
                del self.prompt_list[i]
                self.init_prompts()
                self.conf.save_config()
                yield event.plain_result(f"🗑️ 已删除提示词：「{trigger_word}」")
                return
            # 处理多触发词
            if cmd.startswith("[") and cmd.endswith("]"):
                # 移除括号并按逗号分割
                cmd_list = cmd[1:-1].split(",")
                if trigger_word not in cmd_list:
                    continue

                yield event.plain_result(
                    "⚠️ 检测到该提示词为多触发词配置，请选择删除方案\nA. 单独删除该触发词\nB. 删除该多触发词\nC. 取消操作"
                )

                # 删除多触发词时，进行二次确认
                @session_waiter(timeout=30, record_history_chains=False)  # type: ignore
                async def waiter(
                    controller: SessionController, event: AstrMessageEvent
                ):
                    # 先鉴权
                    if not self.is_global_admin(event):
                        logger.info(
                            f"用户 {event.get_sender_id()} 试图执行管理员命令 lm删除，权限不足"
                        )
                        return

                    # 获取用户回复内容
                    reply_content = event.message_str.strip().upper()
                    if reply_content not in ["A", "B", "C"]:
                        await event.send(
                            event.plain_result("❌ 请输入有效的选项：A、B 或 C。")
                        )
                        return

                    if reply_content == "C":
                        await event.send(event.plain_result("🍌 操作已取消。"))
                        controller.stop()
                        return
                    if reply_content == "B":
                        # 删除整个多触发词配置
                        del self.prompt_list[i]
                        await event.send(
                            event.plain_result(f"🗑️ 已删除多触发提示词：{cmd}")
                        )
                        self.conf.save_config()
                        controller.stop()
                        return
                    if reply_content == "A":
                        # 将这个提示词从多触发提示词中移除
                        cmd_list.remove(trigger_word)
                        # 重新构建提示词字符串
                        if len(cmd_list) == 1:
                            # 仅剩一个触发词，改为单触发词形式
                            new_config_item = f"{cmd_list[0]} {prompt_str}"
                        else:
                            new_cmd = "[" + ",".join(cmd_list) + "]"
                            new_config_item = f"{new_cmd} {prompt_str}"
                        self.prompt_list[i] = new_config_item
                        # 最后更新字典
                        del self.prompt_dict[trigger_word]
                        # 更新内存字典
                        self.init_prompts()
                        await event.send(
                            event.plain_result(
                                f"🗑️ 已从多触发提示词中移除：「{trigger_word}」"
                            )
                        )
                        self.conf.save_config()
                        controller.stop()
                        return

                try:
                    await waiter(event)
                except TimeoutError as _:
                    yield event.plain_result("❌ 超时了，操作已取消！")
                except Exception as e:
                    logger.error(f"大香蕉删除提示词出现错误: {e}", exc_info=True)
                    yield event.plain_result("❌ 处理时发生了一个内部错误。")
                finally:
                    event.stop_event()
        else:
            logger.error(
                f"提示词列表和提示词字典不一致，未找到提示词：「{trigger_word}」"
            )
            yield event.plain_result(f"❌ 未找到提示词：「{trigger_word}」")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_message(self, event: AstrMessageEvent):
        """绘图命令消息入口"""

        # 取出所有 Plain 类型的组件拼接成纯文本内容
        plain_components = [
            comp for comp in event.get_messages() if isinstance(comp, Comp.Plain)
        ]

        # 拼接成一个字符串
        if plain_components:
            message_str = " ".join(comp.text for comp in plain_components).strip()
        else:
            message_str = event.message_str
        # 跳过空消息
        if not message_str:
            return

        # 先处理前缀
        matched_prefix = False
        for prefix in self.prefix_list:
            if message_str.startswith(prefix):
                message_str = message_str.removeprefix(prefix).lstrip()
                matched_prefix = True
                break

        # 若未@机器人且未开启混合模式，且配置了前缀列表但消息未匹配到任何前缀，则跳过处理
        if (
            not event.is_at_or_wake_command
            and not self.coexist_enabled
            and self.prefix_list
            and not matched_prefix
        ):
            return

        cmd = message_str.split(" ", 1)[0]
        # 检查命令是否在提示词配置中
        if cmd not in self.prompt_dict:
            return

        # 群白名单判断
        if (
            self.group_whitelist_enabled
            and event.unified_msg_origin not in self.group_whitelist
        ):
            logger.info(f"群 {event.unified_msg_origin} 不在白名单内，跳过处理")
            return

        # 用户白名单判断
        if (
            self.user_whitelist_enabled
            and event.get_sender_id() not in self.user_whitelist
        ):
            logger.info(f"用户 {event.get_sender_id()} 不在白名单内，跳过处理")
            return

        # 获取提示词配置 (使用 .copy() 防止修改污染全局预设)
        params = self.prompt_dict.get(cmd, {}).copy()
        params["__trigger_cmd__"] = cmd
        # 先从预设提示词参数字典字典中取出提示词
        preset_prompt = params.get("prompt", "{{user_text}}")

        _, user_params = self.parsing_prompt_params(message_str)
        user_overrode_min_images = "min_images" in user_params
        user_overrode_model = "model" in user_params
        params["__user_overrode_model__"] = user_overrode_model
        preset_name = user_params.pop("preset", None)
        user_prompt = user_params.get("prompt", "anything").strip()

        if preset_name:
            preset_key = str(preset_name).strip().strip(",，")
            preset_params = self.prompt_dict.get(preset_key, None)
            if not preset_params and preset_key:
                for k, v in self.prompt_dict.items():
                    if isinstance(k, str) and k.strip().strip(",，") == preset_key:
                        preset_params = v
                        break
            if not preset_params:
                yield event.plain_result(f"❌ 未找到预设提示词：「{preset_name}」")
                return

            selected_params = preset_params.copy()
            selected_prompt = selected_params.get("prompt", "{{user_text}}")

            if (
                selected_params.get("preset_append", self.common_config.preset_append)
                and "{{user_text}}" not in selected_prompt
            ):
                selected_prompt += " {{user_text}}"

            if "{{user_text}}" in selected_prompt:
                final_prompt = selected_prompt.replace("{{user_text}}", user_prompt)
            else:
                final_prompt = selected_prompt

            selected_params.pop("prompt", None)
            if "__model_name__" in params:
                selected_params.pop("__model_name__", None)
            params.update(selected_params)
            params.update({k: v for k, v in user_params.items() if k != "prompt"})
            params["prompt"] = final_prompt
        else:
            params.update({k: v for k, v in user_params.items() if k != "prompt"})

        if not user_overrode_model:
            selected_model = self.user_selected_provider_model.get(event.get_sender_id())
            if selected_model and (preset_name is not None or "__model_name__" not in params):
                params.setdefault("model", selected_model)

        # 处理预设提示词补充参数preset_append
        if (
            params.get("preset_append", self.common_config.preset_append)
            and "{{user_text}}" not in preset_prompt
        ):
            preset_prompt += " {{user_text}}"

        # 检查预设提示词中是否包含动态参数占位符
        if not preset_name and "{{user_text}}" in preset_prompt:
            new_prompt = preset_prompt.replace("{{user_text}}", user_prompt)
            params["prompt"] = new_prompt

        if cmd in {"iv1", "iv2"} and not user_overrode_min_images:
            params["min_images"] = 2

        is_nanobanana = params.get("__model_name__") == "nano-banana" or cmd in {
            "bnn",
            "bnt",
            "bna",
        }
        if is_nanobanana:
            if (
                self.nanobanana_group_whitelist_enabled
                and event.unified_msg_origin not in self.nanobanana_group_whitelist
            ):
                logger.info(
                    f"群 {event.unified_msg_origin} 不在 nano-banana 白名单内，跳过处理"
                )
                return
            if (
                self.nanobanana_user_whitelist_enabled
                and event.get_sender_id() not in self.nanobanana_user_whitelist
            ):
                logger.info(
                    f"用户 {event.get_sender_id()} 不在 nano-banana 白名单内，跳过处理"
                )
                return

        if cmd == "反推":
            min_required_images = params.get("min_images", self.prompt_config.min_images)
            try:
                min_required_images = int(min_required_images)
            except Exception:
                min_required_images = 1

            content = await self._image_to_prompt(
                event=event,
                prompt=str(params.get("prompt", "详细描述这张图片")),
                min_required_images=min_required_images,
            )
            yield event.chain_result(
                [
                    Comp.Reply(id=event.message_obj.message_id),
                    Comp.Plain(content),
                ]
            )
            return

        # 处理收集模式
        image_urls = []
        if params.get("gather_mode", self.prompt_config.gather_mode):
            # 记录操作员账号
            operator_id = event.get_sender_id()
            # 取消标记
            is_cancel = False
            yield event.plain_result(f"""📝 绘图收集模式已启用：
文本：{params["prompt"]}
图片：{len(image_urls)} 张

💡 继续发送图片或文本，或者：
• 发送「开始」开始生成
• 发送「取消」取消操作
• 60 秒内有效
""")

            @session_waiter(timeout=60, record_history_chains=False)  # type: ignore
            async def waiter(controller: SessionController, event: AstrMessageEvent):
                nonlocal is_cancel
                # 判断消息来源是否是同一用户
                if event.get_sender_id() != operator_id:
                    return

                if event.message_str.strip() == "取消":
                    is_cancel = True
                    await event.send(event.plain_result("✅ 操作已取消。"))
                    controller.stop()
                    return
                if event.message_str.strip() == "开始":
                    controller.stop()
                    return
                # 开始收集文本和图片
                for comp in event.get_messages():
                    if isinstance(comp, Comp.Plain) and comp.text:
                        # 追加文本到提示词
                        params["prompt"] += " " + comp.text.strip()
                    elif isinstance(comp, Comp.Image) and comp.url:
                        image_urls.append(comp.url)
                    elif (
                        isinstance(comp, Comp.File)
                        and comp.url
                        and comp.url.startswith("http")
                        and comp.url.lower().endswith(SUPPORTED_FILE_FORMATS)
                    ):
                        image_urls.append(comp.url)
                await event.send(
                    event.plain_result(f"""📝 绘图追加模式已收集内容：
文本：{params["prompt"]}
图片：{len(image_urls)} 张

💡 继续发送图片或文本，或者：
• 发送「开始」开始生成
• 发送「取消」取消操作
• 60 秒内有效
""")
                )
                controller.keep(timeout=60, reset_timeout=True)

            try:
                await waiter(event)
            except TimeoutError as _:
                yield event.plain_result("❌ 超时了，操作已取消！")
                return
            except Exception as e:
                logger.error(f"绘图提示词追加模式出现错误: {e}", exc_info=True)
                yield event.plain_result("❌ 处理时发生了一个内部错误。")
                return
            finally:
                if is_cancel:
                    event.stop_event()
                    return

        logger.info(f"正在生成图片，提示词: {params['prompt'][:60]}")
        logger.debug(
            f"生成图片应用参数: { {k: v for k, v in params.items() if k != 'prompt'} }"
        )
        task = asyncio.create_task(
            self._run_job_with_limit(event, params, image_urls=image_urls)
        )
        task_id = event.message_obj.message_id
        self.running_tasks[task_id] = task
        task_temp_dir = self.temp_dir / str(task_id)
        os.makedirs(task_temp_dir, exist_ok=True)

        try:
            results, err_msg = await task
            if not results or err_msg:
                if isinstance(err_msg, str) and err_msg.strip().startswith("图片生成失败"):
                    err_text = err_msg.strip()
                else:
                    err_text = f"图片生成失败：{err_msg}"
                yield event.chain_result(
                    [
                        Comp.Reply(id=event.message_obj.message_id),
                        Comp.Plain(f"❌ {err_text}"),
                    ]
                )
                return

            # 组装消息链
            os.makedirs(task_temp_dir, exist_ok=True)
            msg_chain = self.build_message_chain(event, results, task_temp_dir)

            yield event.chain_result(msg_chain)
        except asyncio.CancelledError:
            logger.info(f"{task_id} 任务被取消")
            return
        finally:
            self.running_tasks.pop(task_id, None)
            clear_cache(task_temp_dir)

    async def job(
        self,
        event: AstrMessageEvent,
        params: dict,
        image_urls: list[str] | None = None,
        referer_id: list[str] | None = None,
        is_llm_tool: bool = False,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        """负责参数处理、调度提供商、保存图片等逻辑，返回图片b64列表或错误信息"""
        # 收集图片URL，后面统一处理
        if image_urls is None:
            image_urls = []

        if referer_id is None:
            referer_id = []
        # 小标记，用于优化At头像。当At对象是被引用消息的发送者时，跳过一次。
        skipped_at_qq = False
        reply_sender_id = ""
        for comp in event.get_messages():
            if isinstance(comp, Comp.Reply):
                reply_urls, reply_sender_id = await self._collect_reply_image_urls(event, comp)
                if reply_urls:
                    image_urls.extend(reply_urls)
            # 处理At对象的QQ头像（对于艾特机器人的问题，还没有特别好的解决方案）
            elif (
                isinstance(comp, Comp.At)
                and comp.qq
                and event.platform_meta.name == "aiocqhttp"
            ):
                qq = str(comp.qq)
                self_id = event.get_self_id()
                if not skipped_at_qq and (
                    # 如果At对象是被引用消息的发送者，跳过一次
                    (qq == reply_sender_id and self.preference_config.skip_quote_first)
                    or (
                        qq == self_id
                        and event.is_at_or_wake_command
                        and self.preference_config.skip_at_first
                    )  # 通过At唤醒机器人，跳过一次
                    or (
                        qq == self_id
                        and self.preference_config.skip_llm_at_first
                        and is_llm_tool
                    )  # 通过At唤醒机器人，且是函数调用工具，跳过一次
                ):
                    skipped_at_qq = True
                    continue
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

        q_value = params.get("q")
        if q_value and event.platform_meta.name == "aiocqhttp":
            for qq in re.findall(r"\d+", str(q_value)):
                build_url = f"https://q.qlogo.cn/g?b=qq&s=0&nk={qq}"
                if build_url not in image_urls:
                    image_urls.append(build_url)

        # 处理referer_id参数，获取指定用户头像
        if is_llm_tool and referer_id and event.platform_meta.name == "aiocqhttp":
            for target_id in referer_id:
                target_id = target_id.strip()
                if target_id:
                    build_url = f"https://q.qlogo.cn/g?b=qq&s=0&nk={target_id}"
                    if build_url not in image_urls:
                        image_urls.append(
                            f"https://q.qlogo.cn/g?b=qq&s=0&nk={target_id}"
                        )

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

        min_required_images = params.get("min_images", self.prompt_config.min_images)
        max_allowed_images = params.get("max_images", self.prompt_config.max_images)
        # 如果图片数量不满足最小要求，且消息平台是Aiocqhttp，取消息发送者头像作为参考图片
        if (
            len(image_urls) < min_required_images
            and int(min_required_images or 0) == 1
            and event.platform_meta.name == "aiocqhttp"
        ):
            image_urls.append(
                f"https://q.qlogo.cn/g?b=qq&s=0&nk={event.get_sender_id()}"
            )

        # 图片b64列表，每个元素是 (mime_type, b64_data) 元组
        image_b64_list = []
        # 处理 refer_images 参数
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
        # 图片去重
        image_urls = list(dict.fromkeys(image_urls))
        # 判断图片数量是否满足最小要求
        if len(image_urls) + len(image_b64_list) < min_required_images:
            warn_msg = f"图片数量不足，最少需要 {min_required_images} 张图片，当前仅 {len(image_urls) + len(image_b64_list)} 张"
            logger.warning(warn_msg)
            return None, warn_msg

        # 检查图片数量是否超过最大允许数量，不超过则可从url中下载图片
        append_count = max_allowed_images - len(image_b64_list)
        if append_count > 0 and image_urls:
            # 取前n张图片，下载并转换为Base64，追加到b64图片列表
            if len(image_b64_list) + len(image_urls) > max_allowed_images:
                logger.warning(
                    f"参考图片数量超过或等于最大图片数量，将只使用前 {max_allowed_images} 张参考图片"
                )
            fetched = await self.downloader.fetch_images(image_urls[:append_count])
            if fetched:
                image_b64_list.extend(fetched)

            # 如果 min_required_images 为 0，列表为空是允许的
            if not image_b64_list and min_required_images > 0:
                logger.error("全部参考图片下载失败")
                return None, "全部参考图片下载失败"

        # 发送绘图中提示
        await event.send(MessageChain().message("🎨 在画了，请稍等一会..."))

        # 调度提供商生成图片
        images_result, err = await self._dispatch(
            params=params, image_b64_list=image_b64_list
        )

        # 再次检查图片结果是否为空
        valid_results = [(mime, b64) for mime, b64 in (images_result or []) if b64]

        if not valid_results:
            if not err:
                err = "图片生成失败：响应中未包含图片数据"
                logger.error(err)
            return None, err

        # 保存图片到本地
        if self.save_images:
            save_images(valid_results, self.save_dir)

        return valid_results, None

    async def _run_job_with_limit(
        self,
        event: AstrMessageEvent,
        params: dict,
        image_urls: list[str] | None = None,
        referer_id: list[str] | None = None,
        is_llm_tool: bool = False,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        if self.is_global_admin(event):
            return await self.job(
                event=event,
                params=params,
                image_urls=image_urls,
                referer_id=referer_id,
                is_llm_tool=is_llm_tool,
            )

        sem = self.job_semaphore
        if sem is None:
            return await self.job(
                event=event,
                params=params,
                image_urls=image_urls,
                referer_id=referer_id,
                is_llm_tool=is_llm_tool,
            )

        await sem.acquire()
        try:
            return await self.job(
                event=event,
                params=params,
                image_urls=image_urls,
                referer_id=referer_id,
                is_llm_tool=is_llm_tool,
            )
        finally:
            sem.release()

    async def _dispatch(
        self,
        params: dict,
        image_b64_list: list[tuple[str, str]] = [],
        allow_fallback: bool = True,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        """提供商调度器"""
        err = None
        
        # 1. 确定使用的模型
        model_name = params.get("__model_name__")
        target_model = None
        requested_provider_model = (params.get("model") or "").strip()
        
        if model_name:
            for model in self.models:
                if model.name == model_name:
                    target_model = model
                    break
        
        # 如果未找到指定模型（或未指定），使用第一个启用的模型作为默认
        if not target_model and self.models:
            if requested_provider_model:
                for model in self.models:
                    if any(
                        (p.model or "").strip() == requested_provider_model
                        for p in model.providers
                    ):
                        target_model = model
                        break
            if not target_model:
                preferred_provider_model = "gemini-3.0-pro-image-portrait"
                for model in self.models:
                    if any(
                        (p.model or "").strip() == preferred_provider_model
                        for p in model.providers
                    ):
                        target_model = model
                        break
            if not target_model:
                target_model = self.models[0]
            
        if not target_model:
            return None, "未配置任何模型。"

        # 2. 获取该模型的提供商列表
        all_candidate_providers = target_model.providers
        candidate_providers = all_candidate_providers
        
        # 3. 筛选提供商 (如果 params 指定了 provider)
        target_provider_name = params.get("provider")
        
        filtered_by_model_applied = False
        if requested_provider_model and not target_provider_name:
            filtered_by_model = [
                p
                for p in candidate_providers
                if (p.model or "").strip() == requested_provider_model
            ]
            if filtered_by_model:
                candidate_providers = filtered_by_model
                filtered_by_model_applied = True

        if target_provider_name:
            # 尝试匹配 name
            filtered = [
                p for p in candidate_providers 
                if p.name == target_provider_name
            ]
            
            if filtered:
                candidate_providers = filtered
            else:
                logger.warning(f"在模型 {target_model.name} 中未找到指定提供商: {target_provider_name}，将使用默认顺序")

        if not candidate_providers:
             return None, f"模型 {target_model.name} 未配置有效的提供商。"

        # 调度提供商
        for i, provider in enumerate(candidate_providers):
            if provider.api_type not in self.provider_map:
                logger.warning(f"提供商类型 {provider.api_type} 未初始化，跳过")
                continue
                
            images_result, err = await self.provider_map[
                provider.api_type
            ].generate_images(
                provider_config=provider,
                params=params,
                image_b64_list=image_b64_list,
            )
            if images_result:
                logger.info(f"模型 {target_model.name} - {provider.name} 图片生成成功")
                return images_result, None
            
            # 如果不是最后一个提供商，且配置了重试逻辑（隐含在列表顺序中），则继续
            if i < len(candidate_providers) - 1:
                logger.warning(f"{provider.name} 生成图片失败，尝试使用下一个提供商...")

        if (
            allow_fallback
            and filtered_by_model_applied
            and not params.get("__user_overrode_model__", False)
        ):
            remaining = [p for p in all_candidate_providers if p not in candidate_providers]
            if remaining:
                fallback_params = params.copy()
                fallback_params.pop("model", None)
                fallback_params.pop("__user_overrode_model__", None)
                for i, provider in enumerate(remaining):
                    if provider.api_type not in self.provider_map:
                        continue
                    images_result, err = await self.provider_map[
                        provider.api_type
                    ].generate_images(
                        provider_config=provider,
                        params=fallback_params,
                        image_b64_list=image_b64_list,
                    )
                    if images_result:
                        logger.info(f"模型 {target_model.name} - {provider.name} 图片生成成功")
                        return images_result, None

        if allow_fallback:
            fallback_probe_model = (params.get("model") or candidate_providers[0].model or "").strip()
            is_video_model = fallback_probe_model.startswith("veo_")
            is_flow2api = any((p.name or "").strip() == "flow2api" for p in candidate_providers)
            html_like_error = isinstance(err, str) and (
                "HTML" in err
                or "响应内容格式错误" in err
                or "媒体下载失败" in err
                or "API 地址不存在" in err
                or "状态码 401" in err
                or "状态码 403" in err
                or "状态码 404" in err
            )
            if is_flow2api and not is_video_model and html_like_error:
                fallback_model = next(
                    (m for m in self.models if m.enabled and m.name in {"nano-banana", "Z-Image-Turbo"}),
                    None,
                )
                if not fallback_model:
                    fallback_model = next(
                        (
                            m
                            for m in self.models
                            if m.enabled
                            and m.name != target_model.name
                            and any((p.name or "").strip() != "flow2api" for p in m.providers)
                        ),
                        None,
                    )
                if fallback_model:
                    fallback_params = params.copy()
                    fallback_params["__model_name__"] = fallback_model.name
                    fallback_params.pop("provider", None)
                    images_result, fallback_err = await self._dispatch(
                        params=fallback_params,
                        image_b64_list=image_b64_list,
                        allow_fallback=False,
                    )
                    if images_result:
                        return images_result, None
                    if fallback_err:
                        err = fallback_err

        # 处理错误信息
        if not err:
            err = "所有提供商均生成失败，请检查日志。"
        return None, err

    def build_message_chain(
        self,
        event: AstrMessageEvent,
        results: list[tuple[str, str]],
        temp_dir=None,
    ) -> list[BaseMessageComponent]:
        """构建消息链"""
        msg_chain: list[BaseMessageComponent] = [
            Comp.Reply(id=event.message_obj.message_id)
        ]

        if temp_dir is None:
            temp_dir = self.temp_dir

        image_items: list[tuple[str, str]] = []
        file_items: list[tuple[str, str]] = []

        telegram_force_file = event.platform_meta.name == "telegram" and any(
            (mime or "").startswith("image/") and b64 and len(b64) > MAX_SIZE_B64_LEN
            for mime, b64 in results
        )

        for mime, b64 in results:
            mime = (mime or "").strip()
            if not b64:
                continue
            if telegram_force_file:
                file_items.append((mime or "application/octet-stream", b64))
                continue
            if mime.startswith("image/"):
                image_items.append((mime, b64))
            else:
                file_items.append((mime or "application/octet-stream", b64))

        if file_items:
            save_results = save_images(file_items, temp_dir)
            for name_, path_ in save_results:
                msg_chain.append(Comp.File(name=name_, file=str(path_)))

        if image_items:
            msg_chain.extend(Comp.Image.fromBase64(b64) for _, b64 in image_items)

        return msg_chain

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
        # 取消所有生成任务
        for task in list(self.running_tasks.values()):
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        self.running_tasks.clear()
        # 清理网络客户端会话
        await self.http_manager.close_session()
        # 卸载函数调用工具
        remove_tools(self.context)
