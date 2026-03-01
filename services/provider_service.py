"""
提供商服务 - 负责提供商管理、调度和模型选择
"""

from typing import TYPE_CHECKING

from astrbot.api import logger

from .core import BaseProvider
from .core.data import ModelConfig, ProviderConfig

if TYPE_CHECKING:
    from .main import BigBanana


class ProviderService:
    """提供商服务类"""

    def __init__(self, plugin: "BigBanana"):
        self.plugin = plugin
        self.models: list[ModelConfig] = plugin.models
        self.provider_map: dict[str, BaseProvider] = plugin.provider_map
        self.conf = plugin.conf

    def get_model_by_trigger(self, trigger: str) -> ModelConfig | None:
        """根据触发词获取模型配置"""
        for model in self.models:
            if trigger in model.triggers:
                return model
        return None

    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """根据模型名称获取配置"""
        for model in self.models:
            if model.name == name:
                return model
        return None

    def select_provider(
        self, model_config: ModelConfig, preferred_api_type: str | None = None
    ) -> ProviderConfig | None:
        """
        选择提供商

        Args:
            model_config: 模型配置
            preferred_api_type: 首选 API 类型（可选）

        Returns:
            选中的提供商配置，如果没有可用提供商则返回 None
        """
        # 按优先级排序
        sorted_providers = sorted(model_config.providers, key=lambda x: x.priority)

        # 如果有首选类型，优先选择
        if preferred_api_type:
            for p in sorted_providers:
                if p.enabled and p.api_type == preferred_api_type:
                    return p

        # 否则返回第一个启用的提供商
        for p in sorted_providers:
            if p.enabled and p.api_type in self.provider_map:
                return p

        return None

    def list_models(self, include_disabled: bool = False) -> list[dict]:
        """
        列出所有模型

        Args:
            include_disabled: 是否包含禁用的模型

        Returns:
            模型信息列表
        """
        result = []
        for model in self.models:
            if not include_disabled and not model.enabled:
                continue

            providers_info = []
            for p in model.providers:
                providers_info.append(
                    {
                        "name": p.name,
                        "api_type": p.api_type,
                        "priority": p.priority,
                        "enabled": p.enabled,
                    }
                )

            result.append(
                {
                    "name": model.name,
                    "triggers": model.triggers,
                    "enabled": model.enabled,
                    "providers": providers_info,
                }
            )
        return result

    def switch_user_model(self, user_id: str, model_name: str) -> tuple[bool, str]:
        """
        切换用户选择的模型

        Args:
            user_id: 用户 ID
            model_name: 模型名称

        Returns:
            (成功标志，消息)
        """
        model = self.get_model_by_name(model_name)
        if not model:
            return False, f"❌ 未找到模型：{model_name}"

        if not model.enabled:
            return False, f"❌ 模型 {model_name} 已禁用"

        # 保存用户选择
        if not hasattr(self.plugin, "user_selected_provider_model"):
            self.plugin.user_selected_provider_model = {}

        self.plugin.user_selected_provider_model[user_id] = model_name
        self.conf["user_selected_provider_model"] = (
            self.plugin.user_selected_provider_model
        )
        self.conf.save_config()

        return (
            True,
            f"✅ 已切换到模型：{model_name}（触发词：{', '.join(model.triggers)}）",
        )

    def get_user_model(self, user_id: str) -> ModelConfig | None:
        """获取用户选择的模型"""
        if not hasattr(self.plugin, "user_selected_provider_model"):
            return None

        selected = self.plugin.user_selected_provider_model.get(user_id)
        if not selected:
            return None

        return self.get_model_by_name(selected)

    def clear_user_model(self, user_id: str) -> bool:
        """清除用户选择的模型"""
        if not hasattr(self.plugin, "user_selected_provider_model"):
            return False

        if user_id in self.plugin.user_selected_provider_model:
            del self.plugin.user_selected_provider_model[user_id]
            self.conf["user_selected_provider_model"] = (
                self.plugin.user_selected_provider_model
            )
            self.conf.save_config()
            return True
        return False
