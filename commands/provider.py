"""
提供商命令处理器 - 模型切换等
"""

from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent, filter

if TYPE_CHECKING:
    from .main import BigBanana


class ProviderCommands:
    """提供商命令类"""

    def __init__(self, plugin: "BigBanana"):
        self.plugin = plugin
        self.provider_service = plugin.provider_service
        self.conf = plugin.conf

    @filter.command("模型切换", alias={"使用模型切换"})
    async def switch_model(self, event: AstrMessageEvent, model_name: str):
        """切换用户选择的模型"""
        if not model_name:
            # 显示当前选择的模型
            user_model = self.provider_service.get_user_model(event.get_sender_id())
            if user_model:
                yield event.plain_result(
                    f"📋 当前选择的模型：{user_model.name}\n"
                    f"触发词：{', '.join(user_model.triggers)}\n\n"
                    f"💡 使用：模型切换 <模型名称> 来切换模型"
                )
            else:
                yield event.plain_result(
                    "📋 当前未选择模型，将使用默认配置\n\n"
                    "💡 使用：模型切换 <模型名称> 来选择模型"
                )
            return

        # 切换模型
        success, msg = self.provider_service.switch_user_model(
            event.get_sender_id(), model_name
        )
        yield event.plain_result(msg)

    @filter.command("模型列表", alias={"查看模型"})
    async def list_models(self, event: AstrMessageEvent):
        """列出所有可用模型"""
        models = self.provider_service.list_models()
        if not models:
            yield event.plain_result("❌ 未配置任何模型")
            return

        user_model = self.provider_service.get_user_model(event.get_sender_id())
        current_model_name = user_model.name if user_model else None

        lines = ["📋 可用模型列表：\n"]
        for i, m in enumerate(models, 1):
            status = "✅" if m["enabled"] else "❌"
            current = "👉 (当前)" if m["name"] == current_model_name else ""
            triggers = ", ".join(m["triggers"]) if m["triggers"] else "无"
            providers = ", ".join([p["name"] for p in m["providers"]])

            lines.append(
                f"{i}. {status} {m['name']} {current}\n"
                f"   触发词：{triggers}\n"
                f"   提供商：{providers}\n"
            )

        lines.append("\n💡 使用：模型切换 <模型名称> 来切换模型")
        yield event.plain_result("\n".join(lines))
