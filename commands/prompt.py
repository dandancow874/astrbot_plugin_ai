"""
提示词命令处理器 - 提示词管理等
"""

from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent, filter

if TYPE_CHECKING:
    from .main import BigBanana


class PromptCommands:
    """提示词命令类"""

    def __init__(self, plugin: "BigBanana"):
        self.plugin = plugin
        self.prompt_service = plugin.prompt_service
        self.conf = plugin.conf

    @filter.command("提示词列表", alias={"查看提示词"})
    async def list_prompts(self, event: AstrMessageEvent):
        """列出所有提示词"""
        prompts = self.prompt_service.list_prompts()
        if not prompts:
            yield event.plain_result("❌ 未配置任何提示词")
            return

        lines = ["📋 提示词列表：\n"]
        for i, p in enumerate(prompts[:20], 1):  # 限制显示 20 个
            model_tag = f" [{p['model']}]" if p.get("model") else ""
            lines.append(f"{i}. {p['cmd']}{model_tag}")

        if len(prompts) > 20:
            lines.append(f"\n... 还有 {len(prompts) - 20} 个提示词")

        lines.append("\n💡 使用：lm 添加/删除 <触发词> 来管理提示词")
        yield event.plain_result("\n".join(lines))
