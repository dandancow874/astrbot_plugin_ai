"""
管理员命令处理器 - 白名单管理等
"""

from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core.message.components import At, Plain, Reply

if TYPE_CHECKING:
    from .main import BigBanana


class AdminCommands:
    """管理员命令类"""

    def __init__(self, plugin: "BigBanana"):
        self.plugin = plugin
        self.conf = plugin.conf

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        """检查是否为全局管理员"""
        return event.get_sender_id() in self.conf.get("admin_id", [])

    @filter.command("lm 白名单添加", alias={"lmawl"})
    async def add_whitelist(
        self, event: AstrMessageEvent, unified_msg_origin: str, ids: str
    ):
        """添加白名单"""
        if not self.is_global_admin(event):
            return

        if not ids:
            yield event.plain_result("❌ 格式错误：lm 白名单添加 (id1,id2,id3...)")
            return

        id_list = [x.strip() for x in ids.replace("，", ",").split(",") if x.strip()]
        if not id_list:
            yield event.plain_result("❌ 请输入有效的 ID 列表")
            return

        # 判断是群组还是用户白名单
        whitelist_key = "whitelist"
        enabled_key = "enabled"
        if "user" in unified_msg_origin.lower():
            whitelist_key = "user_whitelist"
            enabled_key = "user_enabled"

        whitelist = self.conf.get("whitelist_config", {}).get(whitelist_key, [])
        if not isinstance(whitelist, list):
            whitelist = []

        added = []
        skipped = []
        for id_ in id_list:
            if id_ not in whitelist:
                whitelist.append(id_)
                added.append(id_)
            else:
                skipped.append(id_)

        # 更新配置
        if "whitelist_config" not in self.conf:
            self.conf["whitelist_config"] = {}
        self.conf["whitelist_config"][whitelist_key] = whitelist
        self.conf["whitelist_config"][enabled_key] = True
        self.conf.save_config()

        # 更新插件缓存
        if hasattr(self.plugin, "group_whitelist"):
            self.plugin.group_whitelist = set(whitelist)
        if hasattr(self.plugin, "user_whitelist"):
            self.plugin.user_whitelist = set(
                self.conf.get("whitelist_config", {}).get("user_whitelist", [])
            )

        msg = f"✅ 已添加 {len(added)} 个 ID 到白名单"
        if skipped:
            msg += f"\n跳过 {len(skipped)} 个已存在的 ID"
        yield event.plain_result(msg)

    @filter.command("lm 白名单删除", alias={"lmdwl"})
    async def del_whitelist(
        self, event: AstrMessageEvent, unified_msg_origin: str, ids: str
    ):
        """删除白名单"""
        if not self.is_global_admin(event):
            return

        if not ids:
            yield event.plain_result("❌ 格式错误：lm 白名单删除 (id1,id2,id3...)")
            return

        id_list = [x.strip() for x in ids.replace("，", ",").split(",") if x.strip()]
        if not id_list:
            yield event.plain_result("❌ 请输入有效的 ID 列表")
            return

        # 判断是群组还是用户白名单
        whitelist_key = "whitelist"
        if "user" in unified_msg_origin.lower():
            whitelist_key = "user_whitelist"

        whitelist = self.conf.get("whitelist_config", {}).get(whitelist_key, [])
        if not isinstance(whitelist, list):
            whitelist = []

        removed = []
        not_found = []
        for id_ in id_list:
            if id_ in whitelist:
                whitelist.remove(id_)
                removed.append(id_)
            else:
                not_found.append(id_)

        # 更新配置
        if "whitelist_config" not in self.conf:
            self.conf["whitelist_config"] = {}
        self.conf["whitelist_config"][whitelist_key] = whitelist
        self.conf.save_config()

        # 更新插件缓存
        if hasattr(self.plugin, "group_whitelist"):
            self.plugin.group_whitelist = set(whitelist)
        if hasattr(self.plugin, "user_whitelist"):
            self.plugin.user_whitelist = set(
                self.conf.get("whitelist_config", {}).get("user_whitelist", [])
            )

        msg = f"🗑️ 已从白名单删除 {len(removed)} 个 ID"
        if not_found:
            msg += f"\n未找到 {len(not_found)} 个 ID"
        yield event.plain_result(msg)
