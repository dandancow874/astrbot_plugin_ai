"""
提示词服务 - 负责提示词解析、管理和注册
"""

import re
from typing import TYPE_CHECKING

from astrbot.api import logger

from .core.data import PARAMS_ALIAS_MAP, PARAMS_LIST

if TYPE_CHECKING:
    from .main import BigBanana


class PromptService:
    """提示词服务类"""

    def __init__(self, plugin: "BigBanana"):
        self.plugin = plugin
        self.prompt_list = plugin.prompt_list
        self.prompt_dict = plugin.prompt_dict
        self.conf = plugin.conf
        self.models = plugin.models

    def parse_prompt_params(self, prompt: str) -> tuple[list[str], dict]:
        """
        解析提示词中的参数

        Args:
            prompt: 完整的提示词字符串（包括命令和参数）

        Returns:
            (命令列表，参数字典)
        """
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
                    else:
                        params[key] = value
                else:
                    filtered.append(token)
            else:
                filtered.append(token)

        # 将过滤后的提示词拼接成字符串
        params["prompt"] = " ".join(filtered) if filtered else "{{user_text}}"

        return cmd_list, params

    def init_prompts(self) -> None:
        """初始化提示词配置"""
        # 预设提示词列表
        self.prompt_list = self.conf.get("prompt", [])
        self.prompt_dict = {}
        existing_cmds: set[str] = set()

        for item in self.prompt_list:
            cmd_list, params = self.parse_prompt_params(item)
            for cmd in cmd_list:
                existing_cmds.add(cmd)
                self.prompt_dict[cmd] = params

        # 固定提示词（自动补充）
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
            cmd_list, params = self.parse_prompt_params(prompt_line)
            self.prompt_list.append(prompt_line)
            updated_prompts = True
            for cmd in cmd_list:
                existing_cmds.add(cmd)
                self.prompt_dict[cmd] = params

        # 将模型触发词注册到 prompt_dict
        self._register_model_triggers(existing_cmds)

        if updated_prompts:
            self.conf["prompt"] = self.prompt_list
            self.conf.save_config()

    def _register_model_triggers(self, existing_cmds: set[str]) -> None:
        """将模型触发词注册到 prompt_dict"""
        for model in self.models:
            for trigger in model.triggers:
                if trigger not in self.prompt_dict:
                    # 新触发词，添加默认配置
                    self.prompt_dict[trigger] = {
                        "prompt": "{{user_text}}",
                        "__model_name__": model.name,
                    }
                else:
                    # 已存在（有预设提示词），补充模型信息
                    if "__model_name__" not in self.prompt_dict[trigger]:
                        self.prompt_dict[trigger]["__model_name__"] = model.name

                # 标记为已存在
                existing_cmds.add(trigger)

    def get_prompt(self, cmd: str) -> dict | None:
        """获取提示词配置"""
        return self.prompt_dict.get(cmd)

    def add_prompt(self, cmd: str, prompt_str: str) -> tuple[bool, str]:
        """
        添加提示词

        Args:
            cmd: 触发词
            prompt_str: 提示词内容

        Returns:
            (成功标志，消息)
        """
        if cmd in self.prompt_dict:
            return False, f"❌ 提示词已存在：{cmd}"

        cmd_list, params = self.parse_prompt_params(f"{cmd} {prompt_str}")
        self.prompt_list.append(f"{cmd} {prompt_str}")
        self.prompt_dict[cmd] = params

        self.conf["prompt"] = self.prompt_list
        self.conf.save_config()

        return True, f"✅ 已添加提示词：{cmd}"

    def remove_prompt(self, cmd: str) -> tuple[bool, str]:
        """
        删除提示词

        Args:
            cmd: 触发词

        Returns:
            (成功标志，消息)
        """
        if cmd not in self.prompt_dict:
            return False, f"❌ 未找到提示词：{cmd}"

        # 从列表中删除
        for i, v in enumerate(self.prompt_list):
            v_cmd = v.strip().split(" ", 1)[0]
            if v_cmd == cmd:
                del self.prompt_list[i]
                break

        # 从字典中删除
        del self.prompt_dict[cmd]

        self.conf["prompt"] = self.prompt_list
        self.conf.save_config()

        return True, f"🗑️ 已删除提示词：{cmd}"

    def list_prompts(self) -> list[dict]:
        """列出所有提示词"""
        result = []
        for cmd, params in self.prompt_dict.items():
            result.append(
                {
                    "cmd": cmd,
                    "prompt": params.get("prompt", "{{user_text}}"),
                    "model": params.get("__model_name__"),
                }
            )
        return result
