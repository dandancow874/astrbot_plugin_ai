# AstrBot AI 绘图插件

基于 AstrBot 的 AI 绘图插件，支持多个模型和提供商。

## ✨ 功能特性

- 🎨 支持多个 AI 绘图模型
- 🔌 多提供商支持（OpenAI、Gemini、ZImage 等）
- ⚡ 触发词快捷调用
- 🔄 智能重试机制
- 📦 支持图片收集模式

## 🚀 快速开始

### 1. 安装插件

将插件复制到 AstrBot 的 `plugins` 目录：

```bash
# 克隆或下载插件到 plugins 目录
```

### 2. 配置 API Key

在 AstrBot 配置界面配置 API Key。

### 3. 使用方式

```
nb2 画一只猫
nbp 画风景
模型列表
```

## 📋 可用触发词

| 触发词 | 模型 | 说明 |
|--------|------|------|
| `nb2` | nano-banana-2 | 快速绘图 |
| `nbp` | nano-banana-pro | 高质量绘图 |
| `zimg` | z-image-turbo | Gitee 绘图 |
| `kl` | FLUX.2-klein-9B | Flux 模型 |

## 🔧 故障排除

### 触发词不响应

1. 检查是否 @ 机器人
2. 检查白名单配置
3. 运行 `模型列表` 确认模型已启用

### 图片生成失败

1. 检查 API Key 是否有效
2. 检查网络连接
3. 查看日志获取详细错误信息

## 📝 配置说明

### 配置结构

```json
{
  "models": [
    {
      "name": "模型名称",
      "enabled": true,
      "triggers": ["触发词"],
      "providers": [
        {
          "name": "提供商",
          "api_url": "API 地址",
          "api_key": "API Key",
          "api_type": "OpenAI_Chat",
          "model": "模型名称"
        }
      ]
    }
  ]
}
```

## 🛠️ 开发

### 项目结构

```
astrbot_plugin_ai/
├── main.py              # 主入口
├── core/                # 核心模块
├── services/            # 服务层
├── commands/            # 命令层
└── _conf_schema.json   # 配置 Schema
```

## 📄 许可证

MIT License
