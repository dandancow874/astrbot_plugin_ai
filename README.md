# AstrBot Plugin AI
[astrbot_plugin_big_banana](https://github.com/sukafon/astrbot_plugin_big_banana)
拿这个插件改的,增加了几个生图模型

AstrBot 图片生成插件，支持多种 AI 绘图模型。

## 功能特性

- 支持多种 AI 绘图模型：
  - **nano-banana-2** (触发词: `bt1`, `bt2`, `bt3`)
  - **nano-banana-pro** (触发词: `bp1`, `bp2`, `bp3`)
  - **Z-Image-Turbo** (触发词: `zimg`)
  - **Qwen-Image-Edit** (触发词: `edit`) - 图片编辑
  - **Midjourney V7** (触发词: `mj`, `mj2`)
  - **Niji 7** (触发词: `nj`, `nj2`) - 动漫风格
  - **qwen-image-edit-2511** (触发词: `edit`)
  - **Image-to-Prompt** (触发词: `反推`) - 图片反推提示词


- 触发词带2的都是图生图

- 支持自定义预设提示词
- 支持 LLM 函数调用工具
- 支持多账号轮询和重试
- 支持代理配置

## 使用方法

### 基本用法

在聊天中发送：`触发词 提示词`

例如：
```
mj 一只可爱的猫咪
nj 动漫风格少女
bt1 beautiful sunset landscape
zimg cyberpunk city
```

### Midjourney 使用说明

- `mj 提示词` - 使用 Midjourney V7 模型生成图片
- `nj 提示词` - 使用 Niji 7 模型生成动漫风格图片

Midjourney 返回的四宫格图片会自动裁切成 4 张独立图片发送。

### 垫图功能

使用图生图模式的预设词,回复图片并使用绘图命令，图片会作为参考图传递给 AI 模型,没有传图片会使用头像做为参考

### 预设提示词

可在配置文件中设置预设提示词，支持以下占位符：
- `{{user_text}}` - 用户输入的文本

格式：`触发词 提示词 --参数1 参数值1 --参数2 参数值2`

## 配置说明

在 AstrBot 管理面板中配置以下选项：

### 模型配置

各模型配置项包括：
- `enabled` - 是否启用
- `api_url` - API 地址
- `api_key` - API Key
- `tls_verify` - TLS 证书验证
- `impersonate` - TLS/UA 指纹伪装

### Midjourney 配置

在 `midjourney_config` 中配置：
- 默认 API 地址: `https://ai.t8star.cn`
- 填入你的 API Key

### 提示词默认参数

- `min_images` - 最小输入图片数量
- `max_images` - 最大输入图片数量
- `aspect_ratio` - 宽高比
- `image_size` - 分辨率 (1K, 2K, 4K)
- `google_search` - 启用谷歌搜索工具

### 常规配置

- `timeout` - 超时时间（秒）
- `proxy` - HTTP 代理
- `max_retry` - 最大重试次数

## 安装

在 AstrBot 插件市场搜索 `astrbot_plugin_ai` 安装，或直接输入仓库地址：

```
https://github.com/dandancow874/astrbot_plugin_ai
```

## 依赖

- Pillow (用于 Midjourney 四宫格图片裁切)

## 许可证

MIT License