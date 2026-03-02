# AstrBot Plugin AI

[astrbot_plugin_big_banana](https://github.com/sukafon/astrbot_plugin_big_banana) 拿这个插件改的，增加了几个生图模型。

AstrBot 图片生成插件，支持多种 AI 绘图模型。

## 功能特性

- 支持多种 AI 绘图模型：
  - **nano-banana-2** (触发词: `bt1`, `bt2`, `bt3`)
  - **nano-banana-pro** (触发词: `bp1`, `bp2`, `bp3`)
  - **Z-Image-Turbo** (触发词: `zimg`)
  - **Qwen-Image-Edit** (触发词: `edit`) - 图片编辑
  - **Midjourney V7** (触发词: `mj`, `mj2`)
  - **Niji 7** (触发词: `nj`, `nj2`) - 动漫风格
  - **Image-to-Prompt** (触发词: `反推`) - 图片反推提示词

- 触发词带 `2` 的是图生图模式（需要垫图）
- 支持自定义预设提示词
- 支持 LLM 函数调用工具
- 支持多账号轮询和重试
- 支持代理配置
- 支持保存图片及 JSON 元数据

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

| 触发词 | 模式 | 说明 |
|--------|------|------|
| `mj` | 文生图 | 使用 Midjourney V7 生成图片 |
| `nj` | 文生图 | 使用 Niji 7 生成动漫风格图片 |
| `mj2` | 图生图 | 使用 Midjourney V7 + 参考图 |
| `nj2` | 图生图 | 使用 Niji 7 + 参考图 |

Midjourney 返回的四宫格图片会自动裁切成 4 张独立图片发送。

### 垫图功能

使用图生图模式的触发词（带 `2` 的），回复图片并发送绘图命令，图片会作为参考图传递给 AI 模型。如果没有传图片，会使用发送者头像作为参考。

### 预设提示词

#### 添加预设

管理员使用 `lmp` 命令添加预设提示词：

```
lmp myp 1girl, beautiful sunset --ar 16:9
```

#### 调用预设

使用 `--ps` 参数调用预设：

```
bt1 --ps myp
bt1 --ps myp 附加提示词
```

#### 查看预设列表

```
lml 或 lmpl
```

#### 查看预设详情

```
lmc myp 或 lmps myp
```

### 图片保存

在配置中启用「本地存储」后，生成的图片会保存到插件数据目录的 `save_images` 文件夹。

启用「同时保存 JSON 元数据」后，会同时保存同名 JSON 文件：

```json
{
  "tags": ["Midjourney-V7"],
  "annotation": "1girl --ar 4:3"
}
```

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

### 图片存储配置

- `local_save` - 本地存储开关
- `save_json` - 同时保存 JSON 元数据

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