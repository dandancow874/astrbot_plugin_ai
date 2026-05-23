# AstrBot Plugin AI

AstrBot 图片生成插件，支持多种 AI 绘图模型。

## 功能特性

- 支持多种 AI 绘图模型：
  - **nano-banana-2** (触发词: `bt`，兼容 `bt1`, `bt2`, `bt3`)
  - **nano-banana-pro** (触发词: `bp`，兼容 `bp1`, `bp2`, `bp3`)
  - **Z-Image-Turbo** (触发词: `zimg`)
  - **Qwen-Image-Edit** (触发词: `edit`) - 图片编辑
  - **Midjourney V7** (触发词: `mj`，兼容 `mj1`, `mj2`)
  - **Niji 7** (触发词: `nj`，兼容 `nj1`, `nj2`) - 动漫风格
  - **Image-to-Prompt** (触发词: `反推`) - 图片反推提示词

- `gpt`、`gp`、`bp`、`bt`、`mj`、`nj` 会按有图/无图自动沿用图生图/文生图参数；旧的 `1/2` 触发词仍兼容。
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
bt beautiful sunset landscape
zimg cyberpunk city
```

简化触发词会自动判断模式：

```
gpt 猫猫海报
gpt [图片] 把衣服换成蓝色
```

GPT Image 2 VIP 会在 `gpt` 命令指定 2K/4K 时自动启用：

```
gpt 电影海报 --size 2k
gpt [图片] 重绘为高清商业海报 --size 4k
```

VIP 使用单独的 `gpt_image_vip_config` 配置和白名单。`--size` 是 `--image_size` 的别名；不写 `--size 2k/4k` 时仍走普通 GPT Image。

### Midjourney 使用说明

| 触发词 | 模式 | 说明 |
|--------|------|------|
| `mj` | 自动 | 无图沿用 `mj1`，有图沿用 `mj2` |
| `nj` | 自动 | 无图沿用 `nj1`，有图沿用 `nj2` |
| `mj1` / `nj1` | 文生图 | 旧触发词，仍兼容 |
| `mj2` / `nj2` | 图生图 | 旧触发词，仍兼容 |

Midjourney 返回的四宫格图片会自动裁切成 4 张独立图片发送。

### 垫图功能

发送图片、回复图片、使用 `--id`、`--refer_images`、`--q` 或 `--头像` 时，会作为参考图传递给 AI 模型。简化触发词会自动按参考图数量选择文生图或图生图默认参数。

### 预设提示词

#### 添加预设

管理员使用 `lmp` 命令添加预设提示词：

```
lmp myp 1girl, beautiful sunset --ar 16:9
```

模型内置触发词不能作为预设名添加、更新或删除，例如 `gpt`、`gpt1`、`gpt2`、`gp`、`gp1`、`gp2` 以及设置页面里模型配置的触发词。`lml` / `lmpl` 只显示用户自定义预设，不显示这些内置触发词。

#### 调用预设

使用 `--ps` 参数调用预设：

```
bt --ps myp
bt --ps myp 附加提示词
```

预设名本身也可以作为触发词直接调用：

```
myp
myp 附加提示词
```

预设里可以指定参考图。把图片放到插件数据目录的 `refer_images` 文件夹，然后在预设中写文件名：

```
lmp poster 海报设计 --refer_images ref1.png --min_images 1
```

多个参考图用英文逗号分隔：

```
lmp poster 海报设计 --refer_images ref1.png,ref2.jpg --min_images 1
```

创建预设时如果消息里直接带图片，插件会自动把图片保存到 `refer_images`，并把文件名写进预设：

```
lmp [图片] poster 海报设计
```

会变成类似：

```
poster 海报设计 --refer_images poster_1.png --min_images 1
```

调用带参考图的预设时，参考图顺序固定为：预设中的 `refer_images` 在前，`--头像` 收集到的 @ 群友头像或 `--q` 头像在中，调用消息里新发的图片在后。

如果预设需要引用群友头像，在调用时加 `--头像`：

```
poster @张三 海报设计 --头像
poster @张三 @李四 海报设计 --头像
```

如果没有加 `--头像`，不会把 @ 群友当参考图；`gpt2`、`bp2` 这类旧图生图触发词也一样需要显式 `--头像`。若消息里只有图片没有 @，顺序就是：预设 `refer_images` 在前，消息图片在后。

#### 保存常用参考图

管理员可以用 `lmd` 保存常用图片，方便后续用 `--id` 调用。保存同名 id 会自动覆盖旧图片：

```
lmd 衣服参考 [图片]
```

也可以回复一条带图片的消息：

```
lmd 衣服参考
```

查看已保存图片：

```
lmid 衣服参考
```

生图时调用：

```
gpt 衣服换成蓝色 --id 衣服参考
```

`--id` 图片会排在参考图顺序的最后：预设 `refer_images`、`--头像`/`--q` 头像、消息图片、`--id` 图片。

### GPT Image质量

GPT Image 的 `quality` 不填时由上游默认处理，通常等价于 `auto`。可以在命令或预设里指定：

```
gpt 海报设计 --q high
gpt 海报设计 --quality medium
myp --q low
```

`--q high / medium / low / auto` 会作为 GPT Image 的质量参数；`--q 123456` 仍保留为 QQ 头像参考图参数。

### ComfyUI工作流

把 ComfyUI 工作流导出为 API Format JSON，放到插件目录的 `workflow/` 文件夹，例如：

```
workflow/转真人.json
```

管理员也可以在聊天窗口上传工作流，同名文件会直接覆盖：

```
cf上传 转真人
```

发送命令后 30 秒内发送 `转真人.json` 文件即可。也可以命令和文件同一条消息发送。

管理员可以直接开关 ComfyUI 环境：

```
cf开启
cf关闭
```

工作流里可以使用这些占位符：

```
{{prompt}}
{{image1_base64}}
{{image2_base64}}
{{image1_data_url}}
{{image2_data_url}}
{{image1}}
{{image2}}
{{seed}}
```

`{{image1_base64}}` / `{{image2_base64}}` 适合 Load Image Base64 节点；`{{image1}}` / `{{image2}}` 会先上传到 ComfyUI `/upload/image`，再替换为普通 LoadImage 节点需要的文件名。

调用：

```
cf 转真人 更真实一点，电影感
cf 转真人 [图片] 更真实一点，电影感
```

也可以创建预设：

```
lmp 转真人 cf 转真人 {{user_text}} --min_images 1
```

之后直接发送：

```
转真人 [图片] 更真实一点，电影感
```

#### 查看预设列表

```
lml 或 lmpl
```

普通用户也可以查看预设列表。

#### 查看预设详情

```
lmc myp 或 lmps myp
```

普通用户也可以查看预设详情。

#### 预设备份文件

`lmp`、`lm添加`、`lm删除` 改动预设后，会自动把当前预设列表导出为文本文件：

```
prompt_presets.md
```

文件位于插件数据目录中，通常和 `refer_images`、`save_images` 同级。内容是一行一个预设，方便手动复制或定期备份。

同时也会导出一份“单预设一个 Markdown 文件”的版本：

```
prompt_presets/
  myp.md
  海报设计.md
```

每个文件包含触发词、提示词和原始配置行。更新预设会覆盖对应 md，删除预设会自动删除对应 md。插件只会删除自己记录过的生成文件，不会清理你手动放进该目录的其他文件。

### 模型切换

可以在聊天窗口为当前用户切换默认使用的 provider model：

```
模型切换 gpt-image-2
模型切换 nano-banana-pro
模型切换 grok-imagine-1.0
```

也兼容旧的编号写法：

```
模型切换 1
模型切换 2
模型切换 3
```

还可以使用模型方案名切换，例如：

```
模型切换 GPT-Image-2
模型切换 Grok
```

切换结果会按用户保存。没有写死模型的预设（例如直接发送 `myp`）会优先使用当前用户切换过的模型。
生成前的等待提示会显示本次实际使用的模型名，便于确认预设走的是哪个模型。
通过默认模型、模型切换或简化触发词使用预设时，会沿用该模型旧 `1/2` 触发词的默认参数。例如默认模型是 `GPT-Image-2` 时，无图沿用 `gpt1` 的比例设置，带图沿用 `gpt2` 的 `auto` 图生图比例设置；手动传入 `--ar`、`--image_size`、`--model` 会优先于这些默认值。

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

- `default_provider_model` - 默认生图模型。填写 provider model 名称，例如 `gpt-image-2`、`nano-banana-pro`、`grok-imagine-1.0`；留空时使用模型列表中的第一个启用模型
- `timeout` - 超时时间（秒）
- `proxy` - HTTP 代理
- `max_retry` - 最大重试次数

### 群聊触发方式

- `require_at_in_group` - 启用后，群聊里必须 `@机器人` 才会触发绘图命令；私聊不受影响
- `coexist_enabled` - 混合模式，启用后无命令前缀和带命令前缀都可触发
- `prefix_list` - 命令前缀列表

## 安装

在 AstrBot 插件市场搜索 `astrbot_plugin_ai` 安装，或直接输入仓库地址：

```
https://github.com/dandancow874/astrbot_plugin_ai
```

## 依赖

- Pillow (用于 Midjourney 四宫格图片裁切)

## 许可证

MIT License
