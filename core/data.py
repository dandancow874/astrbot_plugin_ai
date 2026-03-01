from dataclasses import dataclass
from typing import Literal

# 常数
DEF_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEF_OPENAI_IMAGES_API_URL = "https://api.openai.com/v1/images/generations"
DEF_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEF_VERTEX_AI_ANONYMOUS_BASE_API = "https://cloudconsole-pa.clients6.google.com"

# 类型枚举
_API_Type = Literal["Gemini", "OpenAI_Chat", "OpenAI_Images", "Vertex_AI_Anonymous"]


@dataclass(repr=False, slots=True)
class ModelParams:
    """模型默认参数"""

    max_images: int = 6
    """最大输入图片数量"""
    image_size: str = "2K"
    """分辨率"""
    aspect_ratio: str = "default"
    """宽高比"""


@dataclass(repr=False, slots=True)
class ModelInfo:
    """模型信息"""

    model_name: str
    """模型名称"""
    triggers: list[str]
    """触发词列表"""
    default_params: ModelParams = ModelParams()
    """默认参数"""


@dataclass(repr=False, slots=True)
class ProviderConfig:
    """服务商配置信息"""

    name: str
    """服务商名称"""
    enabled: bool
    """是否启用"""
    priority: int
    """优先级，数值越小越优先"""
    base_url: str
    """Base URL"""
    api_key: str
    """API Key"""
    api_type: _API_Type
    """API 类型"""
    tls_verify: bool = True
    """TLS 证书验证"""
    impersonate: str = "chrome131"
    """TLS/UA 指纹伪装"""
    models: list[ModelInfo] = None
    """模型列表"""

    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass(repr=False, slots=True)
class ModelConfig:
    """内部模型配置，用于兼容现有代码"""

    name: str
    """模型名称"""
    triggers: list[str]
    """触发指令列表"""
    providers: list[ProviderConfig]
    """提供商列表"""
    enabled: bool = True
    """是否启用"""


@dataclass(repr=False, slots=True)
class PromptConfig:
    """图片生成配置参数"""

    min_images: int = 1
    """最小输入图片数量"""
    max_images: int = 6
    """最大输入图片数量"""
    aspect_ratio: str = "default"
    """图片宽高比"""
    image_size: str = "1K"
    """图片尺寸/分辨率"""
    google_search: bool = False
    """是否启用谷歌搜索功能"""
    refer_images: str | None = None
    """引用参考图片的文件名"""
    gather_mode: bool = False
    """是否启用收集模式"""


@dataclass(repr=False, slots=True)
class CommonConfig:
    """常规配置参数"""

    preset_append: bool = False
    """ 是否在预设提示词后追加用户输入文本 """
    text_response: bool = False
    """是否启用文本响应"""
    tls_verify: bool = True
    """兼容旧配置：已废弃"""
    impersonate: str = ""
    """兼容旧配置：已废弃"""
    smart_retry: bool = True
    """是否启用智能重试"""
    max_retry: int = 3
    """最大重试次数"""
    timeout: float = 300
    """请求超时时间, 单位: 秒"""
    proxy: str | None = None
    """代理"""


@dataclass(repr=False, slots=True)
class PreferenceConfig:
    """偏好配置参数"""

    skip_at_first: bool = True
    """ 跳过第一次@机器人 """
    skip_quote_first: bool = False
    """ 跳过第一次引用@ """
    skip_llm_at_first: bool = False
    """ 跳过第一次LLM@ """


@dataclass(repr=False, slots=True)
class VertexAIAnonymousConfig:
    """Vertex AI Anonymous 配置参数"""

    recaptcha_base_api: str = "https://www.google.com"
    """Recaptcha 基础 API 地址"""
    vertex_ai_anonymous_base_api: str = "https://cloudconsole-pa.clients6.google.com"
    """Vertex AI Anonymous 基础 API 地址"""
    tls_verify: bool = True
    """是否验证 TLS 证书"""
    system_prompt: str | None = None
    """系统提示词"""
    max_retry: int = 10
    """最大重试次数"""
