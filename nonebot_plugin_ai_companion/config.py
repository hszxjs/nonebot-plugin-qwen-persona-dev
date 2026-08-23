from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MLX_LORA_MODULES: tuple[str, ...] = ("language", "projection", "vision_tower")


class AiCompanionConfig(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_prefix(cls, data: Any) -> Any:
        """兼容旧 qwen_persona_* 配置键（.env / admin_config.json），统一改写为 ai_companion_*。"""
        if isinstance(data, dict):
            rewritten = {
                ("ai_companion_" + key.removeprefix("qwen_persona_")) if key.startswith("qwen_persona_") else key: value
                for key, value in data.items()
            }
            return rewritten
        return data

    ai_companion_mlx_model: str = "mlx-community/Qwen3.6-9B-4bit"
    ai_companion_mlx_adapter_path: str = ""
    ai_companion_mlx_auto_load: bool = False
    ai_companion_mlx_trust_remote_code: bool = False
    ai_companion_model_dir: str = "data/qwen_persona/models"
    ai_companion_request_timeout: float = 120.0
    ai_companion_temperature: float = 0.85
    ai_companion_top_p: float = 0.9
    ai_companion_repetition_penalty: float = 1.08
    ai_companion_repetition_context_size: int = 64
    ai_companion_max_tokens: int = 256

    ai_companion_admin_enabled: bool = True
    ai_companion_admin_host: str = "127.0.0.1"
    ai_companion_admin_port: int = 8082
    ai_companion_admin_token: str = ""
    ai_companion_admin_config_file: str = ""

    ai_companion_data_dir: str = "data/qwen_persona"
    ai_companion_enabled_groups: set[str] = Field(default_factory=set)

    ai_companion_enable_learning: bool = True
    ai_companion_enable_reply: bool = True
    ai_companion_learning_priority: int = 9998
    ai_companion_reply_priority: int = 9999
    ai_companion_reply_probability: float = 0.0
    ai_companion_follow_reply_probability: float = 0.03
    ai_companion_text_sticker_reply_probability: float = 0.2
    ai_companion_trigger_keywords: list[str] = Field(default_factory=list)
    ai_companion_enable_repeat: bool = True
    ai_companion_repeat_min_count: int = 2
    ai_companion_repeat_probability: float = 0.65
    ai_companion_repeat_cooldown_seconds: int = 90
    ai_companion_pattern_reply_threshold: int = 3
    ai_companion_context_scope: Literal["group", "global", "mixed"] = "mixed"
    ai_companion_context_messages: int = 16
    ai_companion_min_meme_recent_count: int = 2

    ai_companion_min_sample_chars: int = 2
    ai_companion_max_sample_chars: int = 600
    ai_companion_command_prefixes: list[str] = Field(default_factory=lambda: ["/", "!", "！", "#", "＃", ".", "。"])
    ai_companion_command_regexes: list[str] = Field(
        default_factory=lambda: [
            r"^\s*(/|!|！|#|＃|\.|。)\S+",
            r"^\s*(help|帮助|菜单|menu)(?:\s|$).*",
            r"^\s*(面板|批量炼化灵石)\s*$",
            r"^\s*(修仙签到|修仙|签到|闭关|出关|突破|渡劫|灵石|灵石排行|宗门|宗门信息|秘境|探索|历练|炼丹|炼器|背包|境界|悬赏|天机)(?:\s|$|[，,].*)",
            r"^\s*(?:垂钓|钓鱼|钓)(?:\s*(?:十连|十连抽|10连|10连抽|来一次|\d+\s*次?)?)?\s*$",
            r"^\s*(?:十连|十连抽|10连|10连抽)\s*$",
        ]
    )
    ai_companion_auto_read_picmenu_config: bool = True
    ai_companion_picmenu_config_keys: list[str] = Field(
        default_factory=lambda: ["picmenu", "pic_menu", "nonebot_plugin_picmenu"]
    )
    ai_companion_picmenu_command_keys: list[str] = Field(
        default_factory=lambda: [
            "command",
            "commands",
            "cmd",
            "cmds",
            "alias",
            "aliases",
            "trigger",
            "triggers",
            "keyword",
            "keywords",
            "命令",
            "指令",
            "触发词",
            "别名",
        ]
    )

    ai_companion_sticker_segment_types: set[str] = Field(
        default_factory=lambda: {"face", "mface", "emoji", "sticker", "image"}
    )
    ai_companion_sticker_prompt_limit: int = 12
    ai_companion_min_reply_sticker_count: int = 5
    ai_companion_sticker_repo_dir: str = "data/qwen_persona/sticker_repo"
    ai_companion_sticker_repo_url: str = ""
    ai_companion_enable_sending_stickers: bool = True
    ai_companion_max_reply_stickers: int = 2
    ai_companion_media_image_fallback_text: str = "[图片]"
    ai_companion_media_sticker_fallback_text: str = "[表情]"

    ai_companion_enable_training: bool = True
    ai_companion_train_times: list[str] = Field(default_factory=lambda: ["06:10", "23:50"])
    ai_companion_lora_train_command: str = ""
    ai_companion_after_train_command: str = ""
    ai_companion_train_cwd: str = ""
    ai_companion_train_timeout_seconds: int = 0
    ai_companion_dataset_limit: int = 4000
    ai_companion_dataset_history_window: int = 4
    ai_companion_dataset_excluded_user_ids: set[str] = Field(default_factory=set)
    ai_companion_external_corpus_enabled: bool = True
    ai_companion_external_corpus_limit: int = 0
    ai_companion_min_train_samples: int = 20
    ai_companion_multimodal_training_enabled: bool = True
    ai_companion_mlx_lora_train_modules: list[str] = Field(default_factory=lambda: ["language"])
    ai_companion_mlx_lora_rank: int = 8
    ai_companion_mlx_lora_scale: float = 20.0
    ai_companion_mlx_lora_dropout: float = 0.0
    ai_companion_mlx_lora_learning_rate: float = 2e-5
    ai_companion_mlx_lora_batch_size: int = 1
    ai_companion_mlx_lora_grad_accum_steps: int = 4
    ai_companion_mlx_lora_epochs: int = 1
    ai_companion_mlx_lora_iters: int = 0
    ai_companion_mlx_lora_max_seq_length: int = 2048
    ai_companion_mlx_lora_grad_checkpoint: bool = True
    ai_companion_mlx_lora_mask_prompt: bool = True
    ai_companion_mlx_lora_save_every: int = 100
    ai_companion_mlx_lora_steps_per_report: int = 10
    ai_companion_lora_helper_api_url: str = ""
    ai_companion_lora_helper_api_key: str = ""
    ai_companion_lora_helper_model: str = "glm-5.2"
    ai_companion_lora_helper_timeout: float = 120.0
    ai_companion_lora_helper_sample_limit: int = 0
    ai_companion_lora_helper_extra_body: dict[str, Any] = Field(default_factory=dict)
    ai_companion_lora_helper_system_prompt: str = (
        "你是群聊 LoRA 数据清洗助手。请保留群友自然口吻、梗和表情 token，"
        '去掉命令、隐私、广告和重复噪声。只返回一个 JSON 对象，结构保持 {"messages": [...], "metadata": {...}}。'
    )

    # 对话推理后端切换：mlx = 本地 Qwen MLX，glm = GLM 在线 API（OpenAI 兼容）。
    ai_companion_inference_backend: Literal["mlx", "glm"] = "mlx"
    # GLM 在线推理独立配置（与 lora_helper 辅助训练解耦，可用不同模型）。
    ai_companion_online_api_url: str = ""
    ai_companion_online_api_key: str = ""
    ai_companion_online_model: str = "glm-5.2"
    ai_companion_online_timeout: float = 90.0
    # GLM 端点是否支持多模态。Coding 订阅端点只支持纯文本，默认 False：
    # 遇到图片时丢弃图片仅发文字。普通 paas/v4 + API 资源包可设 True。
    ai_companion_online_multimodal: bool = False

    # 动漫图片识别：用 SauceNAO + iqdb 双路搜索角色/作品/画师，转文字描述进 prompt。
    # 开启后主回复路径的图片会走动漫识别（不依赖本地 Qwen-VL，二次元识别更准）。
    ai_companion_anime_caption_enabled: bool = True
    # SauceNAO API key（saucenao.com 免费注册）。空则只用 iqdb 单路。
    ai_companion_saucenao_api_key: str = ""
    # SauceNAO 数据库掩码：999 = 默认动漫相关库全开。
    ai_companion_saucenao_dbmask: int = 999
    # 命中阈值：相似度低于此值视为未识别，返回空（降级为文字标记）。
    ai_companion_anime_min_similarity: float = 80.0
    # AI 视觉识图：用 GLM-4.6V-Flash（免费）理解任意图的内容（角色/场景/动作），
    # 对插画/截图/同人都能识别，弥补 SauceNAO/iqdb 对冷门图命中率低的问题。
    ai_companion_anime_vision_enabled: bool = True
    # 视觉模型名（z.ai 免费视觉模型）。glm-4.6v-flash 免费，glm-4.5v 需付费资源包。
    ai_companion_anime_vision_model: str = "glm-4.6v-flash"
    # 视觉模型 API（复用 z.ai key）。空则用 online_api_url/key 的同平台凭证。
    ai_companion_anime_vision_api_url: str = "https://api.z.ai/api/paas/v4/chat/completions"
    ai_companion_anime_vision_api_key: str = ""

    # 合并转发消息：调 get_forward_msg API 展开内容，提取文字+图片。
    ai_companion_forward_max_messages: int = 20  # 最多展开的转发条数（防爆 token）
    ai_companion_forward_max_chars: int = 1500  # 转发文字合并后的最大字符
    # 图片智能控制：消息里图片超过此数则跳过识别，只处理文字（省成本/延迟）。
    ai_companion_media_image_limit: int = 5
    # 图文相关性判断：识别图片后，若与文字无关则丢弃图片描述只保留文字。
    # 需多一次轻量 API 调用（~1s），可关闭。
    ai_companion_media_relevance_check: bool = True

    ai_companion_blacklist_words: list[str] = Field(default_factory=list)
    ai_companion_blacklist_regexes: list[str] = Field(default_factory=list)
    ai_companion_blacklist_user_ids: set[str] = Field(default_factory=set)
    ai_companion_blacklist_group_ids: set[str] = Field(default_factory=set)

    ai_companion_system_prompt: str = (
        "你是在群聊中参与对话的普通成员。优先理解上下文、直接回答问题并接住当前话题；"
        "语气可以自然短句，但不要为了套用固定句式而忽略问题，"
        "不要输出用户 id 或聊天记录格式。不会泄露隐私、不会复述本地插件命令、"
        "不会假装拥有管理员权限。"
    )

    # ---- 提示词分段（文档十节）：开关与排序可运营配置，不改代码 ----
    ai_companion_prompt_segment_order: list[str] = Field(default_factory=list)
    ai_companion_disabled_prompt_segments: list[str] = Field(default_factory=list)

    # ---- 工具调用（默认关闭；GLM 在线后端支持 function calling 回路）----
    ai_companion_enable_tool_calls: bool = False
    ai_companion_tool_max_rounds: int = 2
    ai_companion_tool_call_timeout_seconds: float = 15.0
    ai_companion_tool_result_max_chars: int = 1200

    # ---- 网页搜索 Provider（BaseSearchProvider 实现：searxng / tavily / exa）----
    ai_companion_search_provider: str = ""
    ai_companion_searxng_url: str = ""
    ai_companion_tavily_api_key: str = ""
    ai_companion_exa_api_key: str = ""
    ai_companion_search_result_limit: int = 3

    # ---- 本地推理后端默认端点（Ollama / llama.cpp，可创建多个 DB 实例覆盖）----
    ai_companion_ollama_api_url: str = "http://127.0.0.1:11434"
    ai_companion_llama_cpp_api_url: str = "http://127.0.0.1:8080"

    # ---- 可观测性：持久化日志滚动淘汰上限（inference/sticker/message 各自生效）----
    ai_companion_log_max_records: int = 1000

    # ---- LoRA 训练后自动部署（登记新版本即激活 + 持久化 adapter 路径 + MLX 模式热重载）----
    ai_companion_lora_auto_deploy: bool = True

    # ---- member_mimic 风格分析 ----
    ai_companion_member_mimic_model: str = "glm-4.5-flash"
    ai_companion_member_mimic_sample_limit: int = 120

    model_config = ConfigDict(extra="ignore")


def effective_mlx_model(config: AiCompanionConfig) -> str:
    return config.ai_companion_mlx_model.strip() or "mlx-community/Qwen3.6-9B-4bit"


def effective_chat_model(config: AiCompanionConfig) -> str:
    if config.ai_companion_inference_backend == "glm":
        return (config.ai_companion_online_model or "glm-5.2").strip() or "glm-5.2"
    return effective_mlx_model(config)


def effective_api_base(_config: AiCompanionConfig) -> str:
    return "embedded://mlx-vlm"


def normalize_matcher_priorities(config: AiCompanionConfig) -> list[str]:
    learning_priority = int(config.ai_companion_learning_priority)
    reply_priority = int(config.ai_companion_reply_priority)
    if learning_priority < reply_priority:
        return []
    config.ai_companion_learning_priority = reply_priority - 1
    return ["ai_companion_learning_priority"]


def normalized_lora_modules(config: AiCompanionConfig) -> list[str]:
    modules: list[str] = []
    for item in config.ai_companion_mlx_lora_train_modules:
        value = str(item or "").strip()
        if value in MLX_LORA_MODULES and value not in modules:
            modules.append(value)
    return modules or ["language"]


def _dump_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    if isinstance(config, dict):
        return config
    return {}


def load_plugin_config(driver_config: Any) -> AiCompanionConfig:
    return AiCompanionConfig(**_dump_config(driver_config))


# 兼容别名：旧代码/第三方仍可 from .config import QwenPersonaConfig
QwenPersonaConfig = AiCompanionConfig
