from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MLX_MODEL_PRESETS: list[dict[str, str]] = [
    {
        "value": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "label": "Qwen 3.5 VL 9B 4bit - recommended",
        "description": "Default Apple Silicon VLM target for 16GB unified memory.",
    },
    {
        "value": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "label": "Qwen2.5 VL 7B 4bit - lighter fallback",
        "description": "Smaller VL fallback when the 9B model is too tight.",
    },
]

MLX_LORA_MODULES: tuple[str, ...] = ("language", "projection", "vision_tower")


class QwenPersonaConfig(BaseModel):
    qwen_persona_mlx_model: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
    qwen_persona_mlx_adapter_path: str = ""
    qwen_persona_mlx_auto_load: bool = False
    qwen_persona_mlx_trust_remote_code: bool = False
    qwen_persona_model_dir: str = "data/qwen_persona/models"
    qwen_persona_request_timeout: float = 120.0
    qwen_persona_temperature: float = 0.85
    qwen_persona_top_p: float = 0.9
    qwen_persona_repetition_penalty: float = 1.08
    qwen_persona_repetition_context_size: int = 64
    qwen_persona_max_tokens: int = 256

    qwen_persona_admin_enabled: bool = True
    qwen_persona_admin_host: str = "127.0.0.1"
    qwen_persona_admin_port: int = 8082
    qwen_persona_admin_token: str = ""
    qwen_persona_admin_config_file: str = ""

    qwen_persona_data_dir: str = "data/qwen_persona"
    qwen_persona_enabled_groups: set[str] = Field(default_factory=set)

    qwen_persona_enable_learning: bool = True
    qwen_persona_enable_reply: bool = True
    qwen_persona_learning_priority: int = 9998
    qwen_persona_reply_priority: int = 9999
    qwen_persona_reply_probability: float = 0.0
    qwen_persona_follow_reply_probability: float = 0.03
    qwen_persona_text_sticker_reply_probability: float = 0.2
    qwen_persona_trigger_keywords: list[str] = Field(default_factory=list)
    qwen_persona_enable_repeat: bool = True
    qwen_persona_repeat_min_count: int = 2
    qwen_persona_repeat_probability: float = 0.65
    qwen_persona_repeat_cooldown_seconds: int = 90
    qwen_persona_pattern_reply_threshold: int = 3
    qwen_persona_context_scope: Literal["group", "global", "mixed"] = "mixed"
    qwen_persona_context_messages: int = 16
    qwen_persona_min_meme_recent_count: int = 2

    qwen_persona_min_sample_chars: int = 2
    qwen_persona_max_sample_chars: int = 600
    qwen_persona_command_prefixes: list[str] = Field(
        default_factory=lambda: ["/", "!", "！", "#", "＃", ".", "。"]
    )
    qwen_persona_command_regexes: list[str] = Field(
        default_factory=lambda: [
            r"^\s*(/|!|！|#|＃|\.|。)\S+",
            r"^\s*(help|帮助|菜单|menu)(?:\s|$).*",
            r"^\s*(面板|批量炼化灵石)\s*$",
            r"^\s*(修仙签到|修仙|签到|闭关|出关|突破|渡劫|灵石|灵石排行|宗门|宗门信息|秘境|探索|历练|炼丹|炼器|背包|境界|悬赏|天机)(?:\s|$|[，,].*)",
            r"^\s*(?:垂钓|钓鱼|钓)(?:\s*(?:十连|十连抽|10连|10连抽|来一次|\d+\s*次?)?)?\s*$",
            r"^\s*(?:十连|十连抽|10连|10连抽)\s*$",
        ]
    )
    qwen_persona_auto_read_picmenu_config: bool = True
    qwen_persona_picmenu_config_keys: list[str] = Field(
        default_factory=lambda: ["picmenu", "pic_menu", "nonebot_plugin_picmenu"]
    )
    qwen_persona_picmenu_command_keys: list[str] = Field(
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

    qwen_persona_sticker_segment_types: set[str] = Field(
        default_factory=lambda: {"face", "mface", "emoji", "sticker", "image"}
    )
    qwen_persona_sticker_prompt_limit: int = 12
    qwen_persona_min_reply_sticker_count: int = 5
    qwen_persona_sticker_repo_dir: str = "data/qwen_persona/sticker_repo"
    qwen_persona_sticker_repo_url: str = ""
    qwen_persona_enable_sending_stickers: bool = True
    qwen_persona_max_reply_stickers: int = 2
    qwen_persona_media_image_fallback_text: str = "[图片]"
    qwen_persona_media_sticker_fallback_text: str = "[表情]"

    qwen_persona_enable_training: bool = True
    qwen_persona_train_times: list[str] = Field(default_factory=lambda: ["06:10", "23:50"])
    qwen_persona_lora_train_command: str = ""
    qwen_persona_after_train_command: str = ""
    qwen_persona_train_cwd: str = ""
    qwen_persona_train_timeout_seconds: int = 0
    qwen_persona_dataset_limit: int = 4000
    qwen_persona_dataset_history_window: int = 4
    qwen_persona_dataset_excluded_user_ids: set[str] = Field(default_factory=set)
    qwen_persona_external_corpus_enabled: bool = True
    qwen_persona_external_corpus_limit: int = 0
    qwen_persona_min_train_samples: int = 20
    qwen_persona_multimodal_training_enabled: bool = True
    qwen_persona_mlx_lora_train_modules: list[str] = Field(
        default_factory=lambda: ["language"]
    )
    qwen_persona_mlx_lora_rank: int = 8
    qwen_persona_mlx_lora_scale: float = 20.0
    qwen_persona_mlx_lora_dropout: float = 0.0
    qwen_persona_mlx_lora_learning_rate: float = 2e-5
    qwen_persona_mlx_lora_batch_size: int = 1
    qwen_persona_mlx_lora_grad_accum_steps: int = 4
    qwen_persona_mlx_lora_epochs: int = 1
    qwen_persona_mlx_lora_iters: int = 0
    qwen_persona_mlx_lora_max_seq_length: int = 2048
    qwen_persona_mlx_lora_grad_checkpoint: bool = True
    qwen_persona_mlx_lora_mask_prompt: bool = True
    qwen_persona_mlx_lora_save_every: int = 100
    qwen_persona_mlx_lora_steps_per_report: int = 10
    qwen_persona_lora_helper_api_url: str = ""
    qwen_persona_lora_helper_api_key: str = ""
    qwen_persona_lora_helper_model: str = "glm-5.2"
    qwen_persona_lora_helper_timeout: float = 120.0
    qwen_persona_lora_helper_sample_limit: int = 0
    qwen_persona_lora_helper_extra_body: dict[str, Any] = Field(default_factory=dict)
    qwen_persona_lora_helper_system_prompt: str = (
        "你是群聊 LoRA 数据清洗助手。请保留群友自然口吻、梗和表情 token，"
        "去掉命令、隐私、广告和重复噪声。只返回一个 JSON 对象，结构保持 {\"messages\": [...], \"metadata\": {...}}。"
    )

    # 对话推理后端切换：mlx = 本地 Qwen MLX，glm = GLM 在线 API（OpenAI 兼容）。
    qwen_persona_inference_backend: Literal["mlx", "glm"] = "mlx"
    # GLM 在线推理独立配置（与 lora_helper 辅助训练解耦，可用不同模型）。
    qwen_persona_online_api_url: str = ""
    qwen_persona_online_api_key: str = ""
    qwen_persona_online_model: str = "glm-5.2"
    qwen_persona_online_timeout: float = 90.0
    # GLM 端点是否支持多模态。Coding 订阅端点只支持纯文本，默认 False：
    # 遇到图片时丢弃图片仅发文字。普通 paas/v4 + API 资源包可设 True。
    qwen_persona_online_multimodal: bool = False

    # 动漫图片识别：用 SauceNAO + iqdb 双路搜索角色/作品/画师，转文字描述进 prompt。
    # 开启后主回复路径的图片会走动漫识别（不依赖本地 Qwen-VL，二次元识别更准）。
    qwen_persona_anime_caption_enabled: bool = True
    # SauceNAO API key（saucenao.com 免费注册）。空则只用 iqdb 单路。
    qwen_persona_saucenao_api_key: str = ""
    # SauceNAO 数据库掩码：999 = 默认动漫相关库全开。
    qwen_persona_saucenao_dbmask: int = 999
    # 命中阈值：相似度低于此值视为未识别，返回空（降级为文字标记）。
    qwen_persona_anime_min_similarity: float = 80.0
    # AI 视觉识图：用 GLM-4.6V-Flash（免费）理解任意图的内容（角色/场景/动作），
    # 对插画/截图/同人都能识别，弥补 SauceNAO/iqdb 对冷门图命中率低的问题。
    qwen_persona_anime_vision_enabled: bool = True
    # 视觉模型名（z.ai 免费视觉模型）。glm-4.6v-flash 免费，glm-4.5v 需付费资源包。
    qwen_persona_anime_vision_model: str = "glm-4.6v-flash"
    # 视觉模型 API（复用 z.ai key）。空则用 online_api_url/key 的同平台凭证。
    qwen_persona_anime_vision_api_url: str = "https://api.z.ai/api/paas/v4/chat/completions"
    qwen_persona_anime_vision_api_key: str = ""

    # 合并转发消息：调 get_forward_msg API 展开内容，提取文字+图片。
    qwen_persona_forward_max_messages: int = 20  # 最多展开的转发条数（防爆 token）
    qwen_persona_forward_max_chars: int = 1500   # 转发文字合并后的最大字符
    # 图片智能控制：消息里图片超过此数则跳过识别，只处理文字（省成本/延迟）。
    qwen_persona_media_image_limit: int = 5
    # 图文相关性判断：识别图片后，若与文字无关则丢弃图片描述只保留文字。
    # 需多一次轻量 API 调用（~1s），可关闭。
    qwen_persona_media_relevance_check: bool = True

    qwen_persona_blacklist_words: list[str] = Field(default_factory=list)
    qwen_persona_blacklist_regexes: list[str] = Field(default_factory=list)
    qwen_persona_blacklist_user_ids: set[str] = Field(default_factory=set)
    qwen_persona_blacklist_group_ids: set[str] = Field(default_factory=set)

    qwen_persona_system_prompt: str = (
        "你是在群聊中参与对话的普通成员。优先理解上下文、直接回答问题并接住当前话题；"
        "语气可以自然短句，但不要为了套用固定句式而忽略问题，"
        "不要输出用户 id 或聊天记录格式。不会泄露隐私、不会复述本地插件命令、"
        "不会假装拥有管理员权限。"
    )

    model_config = ConfigDict(extra="ignore")


def effective_mlx_model(config: QwenPersonaConfig) -> str:
    return (
        config.qwen_persona_mlx_model.strip()
        or "mlx-community/Qwen3.5-9B-MLX-4bit"
    )


def effective_chat_model(config: QwenPersonaConfig) -> str:
    if config.qwen_persona_inference_backend == "glm":
        return (config.qwen_persona_online_model or "glm-5.2").strip() or "glm-5.2"
    return effective_mlx_model(config)


def effective_api_base(_config: QwenPersonaConfig) -> str:
    return "embedded://mlx-vlm"


def normalize_matcher_priorities(config: QwenPersonaConfig) -> list[str]:
    learning_priority = int(config.qwen_persona_learning_priority)
    reply_priority = int(config.qwen_persona_reply_priority)
    if learning_priority < reply_priority:
        return []
    config.qwen_persona_learning_priority = reply_priority - 1
    return ["qwen_persona_learning_priority"]


def normalized_lora_modules(config: QwenPersonaConfig) -> list[str]:
    modules: list[str] = []
    for item in config.qwen_persona_mlx_lora_train_modules:
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


def load_plugin_config(driver_config: Any) -> QwenPersonaConfig:
    return QwenPersonaConfig(**_dump_config(driver_config))
