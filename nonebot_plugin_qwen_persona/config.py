from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


LLAMA_MODEL_PRESETS: list[dict[str, str]] = [
    {
        "value": "Qwen/Qwen3-0.6B-GGUF:Q8_0",
        "label": "Qwen3 0.6B Q8_0 - light",
        "description": "Fast local baseline for CPU and small devices.",
    },
    {
        "value": "Qwen/Qwen3-1.7B-GGUF:Q6_K",
        "label": "Qwen3 1.7B Q6_K - small",
        "description": "Better chat quality while still practical on laptops.",
    },
    {
        "value": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
        "label": "Qwen3 4B Q4_K_M - medium",
        "description": "Recommended default for capable local machines.",
    },
    {
        "value": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
        "label": "Qwen3 8B Q4_K_M - large",
        "description": "Higher quality, needs more memory and patience.",
    },
    {
        "value": "Qwen/Qwen3-14B-GGUF:Q4_K_M",
        "label": "Qwen3 14B Q4_K_M - high quality",
        "description": "Strong local chat quality for high-memory machines.",
    },
    {
        "value": "Qwen/Qwen3-32B-GGUF:Q4_K_M",
        "label": "Qwen3 32B Q4_K_M - workstation",
        "description": "Large Qwen preset for workstation or external llama.cpp backends.",
    },
    {
        "value": "Qwen/Qwen3.6-27B-FP8",
        "label": "Qwen 27B FP8 - custom/external",
        "description": "Reserved for high-memory backends or external API mode.",
    },
    {
        "value": "google/gemma-2-9b-it-GGUF:Q4_K_M",
        "label": "Gemma 2 9B IT Q4_K_M",
        "description": "Compact instruction model with strong multilingual baseline.",
    },
    {
        "value": "google/gemma-2-27b-it-GGUF:Q4_K_M",
        "label": "Gemma 2 27B IT Q4_K_M",
        "description": "Larger Gemma option for external or high-memory llama.cpp setups.",
    },
    {
        "value": "meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
        "label": "Llama 3.1 8B Instruct Q4_K_M",
        "description": "General-purpose open instruction model.",
    },
    {
        "value": "mistralai/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M",
        "label": "Mistral 7B Instruct v0.3 Q4_K_M",
        "description": "Fast general instruction model with modest memory needs.",
    },
    {
        "value": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
        "label": "DeepSeek R1 Distill Qwen 14B Q4_K_M",
        "description": "Reasoning-oriented Qwen distill for larger local deployments.",
    },
]


class QwenPersonaConfig(BaseModel):
    qwen_persona_runtime: Literal["llama_cpp", "external_llama", "external_api"] = "llama_cpp"
    qwen_persona_api_base: str = "http://127.0.0.1:8000/v1"
    qwen_persona_api_key: str = ""
    qwen_persona_model: str = "Qwen/Qwen3.6-27B-FP8"
    qwen_persona_request_timeout: float = 120.0
    qwen_persona_temperature: float = 0.85
    qwen_persona_top_p: float = 0.9
    qwen_persona_max_tokens: int = 256

    qwen_persona_llama_server_path: str = ""
    qwen_persona_llama_auto_install: bool = True
    qwen_persona_llama_model: str = "Qwen/Qwen3-0.6B-GGUF:Q8_0"
    qwen_persona_llama_model_file: str = ""
    qwen_persona_llama_host: str = "127.0.0.1"
    qwen_persona_llama_port: int = 18080
    qwen_persona_llama_ctx_size: int = 4096
    qwen_persona_llama_extra_args: list[str] | str = Field(default_factory=list)

    qwen_persona_admin_enabled: bool = True
    qwen_persona_admin_host: str = "127.0.0.1"
    qwen_persona_admin_port: int = 8082
    qwen_persona_admin_token: str = ""
    qwen_persona_admin_config_file: str = ""

    qwen_persona_data_dir: str = "data/qwen_persona"
    qwen_persona_model_dir: str = "data/qwen_persona/models"
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
    qwen_persona_sticker_repo_dir: str = "data/qwen_persona/sticker_repo"
    qwen_persona_sticker_repo_url: str = ""
    qwen_persona_enable_sending_stickers: bool = True
    qwen_persona_max_reply_stickers: int = 2

    qwen_persona_vision_enabled: bool = True
    qwen_persona_vision_auto_start: bool = True
    qwen_persona_vision_python: str = ""
    qwen_persona_vision_host: str = "127.0.0.1"
    qwen_persona_vision_port: int = 18081
    qwen_persona_vision_model: str = "Salesforce/blip-image-captioning-base"
    qwen_persona_vision_timeout: float = 20.0
    qwen_persona_vision_image_fallback_text: str = "[图片]"
    qwen_persona_vision_sticker_fallback_text: str = "[表情]"

    qwen_persona_enable_training: bool = True
    qwen_persona_train_times: list[str] = Field(default_factory=lambda: ["06:10", "23:50"])
    qwen_persona_lora_train_command: str = ""
    qwen_persona_after_train_command: str = ""
    qwen_persona_train_cwd: str = ""
    qwen_persona_train_timeout_seconds: int = 0
    qwen_persona_dataset_limit: int = 4000
    qwen_persona_dataset_history_window: int = 4
    qwen_persona_external_corpus_enabled: bool = True
    qwen_persona_external_corpus_limit: int = 0
    qwen_persona_min_train_samples: int = 20
    qwen_persona_lora_api_url: str = ""
    qwen_persona_lora_api_key: str = ""
    qwen_persona_lora_api_timeout: float = 120.0
    qwen_persona_lora_api_extra_body: dict[str, Any] = Field(default_factory=dict)
    qwen_persona_lora_helper_api_url: str = ""
    qwen_persona_lora_helper_api_key: str = ""
    qwen_persona_lora_helper_model: str = "gpt-5.5-medium"
    qwen_persona_lora_helper_timeout: float = 120.0
    qwen_persona_lora_helper_sample_limit: int = 0
    qwen_persona_lora_helper_extra_body: dict[str, Any] = Field(default_factory=dict)
    qwen_persona_lora_helper_system_prompt: str = (
        "你是群聊 LoRA 数据清洗助手。请保留群友自然口吻、梗和表情 token，"
        "去掉命令、隐私、广告和重复噪声。只返回一个 JSON 对象，结构保持 {\"messages\": [...], \"metadata\": {...}}。"
    )

    qwen_persona_blacklist_words: list[str] = Field(default_factory=list)
    qwen_persona_blacklist_regexes: list[str] = Field(default_factory=list)
    qwen_persona_blacklist_user_ids: set[str] = Field(default_factory=set)
    qwen_persona_blacklist_group_ids: set[str] = Field(default_factory=set)

    qwen_persona_system_prompt: str = (
        "你正在扮演这个群里的一位熟人。你会学习群友说话习惯、梗、短句和表情包使用方式，"
        "但不会泄露隐私、不会复述本地插件命令、不会假装拥有管理员权限。"
    )

    class Config:
        extra = "ignore"


def effective_api_base(config: QwenPersonaConfig) -> str:
    if config.qwen_persona_runtime in {"llama_cpp", "external_llama"}:
        host = config.qwen_persona_llama_host.strip() or "127.0.0.1"
        return f"http://{host}:{int(config.qwen_persona_llama_port)}/v1"
    return config.qwen_persona_api_base.rstrip("/") or "http://127.0.0.1:8000/v1"


def effective_chat_model(config: QwenPersonaConfig) -> str:
    if config.qwen_persona_runtime in {"llama_cpp", "external_llama"}:
        return config.qwen_persona_llama_model.strip() or config.qwen_persona_model
    return config.qwen_persona_model.strip() or config.qwen_persona_llama_model


def normalize_matcher_priorities(config: QwenPersonaConfig) -> list[str]:
    learning_priority = int(config.qwen_persona_learning_priority)
    reply_priority = int(config.qwen_persona_reply_priority)
    if learning_priority < reply_priority:
        return []
    config.qwen_persona_learning_priority = reply_priority - 1
    return ["qwen_persona_learning_priority"]


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
