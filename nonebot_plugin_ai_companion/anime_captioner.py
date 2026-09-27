"""动漫图片识别 captioner：SauceNAO + iqdb 双路搜索。

实现 MediaCaptioner Protocol（media.py:35-37），把群聊里的二次元/动漫图片
丢给 SauceNAO（JSON API，需 key）和 iqdb（HTML，免 key）双路搜索，
返回自然语言描述（角色/作品/画师/来源），拼进回复 prompt。

设计：
- iqdb 优先（免 key、无限流文档、现在就能用），SauceNAO 作为可选增强。
- 双路并发，任一命中（相似度达标）即用；都失败返回 ""（由 media.py 降级为图片标记）。
- 结果按图片内容 hash 缓存（TTL 1 小时），避免重复搜索。
- SauceNAO 配额耗尽（short_remaining/long_remaining <= 0）自动跳过。
- 全程容错：任何异常都返回 ""，绝不阻塞主回复。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from .config import AiCompanionConfig

logger = logging.getLogger(__name__)

# 缓存 TTL（秒）。
_CACHE_TTL = 3600
# 每路超时（秒）。
_PER_ROUTE_TIMEOUT = 20.0
# 整体超时（秒）——多路并发的总等待上限（视觉识图约 10s，留足余量）。
_TOTAL_TIMEOUT = 40.0
# 搜索前缩放的最长边（像素）。SauceNAO/iqdb 对小图更友好，且上传快得多。
_SEARCH_MAX_SIZE = 512
# 缩放后 JPEG 质量。
_SEARCH_JPEG_QUALITY = 85

# 诊断日志文件（独立于 nonebot 日志系统，便于事后排查 captioner 是否被调用）。
_DEBUG_LOG_PATH = "data/qwen_persona/caption_debug.log"


def _debug_log(message: str) -> None:
    """追加一行诊断日志到文件，带时间戳。失败静默。"""
    try:
        from pathlib import Path

        path = Path(_DEBUG_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def _downscale_for_search(image_bytes: bytes, *, max_size: int = _SEARCH_MAX_SIZE) -> bytes:
    """把图片缩放到最长边 max_size 像素，转 JPEG。

    SauceNAO/iqdb 对小图（几百像素）更友好，且上传快得多（9MB → 几十KB）。
    用线程执行避免阻塞事件循环。失败时返回原图（让搜索照常尝试）。
    """
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        max_dim = max(img.width, img.height)
        if max_dim > max_size:
            ratio = max_size / max_dim
            new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_SEARCH_JPEG_QUALITY, optimize=True)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        _debug_log(f"DOWNSCALE_FAIL exc={exc!r} using_orig")
        return image_bytes


def _extract_chat_content(data: dict[str, Any]) -> str:
    """从 OpenAI 兼容的 chat 响应里提取回复文本（兼容 content 拆成多段的情况）。"""
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, dict):
                    parts.append(str(seg.get("text") or seg.get("content") or ""))
                else:
                    parts.append(str(seg))
            content = "".join(parts)
        return str(content or "")
    except (KeyError, IndexError, TypeError):
        return ""


# iqdb 结果页里的图库域名（用于解析来源）。
_IQDB_SOURCE_HOSTS = {
    "danbooru": "Danbooru",
    "gelbooru": "Gelbooru",
    "yandere": "Yande.re",
    "konachan": "Konachan",
    "sankaku": "Sankaku",
    "e-shuushuu": "E-shuushuu",
    "zerochan": "Zerochan",
    "anime-pictures": "Anime-Pictures",
}


# SauceNAO 配额耗尽标记（用于自动跳过）。
class _SauceNaoQuotaExhausted(Exception):
    pass


_GENERAL_REACTION_PROMPT = (
    "这是群聊里出现的一张图，请为回复提供素材。严格按四行输出（每行以标签开头）：\n"
    "类型：梗图/聊天或博客截图/新闻或热点图/照片/二次元插画/涩图擦边/表情包 之一\n"
    "文字：把图内出现的关键文字原样摘录（截图和梗图的文字最重要，没有写'无'）\n"
    "内容：一两句话客观描述画面（能认出的角色/名人/事件/梗顺带点出，不确定就不写）\n"
    "槽点：这张图最值得打趣的一个点（梗在哪/反差在哪/最戳笑点的地方）\n"
    "要求：不要编造看不到的内容；涩图擦边只在'类型'和'槽点'点到程度，不要描述裸露细节。"
)


class AnimeCaptioner:
    """动漫图片识别 captioner。无本地状态，可并发，全程容错。"""

    def __init__(self, config: AiCompanionConfig) -> None:
        self.config = config
        # 缓存：content_hash -> (caption, expire_timestamp)
        self._cache: dict[str, tuple[str, float]] = {}
        # SauceNAO 配额状态（从响应头读取，耗尽时跳过）
        self._saucenao_short_remaining: int | None = None
        self._saucenao_long_remaining: int | None = None

    def reconfigure(self, config: AiCompanionConfig) -> None:
        self.config = config

    async def caption_media(self, part: dict[str, Any]) -> str:
        """识别图片，返回文字描述。失败/未命中返回 ''。"""
        if not getattr(self.config, "ai_companion_anime_caption_enabled", True):
            _debug_log(f"SKIP enabled=False type={part.get('type')}")
            return ""

        image_bytes, content_hash = await self._load_image(part)
        if not image_bytes:
            _debug_log(f"SKIP no_image type={part.get('type')} data_keys={list((part.get('data') or {}).keys())}")
            return ""

        # 缓存命中
        cached = self._cache.get(content_hash)
        if cached and cached[1] > time.time():
            _debug_log(f"CACHE_HIT hash={content_hash[:8]} caption={cached[0]!r}")
            return cached[0]

        min_sim = float(getattr(self.config, "ai_companion_anime_min_similarity", 80.0))
        # 搜索前压缩缩放：大图（如 9MB）直接上传 SauceNAO/iqdb 会超时，且命中率低。
        search_bytes = _downscale_for_search(image_bytes)
        _debug_log(
            f"SEARCH_START hash={content_hash[:8]} orig={len(image_bytes)} search={len(search_bytes)} min_sim={min_sim} saucenao={'yes' if self._saucenao_available() else 'no'}"
        )
        try:
            caption = await asyncio.wait_for(
                self._search_both(search_bytes, content_hash, min_sim),
                timeout=_TOTAL_TIMEOUT,
            )
        except TimeoutError:
            _debug_log(f"TIMEOUT hash={content_hash[:8]}")
            caption = ""
        except Exception as exc:  # noqa: BLE001 — captioner 必须容错，不阻塞主回复
            _debug_log(f"ERROR hash={content_hash[:8]} exc={exc!r}")
            caption = ""

        _debug_log(f"RESULT hash={content_hash[:8]} caption={caption!r}")
        # 写缓存（包括空结果，避免短时间内对同一张失败图反复请求）
        self._cache[content_hash] = (caption, time.time() + _CACHE_TTL)
        # 控制缓存规模
        if len(self._cache) > 500:
            now = time.time()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
        return caption

    async def caption_media_general(self, part: dict[str, Any]) -> str:
        """通用图片理解（回复路径用）：任意图 → 类型/图内文字/内容/槽点，供打趣回复。

        vision 优先（一次调用覆盖梗图/截图/热点/博客/二次元/擦边全类型）；
        仅当识别为二次元角色图且反搜源可用时，附加出处信息作为增强。
        """
        if not getattr(self.config, "ai_companion_anime_caption_enabled", True):
            return ""
        image_bytes, content_hash = await self._load_image(part)
        if not image_bytes:
            return ""
        cache_key = "general:" + content_hash
        cached = self._cache.get(cache_key)
        if cached and cached[1] > time.time():
            return cached[0]

        vision_bytes = _downscale_for_search(image_bytes, max_size=768)
        caption = ""
        try:
            caption = await asyncio.wait_for(
                self._vision_describe_image(vision_bytes, prompt_override=_GENERAL_REACTION_PROMPT, max_tokens=600),
                timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001 — captioner 必须容错，不阻塞主回复
            _debug_log(f"GENERAL_VISION_ERROR hash={content_hash[:8]} exc={exc!r}")
            caption = ""

        # 二次元角色图：反搜出处作为增强（best-effort，失败不影响已有描述）
        if caption and ("二次元" in caption or "动漫" in caption) and self._saucenao_available():
            try:
                source_caption = await asyncio.wait_for(
                    self._search_both(vision_bytes, content_hash, 80.0), timeout=8.0
                )
                if source_caption:
                    caption = f"{caption}\n出处参考：{source_caption[:120]}"
            except Exception:  # noqa: BLE001 — 增强失败静默（已有描述可用）
                pass

        _debug_log(f"GENERAL_RESULT hash={content_hash[:8]} caption={caption!r}")
        self._cache[cache_key] = (caption, time.time() + _CACHE_TTL)
        if len(self._cache) > 500:
            now = time.time()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
        return caption

    async def _load_image(self, part: dict[str, Any]) -> tuple[bytes, str]:
        """从 OneBot 段落加载图片字节 + 计算 content hash。失败返回 (b'', '')。"""
        uri = self._image_uri_from_part(part)
        if not uri:
            return b"", ""
        try:
            if uri.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=_PER_ROUTE_TIMEOUT, follow_redirects=True) as c:
                    r = await c.get(uri)
                    r.raise_for_status()
                    data = r.content
            else:
                path_str = uri[7:] if uri.startswith("file://") else uri
                path = Path(path_str)
                if not path.is_file():
                    return b"", ""
                data = path.read_bytes()
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.debug("anime_captioner: load image failed uri=%r err=%r", uri, exc)
            return b"", ""
        if not data:
            return b"", ""
        return data, hashlib.sha1(data).hexdigest()

    @staticmethod
    def _image_uri_from_part(part: dict[str, Any]) -> str:
        """从 OneBot 段落提取图片 URI（url/file/file_id/uri 任一）。"""
        if not isinstance(part, dict):
            return ""
        if str(part.get("type") or "") not in {"image", "mface", "emoji", "sticker"}:
            return ""
        data = part.get("data")
        if not isinstance(data, dict):
            return ""
        for field in ("url", "file", "file_id", "uri"):
            value = str(data.get(field) or "").strip()
            if value:
                return value
        return ""

    async def _search_both(self, image_bytes: bytes, content_hash: str, min_sim: float) -> str:
        """多路并发：SauceNAO + iqdb + GLM-V 视觉识图，合并所有非空结果。

        视觉模型理解任意图（角色/场景/动作），SauceNAO/iqdb 认知名角色（精确 tag），
        信息互补。任一路失败不影响其他。
        """
        tasks: list[asyncio.Task[str]] = []
        tasks.append(asyncio.create_task(self._search_iqdb(image_bytes, min_sim)))
        if self._saucenao_available():
            tasks.append(asyncio.create_task(self._search_saucenao(image_bytes, min_sim)))
        if self._vision_available():
            tasks.append(asyncio.create_task(self._vision_describe_image(image_bytes)))

        # 收集所有非空结果合并（视觉识图 + 以图搜源信息互补）
        results: list[str] = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
            except _SauceNaoQuotaExhausted:
                self._saucenao_long_remaining = 0
                continue
            except Exception as exc:  # noqa: BLE001
                logger.debug("anime_captioner: one route failed: %r", exc)
                continue
            if result:
                results.append(result)
        return "；".join(results)

    def _vision_available(self) -> bool:
        if not getattr(self.config, "ai_companion_anime_vision_enabled", True):
            return False
        # 需要 url + key（key 空则尝试复用 online_api_key）
        url = (getattr(self.config, "ai_companion_anime_vision_api_url", "") or "").strip()
        return bool(url)

    def _vision_creds(self) -> tuple[str, str]:
        url = (getattr(self.config, "ai_companion_anime_vision_api_url", "") or "").strip()
        key = (getattr(self.config, "ai_companion_anime_vision_api_key", "") or "").strip()
        # key 空则复用 GLM 在线推理或辅助训练的 key（同平台）
        if not key:
            key = (getattr(self.config, "ai_companion_online_api_key", "") or "").strip()
        if not key:
            key = (getattr(self.config, "ai_companion_lora_helper_api_key", "") or "").strip()
        return url, key

    async def _vision_describe_image(
        self, image_bytes: bytes, *, prompt_override: str = "", max_tokens: int = 1024
    ) -> str:
        """用 GLM-4.6V-Flash(免费视觉模型)理解图片内容。

        能识别任意图片（角色/场景/动作/文字），弥补 SauceNAO 对冷门图命中率低。
        prompt_override 提供时使用通用提示词（梗图/截图/热点打趣模式）。
        对 1305(服务过载)做退避重试。
        """
        url, key = self._vision_creds()
        if not url or not key:
            return ""
        model = (getattr(self.config, "ai_companion_anime_vision_model", "glm-4.6v-flash") or "glm-4.6v-flash").strip()

        # 缩放到视觉模型适中的尺寸（避免图太大传得慢）
        vision_bytes = _downscale_for_search(image_bytes, max_size=768)
        b64 = base64.b64encode(vision_bytes).decode("ascii")
        mime = "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_override
                            or (
                                "这是一张二次元/动漫图片。请用一句话描述：1) 如果能认出角色，"
                                "说角色名和出处作品（不确定就说'不确定角色'）；2) 简述画面"
                                "（人物外观/场景/动作/有文字则读出关键文字）。不要编造。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            # 思考型模型会把 max_tokens 烧在 reasoning 上导致 content 为空；
            # 端点不认识 thinking 字段时下面按 400 去参重试。
            "thinking": {"type": "disabled"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

        # 1305(服务过载)退避重试；其他错误直接放弃
        max_retries = 2
        async with httpx.AsyncClient(timeout=40.0) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 429:
                        body = response.json() if response.text else {}
                        code = str((body.get("error") or {}).get("code", ""))
                        if code == "1305" and attempt < max_retries:
                            await asyncio.sleep(min(4.0, 1.0 * (2**attempt)))
                            continue
                        _debug_log(f"VISION_429 code={code} body={response.text[:100]!r}")
                        return ""
                    if response.status_code == 400 and "thinking" in payload:
                        # 端点不认识 thinking 字段（部分 OpenAI 兼容后端）：去参重试
                        payload.pop("thinking", None)
                        _debug_log("VISION_THINKING_UNSUPPORTED retry without thinking")
                        continue
                    response.raise_for_status()
                    data = response.json()
                    content = _extract_chat_content(data)
                    content = content.strip()
                    if content and len(content) >= 4:
                        _debug_log(f"VISION_RESULT model={model} desc={content!r}")
                        return f"图片内容：{content}"
                    return ""
                except httpx.HTTPError as exc:
                    _debug_log(f"VISION_HTTP_ERR attempt={attempt} exc={exc!r}")
                    if attempt >= max_retries:
                        return ""
                    await asyncio.sleep(1.0)
        return ""

    def _saucenao_available(self) -> bool:
        if not getattr(self.config, "ai_companion_saucenao_api_key", "").strip():
            return False
        # 配额耗尽标记
        if self._saucenao_long_remaining is not None and self._saucenao_long_remaining <= 0:
            return False
        return True

    async def _search_iqdb(self, image_bytes: bytes, min_sim: float) -> str:
        """iqdb 单路搜索。POST 表单，解析 HTML。"""
        try:
            async with httpx.AsyncClient(timeout=_PER_ROUTE_TIMEOUT, follow_redirects=True) as c:
                r = await c.post(
                    "https://iqdb.org/",
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.debug("anime_captioner: iqdb request failed: %r", exc)
            return ""
        return _parse_iqdb_html(r.text, min_sim)

    async def _search_saucenao(self, image_bytes: bytes, min_sim: float) -> str:
        """SauceNAO 单路搜索。POST 表单，返回 JSON。"""
        api_key = self.config.ai_companion_saucenao_api_key.strip()
        dbmask = int(getattr(self.config, "ai_companion_saucenao_dbmask", 999))
        try:
            async with httpx.AsyncClient(timeout=_PER_ROUTE_TIMEOUT, follow_redirects=True) as c:
                r = await c.post(
                    "https://saucenao.com/search.php",
                    data={
                        "api_key": api_key,
                        "output_type": 2,
                        "numres": 3,
                        "dbmask": dbmask,
                    },
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
        except httpx.HTTPError as exc:
            logger.debug("anime_captioner: saucenao request failed: %r", exc)
            return ""
        try:
            payload = r.json()
        except ValueError:
            logger.debug("anime_captioner: saucenao non-json response")
            return ""

        header = payload.get("header") or {}
        # 更新配额状态
        self._saucenao_short_remaining = header.get("short_remaining")
        self._saucenao_long_remaining = header.get("long_remaining")
        status = header.get("status")
        if status != 0:
            # 配额耗尽或其他错误
            msg = header.get("message", "")
            if "limit" in str(msg).lower() or "rate" in str(msg).lower():
                raise _SauceNaoQuotaExhausted(msg)
            logger.info("anime_captioner: saucenao status=%s msg=%s", status, msg)
            return ""

        results = payload.get("results") or []
        for result in results:
            similarity_raw = (result.get("header") or {}).get("similarity", "0")
            try:
                similarity = float(similarity_raw)
            except (TypeError, ValueError):
                continue
            if similarity < min_sim:
                continue
            desc = _format_saucenao_result(result, similarity)
            if desc:
                return desc
        return ""


def _parse_iqdb_html(html: str, min_sim: float) -> str:
    """解析 iqdb 结果页 HTML，返回命中的文字描述。未命中返回 ''。"""
    # iqdb 结果在 <table class="etc"> 里，每行一个匹配，含相似度% 和来源链接
    # 相似度模式："similarity" 后跟百分比，或在单元格文本里 "XX% similarity"
    # 来源：href="https://danbooru.donmai.us/posts/123" 等
    best_similarity = 0.0
    best_source = ""

    # 提取所有图库链接 + 其所在区块的相似度
    # iqdb 页面结构：每个结果是一个 <table>，含缩略图、来源链接、相似度
    # 用正则切分结果区块
    # 先找所有"图库链接 + 相似度"配对
    source_pattern = re.compile(
        r'href="(https?://(?:www\.)?([^/]+)/(?:posts|post|index)\.php\?[^"]*|https?://(?:www\.)?([^/]+)/\d+)"',
        re.I,
    )
    sim_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*similar", re.I)

    # 按 <table 切块，逐块找相似度和来源
    blocks = re.split(r'<table[^>]*class="[^"]*pages', html)
    if len(blocks) == 1:
        blocks = re.split(r"<table", html)
    for block in blocks[1:]:
        # 跳过"your image"块（上传的原图）
        if "your image" in block.lower():
            continue
        sim_match = sim_pattern.search(block)
        if not sim_match:
            continue
        try:
            sim = float(sim_match.group(1))
        except ValueError:
            continue
        if sim < min_sim or sim <= best_similarity:
            continue
        src_match = source_pattern.search(block)
        if not src_match:
            continue
        host = (src_match.group(2) or src_match.group(3) or "").lower()
        source_name = _IQDB_SOURCE_HOSTS.get(host, host or "未知图库")
        best_similarity = sim
        best_source = source_name

    if best_similarity < min_sim or not best_source:
        return ""

    # iqdb 不直接给角色 tag，只给来源链接。从 URL 提取 post id 作参考
    parts = [f"二次元图片(相似度 {best_similarity:.0f}%)", f"来源 {best_source}"]
    return "，".join(parts)


def _format_saucenao_result(result: dict[str, Any], similarity: float) -> str:
    """把 SauceNAO 单条结果格式化成自然语言描述。"""
    data = result.get("data") or {}
    header = result.get("header") or {}
    index_name = header.get("index_name") or ""

    # 低于 80% 的命中不可靠，标注"疑似"提示 bot 这是猜测而非确定。
    confidence = "疑似" if similarity < 80 else ""
    prefix = (
        f"二次元图片(相似度 {similarity:.0f}%{('，' + confidence) if confidence else ''})"
        if confidence
        else f"二次元图片(相似度 {similarity:.0f}%)"
    )
    parts: list[str] = [prefix]

    # 角色 tag（Danbooru 风格，如 "hatsune_miku"）
    characters = data.get("characters") or ""
    if characters:
        # tag 用下划线分隔，转成可读名
        names = [n.replace("_", " ").strip() for n in str(characters).split(",") if n.strip()]
        if names:
            parts.append(f"角色 {'、'.join(names[:3])}")

    # 作品/来源（过滤掉 "original"，无意义）
    material = data.get("material") or ""
    if material:
        materials = [
            m.replace("_", " ").strip()
            for m in str(material).split(",")
            if m.strip() and m.strip().lower() != "original"
        ]
        if materials:
            parts.append(f"作品 {'、'.join(materials[:2])}")

    # 画师
    creator = data.get("creator") or data.get("member_name") or ""
    if creator:
        parts.append(f"画师 {creator}")

    # 来源图库（清理 index_name：'Index #9: Danbooru - xxx.jpg' -> 'Danbooru'）
    ext_urls = data.get("ext_urls") or []
    if ext_urls:
        clean_source = _clean_saucenao_index_name(index_name)
        parts.append(f"来源 {clean_source}")

    if len(parts) <= 1:
        return ""
    return "，".join(parts)


def _clean_saucenao_index_name(index_name: str) -> str:
    """把 SauceNAO 的 index_name 清理成干净的图库名。

    'Index #9: Danbooru - 345f1fc6fb8c3cff6b93b38a8cc6e3cc_0.jpg' -> 'Danbooru'
    'Pixiv Images' -> 'Pixiv'
    """
    if not index_name:
        return "图库"
    # 去掉 'Index #N: ' 前缀
    name = re.sub(r"^Index\s*#\d+\s*:\s*", "", index_name, flags=re.I)
    # 去掉 ' - xxx.jpg' 后缀（文件名）
    name = re.split(r"\s*-\s*", name)[0].strip()
    return name or "图库"
