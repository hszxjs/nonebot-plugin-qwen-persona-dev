from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, is_dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .config import LLAMA_MODEL_PRESETS
from .runtime_config import config_to_dict

SECRET_MARKERS = ("api_key", "token", "secret", "password")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ASSETS = {
    "/assets/vendor/react/react.production.min.js": (
        Path("node_modules/react/umd/react.production.min.js"),
        "application/javascript; charset=utf-8",
    ),
    "/assets/vendor/react-dom/react-dom.production.min.js": (
        Path("node_modules/react-dom/umd/react-dom.production.min.js"),
        "application/javascript; charset=utf-8",
    ),
    "/assets/vendor/dayjs/dayjs.min.js": (
        Path("node_modules/dayjs/dayjs.min.js"),
        "application/javascript; charset=utf-8",
    ),
    "/assets/vendor/antd/reset.css": (
        Path("node_modules/antd/dist/reset.css"),
        "text/css; charset=utf-8",
    ),
    "/assets/vendor/antd/antd.min.js": (
        Path("node_modules/antd/dist/antd.min.js"),
        "application/javascript; charset=utf-8",
    ),
}
CONFIG_LABELS = {
    "qwen_persona_runtime": "模型运行方式",
    "qwen_persona_api_base": "模型接口地址",
    "qwen_persona_api_key": "模型接口密钥",
    "qwen_persona_model": "外部 API 模型名称",
    "qwen_persona_request_timeout": "模型请求超时",
    "qwen_persona_temperature": "回复随机度",
    "qwen_persona_top_p": "Top P 采样",
    "qwen_persona_max_tokens": "最大回复长度",
    "qwen_persona_llama_server_path": "llama.cpp 程序路径",
    "qwen_persona_llama_auto_install": "自动安装 llama.cpp",
    "qwen_persona_llama_model": "llama.cpp 模型",
    "qwen_persona_llama_model_file": "本地 GGUF 文件",
    "qwen_persona_llama_host": "llama.cpp 监听地址",
    "qwen_persona_llama_port": "llama.cpp 监听端口",
    "qwen_persona_llama_ctx_size": "上下文窗口",
    "qwen_persona_llama_extra_args": "llama.cpp 额外参数",
    "qwen_persona_admin_enabled": "启用后台网页",
    "qwen_persona_admin_host": "后台监听地址",
    "qwen_persona_admin_port": "后台监听端口",
    "qwen_persona_admin_token": "后台访问令牌",
    "qwen_persona_admin_config_file": "后台配置覆盖文件",
    "qwen_persona_model_dir": "本地模型目录",
    "qwen_persona_data_dir": "插件数据目录",
    "qwen_persona_enabled_groups": "启用群列表",
    "qwen_persona_enable_learning": "启用群聊学习",
    "qwen_persona_enable_reply": "启用自动回复",
    "qwen_persona_learning_priority": "学习处理优先级",
    "qwen_persona_reply_priority": "回复处理优先级",
    "qwen_persona_reply_probability": "随机回复概率",
    "qwen_persona_follow_reply_probability": "普通跟风概率",
    "qwen_persona_text_sticker_reply_probability": "文本消息表情回复概率",
    "qwen_persona_trigger_keywords": "触发关键词",
    "qwen_persona_enable_repeat": "参与群友复读",
    "qwen_persona_repeat_min_count": "复读触发次数",
    "qwen_persona_repeat_probability": "复读参与概率",
    "qwen_persona_repeat_cooldown_seconds": "复读冷却时间",
    "qwen_persona_pattern_reply_threshold": "聊天组合触发次数",
    "qwen_persona_context_scope": "上下文范围",
    "qwen_persona_context_messages": "上下文消息数",
    "qwen_persona_min_sample_chars": "最短学习样本",
    "qwen_persona_max_sample_chars": "最长学习样本",
    "qwen_persona_command_prefixes": "命令前缀",
    "qwen_persona_command_regexes": "命令过滤正则",
    "qwen_persona_auto_read_picmenu_config": "读取 picmenu 配置",
    "qwen_persona_picmenu_config_keys": "picmenu 配置字段",
    "qwen_persona_picmenu_command_keys": "picmenu 命令字段",
    "qwen_persona_sticker_segment_types": "表情消息类型",
    "qwen_persona_sticker_prompt_limit": "提示表情数量",
    "qwen_persona_sticker_repo_dir": "表情包仓库目录",
    "qwen_persona_sticker_repo_url": "表情包仓库地址",
    "qwen_persona_enable_sending_stickers": "允许发送表情",
    "qwen_persona_max_reply_stickers": "单次最多表情",
    "qwen_persona_vision_enabled": "启用视觉理解",
    "qwen_persona_vision_auto_start": "自动启动视觉服务",
    "qwen_persona_vision_python": "视觉服务 Python",
    "qwen_persona_vision_host": "视觉服务监听地址",
    "qwen_persona_vision_port": "视觉服务监听端口",
    "qwen_persona_vision_model": "视觉理解模型",
    "qwen_persona_vision_timeout": "视觉请求超时",
    "qwen_persona_vision_image_fallback_text": "图片降级文本",
    "qwen_persona_vision_sticker_fallback_text": "表情降级文本",
    "qwen_persona_enable_training": "启用定时训练",
    "qwen_persona_train_times": "训练时间点",
    "qwen_persona_lora_train_command": "本地 LoRA 训练命令",
    "qwen_persona_after_train_command": "训练后命令",
    "qwen_persona_train_cwd": "训练工作目录",
    "qwen_persona_train_timeout_seconds": "训练超时时间",
    "qwen_persona_dataset_limit": "训练样本上限",
    "qwen_persona_dataset_history_window": "训练上下文窗口",
    "qwen_persona_external_corpus_enabled": "合并外部语料",
    "qwen_persona_external_corpus_limit": "外部语料上限",
    "qwen_persona_min_train_samples": "最少训练样本",
    "qwen_persona_lora_api_url": "LoRA 训练接口地址",
    "qwen_persona_lora_api_key": "LoRA 训练接口密钥",
    "qwen_persona_lora_api_timeout": "LoRA 训练接口超时",
    "qwen_persona_lora_api_extra_body": "LoRA 训练接口附加参数",
    "qwen_persona_lora_helper_api_url": "GPT 辅助学习接口地址",
    "qwen_persona_lora_helper_api_key": "GPT 辅助学习接口密钥",
    "qwen_persona_lora_helper_model": "GPT 辅助学习模型",
    "qwen_persona_lora_helper_timeout": "GPT 辅助学习超时",
    "qwen_persona_lora_helper_sample_limit": "GPT 辅助样本上限",
    "qwen_persona_lora_helper_extra_body": "GPT 辅助接口附加参数",
    "qwen_persona_lora_helper_system_prompt": "GPT 辅助清洗提示词",
    "qwen_persona_blacklist_words": "黑名单词汇",
    "qwen_persona_blacklist_regexes": "黑名单正则",
    "qwen_persona_blacklist_user_ids": "黑名单用户",
    "qwen_persona_blacklist_group_ids": "黑名单群",
    "qwen_persona_system_prompt": "人格系统提示词",
}

CONFIG_DESCRIPTIONS = {
    "qwen_persona_runtime": "llama_cpp 由插件安装/启动/控制；external_llama 连接你自己后台运行的 llama.cpp；external_api 调用任意外部 OpenAI-compatible 接口。",
    "qwen_persona_api_base": "仅 external_api 模式使用。llama_cpp 模式会根据监听地址和端口自动生成接口地址。",
    "qwen_persona_llama_server_path": "可填写 llama-server 或 llama 可执行文件路径；留空时会搜索插件 bin 目录和 PATH。",
    "qwen_persona_llama_auto_install": "找不到 llama.cpp 时，按当前平台下载官方 release 预编译包到数据目录。",
    "qwen_persona_llama_model": "可选择预设，也可输入任意 Hugging Face 模型 ID、repo:quant，或 GGUF 文件名。",
    "qwen_persona_llama_model_file": "可填绝对路径，或相对模型目录的 GGUF 文件名；填写后优先使用本地文件。",
    "qwen_persona_llama_extra_args": "JSON 数组或字符串，透传给 llama.cpp，例如 [\"--n-gpu-layers\", \"99\"]。",
    "qwen_persona_auto_read_picmenu_config": "开启后自动扫描 picmenu 配置项，识别需要过滤学习的本地命令。",
    "qwen_persona_picmenu_config_keys": "包含这些名称的 NoneBot 配置字段会被当作 picmenu 配置扫描。",
    "qwen_persona_picmenu_command_keys": "picmenu 配置中这些键名下的值会被识别为命令、别名或触发词。",
    "qwen_persona_lora_helper_api_url": "OpenAI-compatible chat/completions 地址，用于调用 GPT-5.5 Medium 辅助清洗 LoRA 数据。",
    "qwen_persona_lora_helper_api_key": "发送给辅助学习接口的 Bearer Token。",
    "qwen_persona_lora_helper_model": "默认 gpt-5.5-medium，可按你的服务实际模型名修改。",
    "qwen_persona_lora_api_url": "最终 LoRA 训练任务接口；辅助清洗后的数据集会作为 dataset 字段传入。",
    "qwen_persona_context_scope": "group 只取当前群，global 取全部群，mixed 保留当前群并混入其他群样本。",
    "qwen_persona_external_corpus_enabled": "开启后训练导出会把 external_corpus_samples 表中的外部公开语料合并进数据集。",
    "qwen_persona_external_corpus_limit": "每次训练最多合并多少条外部语料，0 表示不限制；不会影响群聊自学习样本保存。",
    "qwen_persona_blacklist_words": "包含这些词的消息不会学习，回复中命中的行也会被丢弃。",
    "qwen_persona_blacklist_regexes": "用于过滤学习和回复的正则表达式。",
    "qwen_persona_sticker_repo_dir": "学习到的图片表情会保存到 media/，并写入 manifest.jsonl。",
    "qwen_persona_vision_enabled": "开启后图片和表情会先被视觉服务理解成文本，再进入训练样本；不会把 url、file_id 或 base64 写进训练文本。",
    "qwen_persona_vision_auto_start": "开启后插件启动时会自管一个本地视觉服务；依赖缺失时服务仍会安全降级为占位描述。",
    "qwen_persona_vision_python": "留空使用当前 Python；如需独立虚拟环境，可填该环境的 python 可执行文件。",
    "qwen_persona_vision_model": "内置服务默认使用 BLIP 等 image-to-text 模型生成描述；显式配置 CLIP 模型时才使用零样本分类。",
    "qwen_persona_follow_reply_probability": "群聊普通消息不 @ 机器人时，也会按此概率触发跟风回复。",
    "qwen_persona_learning_priority": "学习 matcher 的优先级。默认靠后，让其他插件命令先匹配并阻断，避免命令进入学习数据。",
    "qwen_persona_enable_repeat": "检测到群友连续发送相同文本或表情时，bot 会按概率参与复读。",
    "qwen_persona_repeat_min_count": "同一群内连续出现多少次相同消息后视为复读。",
}


RESTART_HINTS = {
    "qwen_persona_runtime",
    "qwen_persona_llama_server_path",
    "qwen_persona_llama_auto_install",
    "qwen_persona_llama_model",
    "qwen_persona_llama_model_file",
    "qwen_persona_llama_host",
    "qwen_persona_llama_port",
    "qwen_persona_llama_ctx_size",
    "qwen_persona_llama_extra_args",
    "qwen_persona_api_key",
    "qwen_persona_model_dir",
    "qwen_persona_admin_enabled",
    "qwen_persona_admin_host",
    "qwen_persona_admin_port",
    "qwen_persona_data_dir",
    "qwen_persona_train_times",
    "qwen_persona_enable_training",
}

ADMIN_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Persona 后台</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f7f4;
  --panel: #ffffff;
  --line: #d7ddd4;
  --text: #18201c;
  --muted: #647067;
  --accent: #167c62;
  --accent-2: #b35b18;
  --danger: #b42318;
  --field: #fbfcfa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  position: sticky;
  top: 0;
  z-index: 2;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  align-items: center;
  padding: 14px 22px;
  border-bottom: 1px solid var(--line);
  background: rgba(246, 247, 244, .95);
  backdrop-filter: blur(10px);
}
h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
button, input, textarea, select {
  font: inherit;
  border: 1px solid var(--line);
  border-radius: 6px;
}
button {
  min-height: 36px;
  padding: 0 12px;
  background: var(--panel);
  color: var(--text);
  cursor: pointer;
}
button.primary { background: var(--accent); color: white; border-color: var(--accent); }
button.warning { color: var(--accent-2); }
input[type="search"], input[type="password"], input[type="text"] {
  height: 36px;
  padding: 0 10px;
  background: var(--field);
}
select, option {
  background: var(--field);
  color: var(--text);
}
main { display: grid; grid-template-columns: 220px 1fr; gap: 18px; padding: 18px 22px 28px; }
nav {
  position: sticky;
  top: 78px;
  align-self: start;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 6px;
  overflow: hidden;
}
nav a {
  display: block;
  padding: 10px 12px;
  color: var(--text);
  text-decoration: none;
  border-bottom: 1px solid var(--line);
}
nav a:last-child { border-bottom: 0; }
nav a:hover { background: #eef4ef; }
section {
  margin: 0 0 16px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  overflow: hidden;
}
section h2 {
  margin: 0;
  padding: 12px 14px;
  font-size: 15px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfa;
}
.field {
  display: grid;
  grid-template-columns: minmax(220px, 34%) 1fr;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid #edf0eb;
}
.field:last-child { border-bottom: 0; }
label { font-weight: 650; overflow-wrap: anywhere; }
.env { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; font-weight: 500; }
.desc { display: block; margin-top: 5px; color: var(--muted); font-size: 12px; font-weight: 400; }
.editor { min-width: 0; }
.editor input[type="number"], .editor input[type="text"], .editor input[type="password"], .editor textarea {
  width: 100%;
  background: var(--field);
  color: var(--text);
}
.editor input[type="number"], .editor input[type="text"], .editor input[type="password"] { height: 36px; padding: 0 10px; }
.editor textarea { min-height: 72px; resize: vertical; padding: 9px 10px; }
.switch { display: inline-flex; align-items: center; gap: 8px; min-height: 36px; }
.switch input { width: 18px; height: 18px; }
.status { color: var(--muted); min-height: 20px; }
.status.error { color: var(--danger); }
.status.ok { color: var(--accent); }
.hidden { display: none; }
@media (max-width: 860px) {
  header { grid-template-columns: 1fr; }
  .toolbar { justify-content: stretch; }
  .toolbar input, .toolbar button { flex: 1 1 auto; }
  main { grid-template-columns: 1fr; padding: 12px; }
  nav { position: static; }
  .field { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Qwen Persona 后台</h1>
  <div class="toolbar">
    <input id="token" type="password" placeholder="Admin Token">
    <input id="search" type="search" placeholder="搜索配置项">
    <button id="reload">刷新</button>
    <button id="save" class="primary">保存配置</button>
  </div>
</header>
<main>
  <nav id="nav"></nav>
  <div>
    <p id="status" class="status"></p>
    <div id="sections"></div>
  </div>
</main>
<script>
const categoryOrder = ["后台网页", "模型部署", "模型调用", "学习与回复", "表情包", "LoRA 训练", "黑名单", "其他"];
let fields = [];
let values = {};

function tokenQuery() {
  const token = document.getElementById("token").value.trim();
  return token ? `?token=${encodeURIComponent(token)}` : "";
}

function setStatus(text, kind = "") {
  const node = document.getElementById("status");
  node.textContent = text;
  node.className = `status ${kind}`;
}

function fieldType(field) {
  if (field.kind === "bool") return "bool";
  if (field.kind === "int" || field.kind === "float") return "number";
  if (field.kind === "json") return "json";
  return field.multiline ? "textarea" : "text";
}

function render() {
  const query = document.getElementById("search").value.trim().toLowerCase();
  const sections = new Map();
  for (const field of fields) {
    const haystack = `${field.name} ${field.title} ${field.description} ${field.env} ${field.category}`.toLowerCase();
    if (query && !haystack.includes(query)) continue;
    if (!sections.has(field.category)) sections.set(field.category, []);
    sections.get(field.category).push(field);
  }

  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  for (const category of categoryOrder) {
    if (!sections.has(category)) continue;
    const a = document.createElement("a");
    a.href = `#${category}`;
    a.textContent = `${category} (${sections.get(category).length})`;
    nav.appendChild(a);
  }

  const root = document.getElementById("sections");
  root.innerHTML = "";
  for (const category of categoryOrder) {
    const items = sections.get(category);
    if (!items || !items.length) continue;
    const section = document.createElement("section");
    section.id = category;
    const title = document.createElement("h2");
    title.textContent = category;
    section.appendChild(title);
    for (const field of items) section.appendChild(renderField(field));
    root.appendChild(section);
  }
}

function renderField(field) {
  const row = document.createElement("div");
  row.className = "field";
  row.dataset.name = field.name;

  const label = document.createElement("label");
  label.textContent = field.title || field.name;
  const env = document.createElement("span");
  env.className = "env";
  env.textContent = field.env;
  label.appendChild(env);
  if (field.description) {
    const desc = document.createElement("span");
    desc.className = "desc";
    desc.textContent = field.description;
    label.appendChild(desc);
  }
  row.appendChild(label);

  const editor = document.createElement("div");
  editor.className = "editor";
  const type = fieldType(field);
  const value = values[field.name];

  if (field.widget === "combo" && !field.allowCustom) {
    const select = document.createElement("select");
    select.dataset.name = field.name;
    for (const option of field.options || []) {
      const node = document.createElement("option");
      node.value = option.value;
      node.textContent = option.label || option.value;
      node.selected = option.value === value;
      select.appendChild(node);
    }
    editor.appendChild(select);
  } else if (field.widget === "combo") {
    const input = document.createElement("input");
    const listId = `list-${field.name}`;
    input.type = "text";
    input.value = value ?? "";
    input.setAttribute("list", listId);
    input.dataset.name = field.name;
    const list = document.createElement("datalist");
    list.id = listId;
    for (const option of field.options || []) {
      const node = document.createElement("option");
      node.value = option.value;
      node.label = option.label || option.value;
      node.title = option.description || "";
      list.appendChild(node);
    }
    editor.appendChild(input);
    editor.appendChild(list);
  } else if (type === "bool") {
    const wrap = document.createElement("label");
    wrap.className = "switch";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value);
    input.dataset.name = field.name;
    wrap.appendChild(input);
    wrap.appendChild(document.createTextNode(input.checked ? "开启" : "关闭"));
    input.addEventListener("change", () => { wrap.lastChild.textContent = input.checked ? "开启" : "关闭"; });
    editor.appendChild(wrap);
  } else if (type === "number") {
    const input = document.createElement("input");
    input.type = "number";
    input.step = field.kind === "float" ? "0.01" : "1";
    input.value = value ?? "";
    input.dataset.name = field.name;
    editor.appendChild(input);
  } else if (type === "json") {
    const textarea = document.createElement("textarea");
    textarea.spellcheck = false;
    textarea.value = JSON.stringify(value ?? (field.default ?? []), null, 2);
    textarea.dataset.name = field.name;
    editor.appendChild(textarea);
  } else if (type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.spellcheck = false;
    textarea.value = value ?? "";
    textarea.dataset.name = field.name;
    editor.appendChild(textarea);
  } else {
    const input = document.createElement("input");
    input.type = field.secret ? "password" : "text";
    input.value = value ?? "";
    input.dataset.name = field.name;
    editor.appendChild(input);
  }

  row.appendChild(editor);
  return row;
}

function readValues() {
  const output = {};
  for (const field of fields) {
    const input = document.querySelector(
      `input[data-name="${field.name}"], textarea[data-name="${field.name}"], select[data-name="${field.name}"]`
    );
    if (!input) continue;
    const type = fieldType(field);
    if (type === "bool") output[field.name] = input.checked;
    else if (type === "number") output[field.name] = field.kind === "int" ? parseInt(input.value || "0", 10) : parseFloat(input.value || "0");
    else if (type === "json") {
      const raw = input.value.trim();
      output[field.name] = raw ? JSON.parse(raw) : [];
    } else output[field.name] = input.value;
  }
  return output;
}

async function loadConfig() {
  setStatus("读取中...");
  const response = await fetch(`/api/config${tokenQuery()}`);
  if (response.status === 401) {
    setStatus("需要 Admin Token", "error");
    return;
  }
  if (!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  fields = payload.fields;
  values = payload.values;
  render();
  setStatus(`已加载 ${fields.length} 个配置项，覆盖文件：${payload.override_file}`, "ok");
}

async function saveConfig() {
  try {
    setStatus("保存中...");
    const payload = { values: readValues() };
    const response = await fetch(`/api/config${tokenQuery()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (response.status === 401) {
      setStatus("需要 Admin Token", "error");
      return;
    }
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    values = result.values;
    render();
    const restart = result.restart_required.length ? `；需重启：${result.restart_required.join(", ")}` : "";
    setStatus(`已保存，变更 ${result.changed.length} 项${restart}`, "ok");
  } catch (error) {
    setStatus(String(error), "error");
  }
}

document.getElementById("save").addEventListener("click", saveConfig);
document.getElementById("reload").addEventListener("click", loadConfig);
document.getElementById("search").addEventListener("input", render);
loadConfig().catch(error => setStatus(String(error), "error"));
</script>
</body>
</html>
"""

ADMIN_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Persona 控制台</title>
<link rel="stylesheet" href="/assets/vendor/antd/reset.css">
<script src="/assets/vendor/react/react.production.min.js"></script>
<script src="/assets/vendor/react-dom/react-dom.production.min.js"></script>
<script src="/assets/vendor/dayjs/dayjs.min.js"></script>
<script src="/assets/vendor/antd/antd.min.js"></script>
<style>
:root {
  color-scheme: light;
  --bg: #f8fafc;
  --surface: #ffffff;
  --surface-2: #f8fafc;
  --surface-3: #f1f5f9;
  --text: #0f172a;
  --muted: #64748b;
  --line: #d7dee8;
  --line-soft: #e8edf3;
  --signal: #0f766e;
  --signal-soft: #e7f7f4;
  --amber: #a16207;
  --amber-soft: #fef3c7;
  --danger: #b91c1c;
  --danger-soft: #fee2e2;
  --terminal: #0f172a;
  --terminal-line: #1e293b;
  --terminal-text: #e2e8f0;
  --focus: #14b8a6;
  --radius: 8px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100dvh;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: inherit; }
button, input, textarea, select {
  font: inherit;
  border: 1px solid var(--line);
  border-radius: var(--radius);
}
button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible, a:focus-visible {
  outline: 3px solid rgba(20, 184, 166, .34);
  outline-offset: 2px;
}
button {
  min-height: 44px;
  padding: 0 14px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  transition: border-color .18s ease, background-color .18s ease, color .18s ease, opacity .18s ease;
}
button:hover:not(:disabled) { border-color: var(--signal); background: var(--surface-2); }
button:disabled { cursor: not-allowed; opacity: .5; }
button.primary { background: var(--signal); color: white; border-color: var(--signal); }
button.warning { color: var(--amber); border-color: #e4c574; }
button.danger { color: var(--danger); border-color: #efb0b0; }
input[type="search"], input[type="password"], input[type="text"], input[type="number"], select {
  min-height: 44px;
  padding: 0 11px;
  background: var(--surface);
  color: var(--text);
}
option {
  background: var(--surface);
  color: var(--text);
}
textarea {
  width: 100%;
  min-height: 88px;
  padding: 10px 11px;
  resize: vertical;
  background: var(--surface);
  color: var(--text);
}
.app-header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  align-items: center;
  min-height: 68px;
  padding: 12px 22px;
  border-bottom: 1px solid var(--line);
  background: rgba(248, 250, 252, .94);
  backdrop-filter: blur(14px);
}
.brand { display: flex; gap: 12px; align-items: center; min-width: 0; }
.brand-mark {
  width: 36px;
  height: 36px;
  border-radius: var(--radius);
  border: 1px solid var(--line);
  background:
    linear-gradient(135deg, rgba(15, 118, 110, .12), rgba(161, 98, 7, .12)),
    var(--surface);
}
h1 { margin: 0; font-size: 18px; line-height: 1.2; letter-spacing: 0; }
.subtitle { margin-top: 2px; color: var(--muted); font-size: 12px; }
.header-tools { display: flex; gap: 8px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
.nav-tabs {
  position: fixed;
  top: 68px;
  bottom: 0;
  left: 0;
  z-index: 9;
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 232px;
  padding: 16px 12px;
  border-right: 1px solid var(--line);
  background: var(--surface);
}
.nav-tabs a {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 44px;
  padding: 0 12px;
  border: 1px solid transparent;
  border-radius: var(--radius);
  color: var(--muted);
  font-weight: 650;
  text-decoration: none;
}
.nav-tabs a::before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: currentColor;
  opacity: .42;
}
body[data-page="dashboard"] a[href="/"],
body[data-page="stickers"] a[href="/stickers"],
body[data-page="settings"] a[href="/settings"] {
  color: var(--text);
  border-color: var(--line);
  background: var(--surface-3);
}
main { margin-left: 232px; padding: 20px 22px 30px; }
.view { display: none; }
body[data-page="dashboard"] #dashboard-view,
body[data-page="stickers"] #stickers-view,
body[data-page="settings"] #settings-view { display: block; }
body[data-page="dashboard"] .settings-only,
body[data-page="stickers"] .settings-only,
body[data-page="settings"] .dashboard-only { display: none; }
.dashboard-grid {
  display: grid;
  grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.dashboard-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 16px;
  align-items: end;
  margin-bottom: 16px;
}
.page-kicker {
  margin: 0 0 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0;
  text-transform: uppercase;
}
.page-title {
  margin: 0;
  font-size: 26px;
  line-height: 1.2;
  letter-spacing: 0;
}
.page-copy { margin: 6px 0 0; color: var(--muted); }
.hero-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: flex-end;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.metric-card {
  min-height: 116px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  padding: 14px;
}
.metric-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 750;
}
.metric-value {
  margin-top: 10px;
  font-size: 28px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: 0;
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
.metric-note {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}
.dashboard-stack { display: grid; gap: 16px; }
.panel {
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  overflow: hidden;
}
.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  padding: 13px 14px;
  border-bottom: 1px solid var(--line-soft);
  background: var(--surface-2);
}
.panel-head h2 { margin: 0; font-size: 15px; letter-spacing: 0; }
.panel-body { padding: 14px; }
.segmented {
  display: inline-flex;
  gap: 3px;
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
}
.segmented button {
  min-height: 34px;
  padding: 0 10px;
  border: 0;
  color: var(--muted);
  background: transparent;
}
.segmented button.active {
  color: var(--text);
  background: var(--surface-3);
}
.chart-wrap {
  height: 280px;
  padding: 14px;
}
.chart-wrap svg {
  width: 100%;
  height: 100%;
  display: block;
}
.chart-axis { stroke: var(--line-soft); stroke-width: 1; }
.chart-bar { fill: var(--signal); opacity: .88; }
.chart-line { fill: none; stroke: var(--signal); stroke-width: 2.5; }
.chart-point { fill: var(--surface); stroke: var(--signal); stroke-width: 2; }
.chart-label { fill: var(--muted); font: 11px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.status-spine {
  display: grid;
  gap: 12px;
  border-left: 4px solid var(--line);
  padding-left: 12px;
}
.status-spine.ok { border-left-color: var(--signal); }
.status-spine.warn { border-left-color: var(--amber); }
.status-spine.error { border-left-color: var(--danger); }
.state-line { display: flex; gap: 9px; align-items: center; min-width: 0; }
.state-dot { width: 10px; height: 10px; border-radius: 999px; background: var(--line); flex: 0 0 auto; }
.status-spine.ok .state-dot { background: var(--signal); }
.status-spine.warn .state-dot { background: var(--amber); }
.status-spine.error .state-dot { background: var(--danger); }
.state-name { font-size: 22px; font-weight: 750; line-height: 1.2; overflow-wrap: anywhere; }
.state-detail { color: var(--muted); overflow-wrap: anywhere; }
.fact-grid { display: grid; gap: 1px; margin-top: 14px; border: 1px solid var(--line-soft); background: var(--line-soft); }
.fact {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: 10px;
  padding: 9px 10px;
  background: var(--surface);
}
.fact dt { margin: 0; color: var(--muted); font-weight: 650; }
.fact dd { margin: 0; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.control-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 14px; }
.model-switch {
  display: grid;
  gap: 9px;
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--line-soft);
}
.model-switch label {
  display: grid;
  gap: 5px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.model-actions { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 2px; }
.notice {
  min-height: 24px;
  margin-top: 12px;
  color: var(--muted);
  overflow-wrap: anywhere;
}
.notice.error { color: var(--danger); }
.notice.ok { color: var(--signal); }
.log-tools { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.terminal {
  height: min(62dvh, 680px);
  min-height: 360px;
  overflow: auto;
  background: var(--terminal);
  color: var(--terminal-text);
  font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.terminal.compact {
  height: 260px;
  min-height: 220px;
}
.log-line {
  display: grid;
  grid-template-columns: 172px 64px minmax(0, 1fr);
  gap: 10px;
  padding: 4px 12px;
  border-bottom: 1px solid var(--terminal-line);
}
.log-line .stream { color: #8fdbe1; }
.log-line .time { color: #91a39e; }
.log-line .message { white-space: pre-wrap; overflow-wrap: anywhere; }
.empty {
  padding: 28px 14px;
  color: var(--muted);
  text-align: center;
}
.settings-shell {
  display: grid;
  grid-template-columns: 230px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.settings-nav {
  position: sticky;
  top: 134px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--surface);
  overflow: hidden;
}
.settings-nav a {
  display: block;
  padding: 11px 12px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text);
  text-decoration: none;
}
.settings-nav a:last-child { border-bottom: 0; }
.settings-nav a:hover { background: var(--signal-soft); }
#settings-sections {
  display: grid;
  gap: 16px;
}
.field {
  display: grid;
  grid-template-columns: minmax(210px, 34%) minmax(0, 1fr);
  gap: 12px;
  padding: 13px 14px;
  border-bottom: 1px solid var(--line-soft);
}
.field:last-child { border-bottom: 0; }
label.field-label { font-weight: 700; overflow-wrap: anywhere; }
.env { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; font-weight: 550; }
.desc { display: block; margin-top: 6px; color: var(--muted); font-size: 12px; font-weight: 400; }
.editor { min-width: 0; }
.editor input[type="number"], .editor input[type="text"], .editor input[type="password"], .editor select { width: 100%; }
.switch { display: inline-flex; align-items: center; gap: 9px; min-height: 44px; }
.switch input { width: 20px; height: 20px; }
.sticker-react-root {
  min-height: min(760px, calc(100dvh - 178px));
}
.sticker-gallery-card {
  height: min(760px, calc(100dvh - 178px));
  min-height: 560px;
}
.sticker-gallery-card > .ant-card-body {
  height: calc(100% - 57px);
  display: grid;
  grid-template-rows: auto auto auto minmax(0, 1fr) auto;
  gap: 12px;
  overflow: hidden;
}
.sticker-gallery-toolbar {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) 180px 144px;
  gap: 10px;
}
.sticker-type-strip {
  display: flex;
  gap: 8px;
  align-items: center;
  overflow-x: auto;
  min-width: 0;
}
.selection-count {
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  align-self: center;
}
.sticker-gallery-table { min-height: 0; }
.sticker-table-media {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  min-width: 0;
}
.sticker-thumb {
  width: 48px;
  height: 48px;
  padding: 0;
  overflow: hidden;
  display: grid;
  place-items: center;
  border: 1px solid #f0f0f0;
  border-radius: 6px;
  background:
    linear-gradient(45deg, #eef2f7 25%, transparent 25% 75%, #eef2f7 75%),
    linear-gradient(45deg, #eef2f7 25%, transparent 25% 75%, #eef2f7 75%),
    #fbfcfe;
  background-position: 0 0, 8px 8px;
  background-size: 16px 16px;
}
.sticker-thumb img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}
.sticker-fallback {
  display: grid;
  gap: 4px;
  place-items: center;
  color: var(--muted);
  text-align: center;
  padding: 10px;
}
.sticker-fallback strong {
  color: var(--text);
  font-size: 18px;
  font-variant-numeric: tabular-nums;
}
.sticker-title {
  color: rgba(0, 0, 0, .88);
  font-size: 14px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sticker-subtitle {
  margin-top: 4px;
  color: rgba(0, 0, 0, .45);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sticker-gallery-pager {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  align-items: center;
}
.sticker-preview-layout {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  gap: 18px;
  min-height: 520px;
}
.preview-media {
  min-height: 0;
  display: grid;
  place-items: center;
  padding: 24px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  background: #fafafa;
}
.preview-media img, .preview-media .ant-image-img {
  max-width: 100%;
  max-height: 500px;
  object-fit: contain;
}
.preview-detail {
  min-width: 0;
  max-height: 520px;
  overflow: auto;
}
@media (max-width: 920px) {
  .app-header { grid-template-columns: 1fr; align-items: stretch; padding: 12px; }
  .header-tools { justify-content: stretch; }
  .header-tools input, .header-tools button { flex: 1 1 auto; }
  .nav-tabs {
    position: static;
    width: auto;
    flex-direction: row;
    padding: 8px 12px;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  main { margin-left: 0; padding: 12px; }
  .dashboard-hero { grid-template-columns: 1fr; }
  .hero-actions { justify-content: stretch; }
  .hero-actions button { flex: 1 1 auto; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dashboard-grid, .settings-shell, .preview-panel { grid-template-columns: 1fr; }
  .settings-nav { position: static; }
  .sticker-gallery-toolbar { grid-template-columns: 1fr; }
  .sticker-preview-layout { grid-template-columns: 1fr; }
  .field { grid-template-columns: 1fr; }
  .control-row { grid-template-columns: 1fr; }
  .log-line { grid-template-columns: 1fr; gap: 2px; padding: 8px 10px; }
}
@media (max-width: 560px) {
  .metric-grid { grid-template-columns: 1fr; }
  .nav-tabs a { flex: 1 1 0; justify-content: center; }
  .chart-wrap { height: 220px; }
}
</style>
</head>
<body>
<header class="app-header">
  <div class="brand">
    <div class="brand-mark" aria-hidden="true"></div>
    <div>
      <h1>Qwen Persona 控制台</h1>
      <div class="subtitle">本地模型运行、日志与插件设置</div>
    </div>
  </div>
  <div class="header-tools">
    <input id="token" type="password" placeholder="Admin Token" autocomplete="current-password">
    <input id="search" class="settings-only" type="search" placeholder="搜索设置项">
    <button id="refresh" type="button">刷新</button>
    <button id="save" class="primary settings-only" type="button">保存设置</button>
  </div>
</header>
<nav class="nav-tabs" aria-label="主导航">
  <a href="/">运行</a>
  <a href="/stickers">表情库</a>
  <a href="/settings">设置</a>
</nav>
<main>
  <section id="dashboard-view" class="view" aria-labelledby="dashboard-title">
    <div class="dashboard-hero">
      <div>
        <p class="page-kicker">Dashboard</p>
        <h2 id="dashboard-title" class="page-title">运行概览</h2>
        <p class="page-copy">人格学习、模型推理和本地 llama.cpp 服务的关键状态。</p>
      </div>
      <div class="hero-actions">
        <button id="training-run" class="primary" type="button">立即训练</button>
        <button id="training-run-external" type="button">全量外部语料训练</button>
      </div>
    </div>
    <div id="metric-tiles" class="metric-grid" aria-live="polite"></div>
    <div class="dashboard-grid">
      <div class="dashboard-stack">
        <div class="panel">
          <div class="panel-head">
            <h2>训练进度</h2>
          </div>
          <div class="panel-body">
            <dl id="training-progress" class="fact-grid"></dl>
            <pre id="training-log-tail" class="terminal compact"></pre>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h2>llama.cpp 状态</h2>
          </div>
          <div class="panel-body">
            <div id="llama-status" class="status-spine">
              <div class="state-line">
                <span class="state-dot" aria-hidden="true"></span>
                <div>
                  <div class="state-name">读取中</div>
                  <div class="state-detail">等待状态接口返回</div>
                </div>
              </div>
            </div>
            <dl id="llama-facts" class="fact-grid"></dl>
            <div class="model-switch">
              <label for="llama-model-input">目标模型
                <input id="llama-model-input" type="text" list="llama-model-presets" autocomplete="off">
              </label>
              <datalist id="llama-model-presets"></datalist>
              <label for="llama-model-file-input">本地 GGUF 文件
                <input id="llama-model-file-input" type="text" autocomplete="off" placeholder="留空时使用 Hugging Face 模型">
              </label>
              <div class="control-row model-actions">
                <button id="llama-reload" class="warning" type="button">重载模型</button>
                <button id="llama-switch" class="primary" type="button">切换模型</button>
              </div>
            </div>
            <div class="control-row">
              <button id="llama-start" class="primary" type="button">启动/检查</button>
              <button id="llama-stop" class="danger" type="button">停止</button>
              <button id="llama-restart" class="warning" type="button">重启</button>
            </div>
            <div id="dashboard-status" class="notice" role="status"></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h2>视觉理解服务</h2>
          </div>
          <div class="panel-body">
            <div id="vision-status" class="status-spine">
              <div class="state-line">
                <span class="state-dot" aria-hidden="true"></span>
                <div>
                  <div class="state-name">读取中</div>
                  <div class="state-detail">等待状态接口返回</div>
                </div>
              </div>
            </div>
            <dl id="vision-facts" class="fact-grid"></dl>
            <div class="control-row">
              <button id="vision-start" class="primary" type="button">启动/检查</button>
              <button id="vision-stop" class="danger" type="button">停止</button>
              <button id="vision-restart" class="warning" type="button">重启</button>
            </div>
            <div id="vision-notice" class="notice" role="status"></div>
          </div>
        </div>
      </div>
      <div class="dashboard-stack">
        <div class="panel">
          <div class="panel-head">
            <h2>趋势</h2>
            <div class="segmented" role="tablist" aria-label="图表视图">
              <button class="active" data-chart-mode="learning" type="button">学习</button>
              <button data-chart-mode="performance" type="button">推理</button>
              <button data-chart-mode="groups" type="button">群组</button>
            </div>
          </div>
          <div id="dashboard-chart" class="chart-wrap" role="img" aria-label="Dashboard chart"></div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h2>llama.cpp 日志</h2>
            <div class="log-tools">
              <button id="logs-pause" type="button">暂停刷新</button>
              <button id="logs-copy" type="button">复制日志</button>
            </div>
          </div>
          <div id="llama-logs" class="terminal" aria-live="polite"></div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h2>视觉服务日志</h2>
            <div class="log-tools">
              <button id="vision-logs-pause" type="button">暂停刷新</button>
              <button id="vision-logs-copy" type="button">复制日志</button>
            </div>
          </div>
          <div id="vision-logs" class="terminal compact" aria-live="polite"></div>
        </div>
      </div>
    </div>
  </section>
  <section id="stickers-view" class="view" aria-labelledby="stickers-title" data-design-system="ant-design">
    <div id="sticker-app" class="sticker-react-root" data-ant-design-component="StickerGalleryApp"></div>
  </section>
  <section id="settings-view" class="view" aria-labelledby="settings-title">
    <div class="settings-shell">
      <nav id="settings-nav" class="settings-nav" aria-label="设置分类"></nav>
      <div>
        <div id="settings-status" class="notice" role="status"></div>
        <div id="settings-sections"></div>
      </div>
    </div>
  </section>
</main>
<script>
const page = location.pathname === "/settings" ? "settings" : (location.pathname === "/stickers" ? "stickers" : "dashboard");
document.body.dataset.page = page;

const categoryOrder = ["后台网页", "模型调用", "模型部署", "视觉理解", "学习与回复", "表情包", "LoRA 训练", "黑名单", "其他"];
const stateLabels = {
  external_api_ready: "外部接口可用",
  external_api_unchecked: "外部接口未确认",
  external_llama_ready: "外部 llama 可用",
  external_llama_unavailable: "外部 llama 不可用",
  server_running: "运行中",
  server_starting: "启动中",
  server_present: "已有兼容服务",
  server_orphaned: "遗留自管进程",
  server_orphaned_unhealthy: "遗留进程异常",
  port_in_use: "端口被占用",
  server_exited: "进程已退出",
  server_stopped: "已停止",
  vision_disabled: "视觉理解关闭",
  vision_running: "视觉服务运行中",
  vision_starting: "视觉服务启动中",
  vision_present: "已有视觉服务",
  vision_exited: "视觉服务已退出",
  vision_stopped: "视觉服务已停止",
};
let fields = [];
let values = {};
let logsPaused = false;
let visionLogsPaused = false;
let modelFieldsDirty = false;
let dashboardStats = null;
let llamaStatus = null;
let visionStatus = null;
let chartMode = "learning";
let stickerState = {
  items: [],
  total: 0,
  limit: 80,
  offset: 0,
  query: "",
  type: "",
  typeCounts: {},
  selected: new Set(),
  previewKey: "",
};
let stickerSearchTimer = null;

function withToken(path) {
  const token = document.getElementById("token").value.trim();
  if (!token) return path;
  return `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

function setNotice(id, text, kind = "") {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = text;
  node.className = `notice ${kind}`;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(withToken(path), options);
  if (response.status === 401) throw new Error("需要 Admin Token");
  if (!response.ok) throw new Error(await response.text());
  return await response.json();
}

function stateKind(state) {
  if (["server_running", "server_present", "external_api_ready", "external_llama_ready", "vision_running", "vision_present"].includes(state)) return "ok";
  if (["server_starting", "external_api_unchecked", "external_llama_unavailable", "port_in_use", "server_orphaned", "server_orphaned_unhealthy", "vision_starting", "vision_stopped", "vision_disabled"].includes(state)) return "warn";
  return "error";
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h ? `${h}h ${m}m ${s}s` : `${m}m ${s}s`;
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

function formatTps(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return `${Number(value).toFixed(2)} tok/s`;
}

function formatTimestamp(seconds) {
  if (!seconds) return "暂无记录";
  return new Date(Number(seconds) * 1000).toLocaleString("zh-CN", { hour12: false });
}

async function loadDashboardStats() {
  try {
    dashboardStats = await fetchJson("/api/dashboard/stats");
    renderMetricTiles();
    renderDashboardChart();
    renderTrainingProgress();
    renderTrainingControl();
  } catch (error) {
    document.getElementById("metric-tiles").innerHTML = `<div class="metric-card"><div class="metric-label">Dashboard</div><div class="metric-note">${escapeHtml(String(error))}</div></div>`;
    document.getElementById("dashboard-chart").innerHTML = `<div class="empty">${escapeHtml(String(error))}</div>`;
    renderTrainingProgress();
    renderTrainingControl();
  }
}

function trainingTargetConfigured() {
  const trainingConfig = dashboardStats && dashboardStats.training_config ? dashboardStats.training_config : {};
  return trainingConfig.target_configured !== false;
}

function renderTrainingControl() {
  const button = document.getElementById("training-run");
  const externalButton = document.getElementById("training-run-external");
  if (!button && !externalButton) return;
  const buttons = [button, externalButton].filter(Boolean);
  const operation = dashboardStats && dashboardStats.training_operation ? dashboardStats.training_operation : {};
  if (operation.available === false) {
    for (const item of buttons) {
      item.disabled = true;
      item.title = "trainer unavailable";
    }
    if (button) button.textContent = "训练不可用";
    if (externalButton) externalButton.textContent = "训练不可用";
    return;
  }
  if (operation.running) {
    for (const item of buttons) {
      item.disabled = true;
      item.title = operation.reason ? `reason: ${operation.reason}` : "training is running";
    }
    if (button) button.textContent = "训练中...";
    if (externalButton) externalButton.textContent = "训练中...";
    return;
  }
  if (!trainingTargetConfigured()) {
    for (const item of buttons) {
      item.disabled = true;
      item.title = "先在设置里填写 LoRA 训练接口地址或本地 LoRA 训练命令";
    }
    if (button) button.textContent = "配置训练目标";
    if (externalButton) externalButton.textContent = "配置训练目标";
    return;
  }
  for (const item of buttons) item.disabled = false;
  if (button) button.textContent = "立即训练";
  if (externalButton) externalButton.textContent = "全量外部语料训练";
  const last = operation.last_result;
  const title = last && last.status ? `上次训练：${last.status}` : "";
  if (button) button.title = title;
  if (externalButton) externalButton.title = title ? `${title}；本次将忽略数据集上限并合并全部外部语料` : "忽略数据集上限并合并全部外部语料";
}

function renderTrainingProgress() {
  const facts = document.getElementById("training-progress");
  const log = document.getElementById("training-log-tail");
  if (!facts || !log) return;
  const operation = dashboardStats && dashboardStats.training_operation ? dashboardStats.training_operation : {};
  const progress = operation.progress || {};
  const rows = [
    ["阶段", progress.stage || (operation.running ? "running" : "idle")],
    ["状态", progress.status || "-"],
    ["原因", progress.reason || operation.reason || "-"],
    ["样本", formatNumber(progress.sample_count)],
    ["数据集", progress.dataset_path || "-"],
    ["输出目录", progress.output_dir || "-"],
  ];
  facts.innerHTML = rows.map(([key, value]) => `
    <div class="fact"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>
  `).join("");
  log.textContent = progress.log_tail || (operation.last_result && operation.last_result.log) || "暂无训练日志";
}

function renderMetricTiles() {
  const root = document.getElementById("metric-tiles");
  if (!root) return;
  const totals = dashboardStats && dashboardStats.totals ? dashboardStats.totals : {};
  const inference = dashboardStats && dashboardStats.inference ? dashboardStats.inference : {};
  const training = dashboardStats && dashboardStats.training ? dashboardStats.training : {};
  const trainingConfig = dashboardStats && dashboardStats.training_config ? dashboardStats.training_config : {};
  const lastTps = inference.last_tps;
  const statusLabel = llamaStatus ? (stateLabels[llamaStatus.state] || llamaStatus.state || "-") : "读取中";
  const visionLabel = visionStatus ? (stateLabels[visionStatus.state] || visionStatus.state || "-") : "读取中";
  const lastRun = training.last_run;
  const lastSampleAt = totals.last_sample_at ? formatTimestamp(totals.last_sample_at) : "暂无样本";
  const pendingNote = !trainingTargetConfigured()
    ? `未配置 LoRA 训练目标；这些样本尚未被真正训练`
    : training.last_success
      ? `上次成功训练后新增；最低 ${formatNumber(trainingConfig.min_train_samples)} 条`
      : `尚无成功训练，显示全部可训练样本；最低 ${formatNumber(trainingConfig.min_train_samples)} 条`;
  const tiles = [
    {
      label: "待训练样本",
      value: formatNumber(totals.pending_training_samples),
      note: pendingNote,
    },
    {
      label: "已入库消息",
      value: formatNumber(totals.learned_messages),
      note: `${formatNumber(totals.trainable_samples)} 条可训练样本`,
    },
    {
      label: "今日新增",
      value: formatNumber(totals.today_samples),
      note: `近 24h ${formatNumber(totals.last_24h_samples)} 条；最近 ${lastSampleAt}`,
    },
    {
      label: "覆盖范围",
      value: `${formatNumber(totals.groups)} 群`,
      note: `${formatNumber(totals.users)} 位用户贡献样本`,
    },
    {
      label: "表情记忆",
      value: formatNumber(totals.stickers),
      note: `${formatNumber(totals.samples_with_stickers)} 条样本含表情`,
    },
    {
      label: "训练记录",
      value: formatNumber(totals.training_runs),
      note: lastRun ? `${lastRun.status} · ${formatTimestamp(lastRun.finished_at || lastRun.started_at)}` : (trainingConfig.target_configured ? "暂无训练记录" : "未配置 LoRA 训练目标"),
    },
    {
      label: "llama 状态",
      value: statusLabel,
      note: llamaStatus ? `${llamaStatus.runtime || "-"} · ${llamaStatus.pid ? `PID ${llamaStatus.pid}` : "无 PID"}` : "等待状态接口",
    },
    {
      label: "视觉服务",
      value: visionLabel,
      note: visionStatus ? `${visionStatus.model || "-"} · ${visionStatus.pid ? `PID ${visionStatus.pid}` : "无 PID"}` : "等待状态接口",
    },
  ];
  if (lastTps && lastTps.tps) {
    tiles.splice(2, 0, {
      label: "上次推理 TPS",
      value: formatTps(lastTps.tps),
      note: `${lastTps.source || "unknown"} · ${lastTps.model || "-"}`,
    });
  }
  root.innerHTML = tiles.map(tile => `
    <article class="metric-card">
      <div class="metric-label">${escapeHtml(tile.label)}</div>
      <div class="metric-value">${escapeHtml(tile.value)}</div>
      <div class="metric-note">${escapeHtml(tile.note)}</div>
    </article>
  `).join("");
}

function renderDashboardChart() {
  const root = document.getElementById("dashboard-chart");
  if (!root) return;
  const charts = dashboardStats && dashboardStats.charts ? dashboardStats.charts : {};
  const map = {
    learning: {
      title: "学习样本趋势",
      unit: "条",
      type: "bar",
      data: (charts.learning_trend || []).map(item => ({ label: item.label, value: Number(item.count || 0) })),
    },
    performance: {
      title: "推理性能趋势",
      unit: "tok/s",
      type: "line",
      data: (charts.performance_trend || []).map((item, index) => ({
        label: item.time ? item.time.slice(11, 19) : String(index + 1),
        value: Number(item.tps || 0),
      })).filter(item => item.value > 0),
    },
    groups: {
      title: "群组贡献分布",
      unit: "条",
      type: "bar",
      data: (charts.group_contribution || []).map(item => ({ label: item.label, value: Number(item.count || 0) })),
    },
  };
  const config = map[chartMode] || map.learning;
  root.setAttribute("aria-label", config.title);
  if (!config.data.length) {
    root.innerHTML = `<div class="empty">${escapeHtml(config.title)}暂无数据</div>`;
    return;
  }
  root.innerHTML = config.type === "line" ? lineChart(config.data, config.unit) : barChart(config.data, config.unit);
}

function barChart(data, unit) {
  const width = 720;
  const height = 260;
  const pad = { top: 18, right: 18, bottom: 34, left: 36 };
  const max = Math.max(1, ...data.map(item => item.value));
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const gap = 8;
  const barW = Math.max(8, (innerW - gap * Math.max(0, data.length - 1)) / data.length);
  const bars = data.map((item, index) => {
    const x = pad.left + index * (barW + gap);
    const h = Math.max(2, item.value / max * innerH);
    const y = pad.top + innerH - h;
    const label = data.length > 10 && index % 2 ? "" : item.label;
    return `
      <rect class="chart-bar" x="${x}" y="${y}" width="${barW}" height="${h}" rx="4"><title>${escapeHtml(item.label)}: ${item.value} ${unit}</title></rect>
      <text class="chart-label" x="${x + barW / 2}" y="${height - 10}" text-anchor="middle">${escapeHtml(label)}</text>
    `;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img">
    <line class="chart-axis" x1="${pad.left}" y1="${pad.top + innerH}" x2="${width - pad.right}" y2="${pad.top + innerH}"></line>
    <text class="chart-label" x="${pad.left}" y="12">${max} ${escapeHtml(unit)}</text>
    ${bars}
  </svg>`;
}

function lineChart(data, unit) {
  const width = 720;
  const height = 260;
  const pad = { top: 18, right: 18, bottom: 34, left: 36 };
  const max = Math.max(1, ...data.map(item => item.value));
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const points = data.map((item, index) => {
    const x = pad.left + (data.length === 1 ? innerW : index / (data.length - 1) * innerW);
    const y = pad.top + innerH - item.value / max * innerH;
    return { ...item, x, y };
  });
  const path = points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
  const dots = points.map((point, index) => {
    const label = data.length > 8 && index % 2 ? "" : point.label;
    return `
      <circle class="chart-point" cx="${point.x}" cy="${point.y}" r="4"><title>${escapeHtml(point.label)}: ${point.value} ${unit}</title></circle>
      <text class="chart-label" x="${point.x}" y="${height - 10}" text-anchor="middle">${escapeHtml(label)}</text>
    `;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img">
    <line class="chart-axis" x1="${pad.left}" y1="${pad.top + innerH}" x2="${width - pad.right}" y2="${pad.top + innerH}"></line>
    <text class="chart-label" x="${pad.left}" y="12">${max.toFixed(2)} ${escapeHtml(unit)}</text>
    <path class="chart-line" d="${path}"></path>
    ${dots}
  </svg>`;
}

function renderFacts(payload) {
  const facts = [
    ["运行方式", payload.runtime],
    ["健康检查", payload.healthy ? "healthy" : "not healthy"],
    ["API", payload.api_base],
    ["监听", `${payload.host}:${payload.port}`],
    ["PID", payload.pid ?? "-"],
    ["运行时长", formatDuration(payload.uptime_seconds)],
    ["模型", payload.model || "-"],
    ["模型文件", payload.model_file || "-"],
    ["程序", payload.executable || "-"],
    ["上次结果", payload.last_result ? payload.last_result.status : "-"],
  ];
  document.getElementById("llama-facts").innerHTML = facts.map(([key, value]) =>
    `<div class="fact"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`
  ).join("");
}

function renderStatus(payload) {
  llamaStatus = payload;
  const state = payload.state || "unknown";
  const root = document.getElementById("llama-status");
  root.className = `status-spine ${stateKind(state)}`;
  const label = stateLabels[state] || state;
  const detail = payload.control && payload.control.reason ? payload.control.reason : payload.command || "状态已更新";
  root.innerHTML = `
    <div class="state-line">
      <span class="state-dot" aria-hidden="true"></span>
      <div>
        <div class="state-name">${escapeHtml(label)}</div>
        <div class="state-detail">${escapeHtml(detail)}</div>
      </div>
    </div>`;
  renderFacts(payload);
  renderModelControls(payload);
  setControl("llama-start", Boolean(payload.control && payload.control.can_start), payload.control && payload.control.reason);
  setControl("llama-stop", Boolean(payload.control && payload.control.can_stop), payload.control && payload.control.reason);
  setControl("llama-restart", Boolean(payload.control && payload.control.can_restart), payload.control && payload.control.reason);
  setControl("llama-reload", Boolean(payload.control && payload.control.can_reload), payload.control && payload.control.reason);
  setControl("llama-switch", Boolean(payload.control && payload.control.can_switch), payload.control && payload.control.reason);
  renderMetricTiles();
}

function renderVisionFacts(payload) {
  const facts = [
    ["健康检查", payload.healthy ? "healthy" : "not healthy"],
    ["端点", payload.endpoint || "-"],
    ["PID", payload.pid ?? "-"],
    ["运行时长", formatDuration(payload.uptime_seconds)],
    ["模型", payload.model || "-"],
    ["启动者", payload.started_by_plugin ? "plugin" : "external"],
    ["上次结果", payload.last_result ? payload.last_result.status : "-"],
    ["命令", payload.command || "-"],
  ];
  document.getElementById("vision-facts").innerHTML = facts.map(([key, value]) =>
    `<div class="fact"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`
  ).join("");
}

function renderVisionStatus(payload) {
  visionStatus = payload;
  const state = payload.state || "unknown";
  const root = document.getElementById("vision-status");
  root.className = `status-spine ${stateKind(state)}`;
  const label = stateLabels[state] || state;
  const detail = payload.control && payload.control.reason ? payload.control.reason : payload.command || payload.endpoint || "状态已更新";
  root.innerHTML = `
    <div class="state-line">
      <span class="state-dot" aria-hidden="true"></span>
      <div>
        <div class="state-name">${escapeHtml(label)}</div>
        <div class="state-detail">${escapeHtml(detail)}</div>
      </div>
    </div>`;
  renderVisionFacts(payload);
  setControl("vision-start", Boolean(payload.control && payload.control.can_start), payload.control && payload.control.reason);
  setControl("vision-stop", Boolean(payload.control && payload.control.can_stop), payload.control && payload.control.reason);
  setControl("vision-restart", Boolean(payload.control && payload.control.can_restart), payload.control && payload.control.reason);
  renderMetricTiles();
}

function renderModelControls(payload) {
  const modelInput = document.getElementById("llama-model-input");
  const fileInput = document.getElementById("llama-model-file-input");
  if (!modelFieldsDirty) {
    modelInput.value = payload.model || "";
    fileInput.value = payload.model_file || "";
  }
  const list = document.getElementById("llama-model-presets");
  list.innerHTML = "";
  for (const option of payload.model_presets || []) {
    const node = document.createElement("option");
    node.value = option.value;
    node.label = option.label || option.value;
    list.appendChild(node);
  }
}

function setControl(id, enabled, reason) {
  const node = document.getElementById(id);
  node.disabled = !enabled;
  node.title = enabled ? "" : (reason || "当前状态不可用");
}

async function loadLlamaStatus() {
  try {
    const payload = await fetchJson("/api/llama/status");
    renderStatus(payload);
    setNotice("dashboard-status", "状态已刷新", "ok");
  } catch (error) {
    setNotice("dashboard-status", String(error), "error");
  }
}

async function loadLlamaLogs() {
  if (logsPaused) return;
  try {
    const payload = await fetchJson("/api/llama/logs?tail=300");
    renderLogs(payload.logs || []);
  } catch (error) {
    renderLogs([{ time: "", stream: "error", message: String(error) }]);
  }
}

function renderLogs(logs) {
  renderLogPanel("llama-logs", logs);
}

function renderLogPanel(id, logs) {
  const root = document.getElementById(id);
  if (!logs.length) {
    root.innerHTML = `<div class="empty">暂无日志</div>`;
    return;
  }
  root.innerHTML = logs.map(entry => `
    <div class="log-line">
      <span class="time">${escapeHtml(entry.time || "")}</span>
      <span class="stream">${escapeHtml(entry.stream || "")}</span>
      <span class="message">${escapeHtml(entry.message || "")}</span>
    </div>`).join("");
  root.scrollTop = root.scrollHeight;
}

function logText(id) {
  const root = document.getElementById(id);
  if (!root) return "";
  return Array.from(root.querySelectorAll(".log-line")).map(node => node.textContent.trim()).join("\n");
}

async function loadVisionStatus() {
  try {
    const payload = await fetchJson("/api/vision/status");
    renderVisionStatus(payload);
    setNotice("vision-notice", "状态已刷新", "ok");
  } catch (error) {
    setNotice("vision-notice", String(error), "error");
  }
}

async function loadVisionLogs() {
  if (visionLogsPaused) return;
  try {
    const payload = await fetchJson("/api/vision/logs?tail=200");
    renderLogPanel("vision-logs", payload.logs || []);
  } catch (error) {
    renderLogPanel("vision-logs", [{ time: "", stream: "error", message: String(error) }]);
  }
}

async function controlLlama(action, body = null) {
  const labels = { stop: "停止", restart: "重启", reload: "重载模型", switch: "切换模型" };
  if (action !== "start" && !confirm(`确认${labels[action] || action} llama.cpp 服务？`)) return;
  try {
    setNotice("dashboard-status", "操作已提交...");
    const options = { method: "POST" };
    if (body) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    const payload = await fetchJson(`/api/llama/${action}`, options);
    setNotice("dashboard-status", `操作结果：${payload.status}`, "ok");
    if (action === "switch") modelFieldsDirty = false;
    await loadLlamaStatus();
    await loadLlamaLogs();
  } catch (error) {
    setNotice("dashboard-status", String(error), "error");
  }
}

async function controlVision(action) {
  const labels = { stop: "停止", restart: "重启" };
  if (action !== "start" && !confirm(`确认${labels[action] || action}视觉理解服务？`)) return;
  try {
    setNotice("vision-notice", "操作已提交...");
    const payload = await fetchJson(`/api/vision/${action}`, { method: "POST" });
    setNotice("vision-notice", `操作结果：${payload.status}`, "ok");
    await loadVisionStatus();
    await loadVisionLogs();
  } catch (error) {
    setNotice("vision-notice", String(error), "error");
  }
}

async function triggerTraining(options = {}) {
  if (!trainingTargetConfigured()) {
    setNotice("dashboard-status", "未配置 LoRA 训练目标：请先在设置中填写训练接口地址或本地训练命令", "error");
    renderTrainingControl();
    return;
  }
  const fullExternalCorpus = Boolean(options.fullExternalCorpus);
  try {
    const button = document.getElementById(fullExternalCorpus ? "training-run-external" : "training-run");
    button.disabled = true;
    button.textContent = "提交中...";
    setNotice("dashboard-status", fullExternalCorpus ? "全量外部语料训练任务提交中..." : "训练任务提交中...");
    const payload = await fetchJson("/api/training/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reason: fullExternalCorpus ? "admin_web_full_external_corpus" : "admin_web",
        include_external_corpus: fullExternalCorpus ? true : undefined,
        full_external_corpus: fullExternalCorpus,
      }),
    });
    setNotice("dashboard-status", `训练任务：${payload.status}`, "ok");
    await loadDashboardStats();
  } catch (error) {
    setNotice("dashboard-status", String(error), "error");
    renderTrainingControl();
  }
}

function switchModel() {
  const model = document.getElementById("llama-model-input").value.trim();
  const modelFile = document.getElementById("llama-model-file-input").value.trim();
  if (!model && !modelFile) {
    setNotice("dashboard-status", "请输入模型 ID 或本地 GGUF 文件", "error");
    return;
  }
  controlLlama("switch", { model, model_file: modelFile });
}

async function loadStickers(resetOffset = false) {
  if (resetOffset) stickerState.offset = 0;
  const search = document.getElementById("sticker-search");
  const type = document.getElementById("sticker-type");
  const limit = document.getElementById("sticker-limit");
  stickerState.query = search ? search.value.trim() : stickerState.query;
  stickerState.type = type ? type.value : stickerState.type;
  stickerState.limit = limit ? parseInt(limit.value || "80", 10) : stickerState.limit;
  const params = new URLSearchParams({
    limit: String(stickerState.limit),
    offset: String(stickerState.offset),
  });
  if (stickerState.query) params.set("q", stickerState.query);
  if (stickerState.type) params.set("type", stickerState.type);
  try {
    setNotice("sticker-status", "读取表情库...");
    const payload = await fetchJson(`/api/stickers?${params.toString()}`);
    stickerState.items = payload.items || [];
    stickerState.total = Number(payload.total || 0);
    stickerState.limit = Number(payload.limit || stickerState.limit);
    stickerState.offset = Number(payload.offset || 0);
    stickerState.query = payload.query || stickerState.query;
    stickerState.type = payload.type || stickerState.type;
    stickerState.typeCounts = payload.type_counts || {};
    renderStickerTypes();
    renderStickerTable();
    renderStickerPager();
    updateStickerSelectionUi();
    setNotice("sticker-status", `已加载 ${formatNumber(stickerState.items.length)} 个，仓库共 ${formatNumber(stickerState.total)} 个`, "ok");
  } catch (error) {
    document.getElementById("sticker-grid").innerHTML = `<div class="empty">${escapeHtml(String(error))}</div>`;
    setNotice("sticker-status", String(error), "error");
  }
}

function renderStickerTypes() {
  const entries = Object.entries(stickerState.typeCounts).sort((a, b) => Number(b[1]) - Number(a[1]) || a[0].localeCompare(b[0]));
  const types = entries.map(([type]) => type);
  if (stickerState.type && !types.includes(stickerState.type)) types.push(stickerState.type);
  const select = document.getElementById("sticker-type");
  if (select) {
    select.innerHTML = `<option value="">全部类型</option>` + types.map(type =>
      `<option value="${escapeHtml(type)}">${escapeHtml(type)} (${formatNumber(stickerState.typeCounts[type] || 0)})</option>`
    ).join("");
    select.value = stickerState.type;
  }
  const total = Object.values(stickerState.typeCounts).reduce((sum, count) => sum + Number(count || 0), 0);
  const strip = document.getElementById("sticker-type-strip");
  if (!strip) return;
  const buttons = [
    `<button class="type-pill ${stickerState.type ? "" : "active"}" type="button" data-sticker-type="">全部 ${formatNumber(total)}</button>`,
    ...entries.map(([type, count]) =>
      `<button class="type-pill ${stickerState.type === type ? "active" : ""}" type="button" data-sticker-type="${escapeHtml(type)}">${escapeHtml(type)} ${formatNumber(count)}</button>`
    ),
  ];
  strip.innerHTML = buttons.join("");
  strip.querySelectorAll("[data-sticker-type]").forEach(button => {
    button.addEventListener("click", () => {
      stickerState.type = button.dataset.stickerType || "";
      const typeSelect = document.getElementById("sticker-type");
      if (typeSelect) typeSelect.value = stickerState.type;
      loadStickers(true);
    });
  });
}

function renderStickerTable() {
  const root = document.getElementById("sticker-grid");
  if (!root) return;
  if (!stickerState.items.length) {
    root.innerHTML = `<div class="sticker-table-viewport"><div class="empty">没有匹配的表情记录</div></div>`;
    return;
  }
  const rows = stickerState.items.map(item => {
    const selected = stickerState.selected.has(item.key);
    const title = stickerTitle(item);
    const key = String(item.key || "");
    return `
      <div class="sticker-table-row ${selected ? "selected" : ""}" role="row">
        <label class="sticker-table-cell sticker-check" role="cell" title="选择 ${escapeHtml(title)}">
          <input type="checkbox" data-sticker-key="${escapeHtml(item.key)}" ${selected ? "checked" : ""}>
        </label>
        <div class="sticker-table-cell sticker-summary-cell" role="cell">
          <button class="sticker-thumb" type="button" data-preview-key="${escapeHtml(item.key)}" aria-label="预览 ${escapeHtml(title)}">
            ${stickerThumbHtml(item)}
          </button>
          <div>
            <div class="sticker-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
            <div class="sticker-subtitle" title="${escapeHtml(key)}">${escapeHtml(key)}</div>
          </div>
        </div>
        <div class="sticker-table-cell" role="cell">${stickerTypeTagHtml(item)}</div>
        <div class="sticker-table-cell" role="cell">${formatNumber(item.count)}</div>
        <div class="sticker-table-cell sticker-subtitle" role="cell" title="${escapeHtml(formatTimestamp(item.last_seen))}">${escapeHtml(formatTimestamp(item.last_seen))}</div>
        <div class="sticker-table-cell" role="cell">
          <button class="ant-btn" type="button" data-preview-key="${escapeHtml(item.key)}">编辑</button>
        </div>
      </div>
    `;
  }).join("");
  root.innerHTML = `
    <div class="sticker-table-viewport" role="table" aria-label="表情记录">
      <div class="sticker-table-head" role="row">
        <div class="sticker-table-cell" role="columnheader">选择</div>
        <div class="sticker-table-cell" role="columnheader">表情 / 提示词</div>
        <div class="sticker-table-cell" role="columnheader">类型</div>
        <div class="sticker-table-cell" role="columnheader">次数</div>
        <div class="sticker-table-cell" role="columnheader">最近使用</div>
        <div class="sticker-table-cell" role="columnheader">操作</div>
      </div>
      ${rows}
    </div>
  `;
  root.querySelectorAll("[data-sticker-key]").forEach(input => {
    input.addEventListener("change", () => {
      if (input.checked) stickerState.selected.add(input.dataset.stickerKey);
      else stickerState.selected.delete(input.dataset.stickerKey);
      renderStickerTable();
      updateStickerSelectionUi();
    });
  });
  root.querySelectorAll("[data-preview-key]").forEach(button => {
    button.addEventListener("click", () => {
      const item = stickerState.items.find(candidate => candidate.key === button.dataset.previewKey);
      if (item) openStickerPreview(item);
    });
  });
}

function renderStickerPager() {
  const info = document.getElementById("sticker-page-info");
  const prev = document.getElementById("sticker-prev");
  const next = document.getElementById("sticker-next");
  const from = stickerState.total ? stickerState.offset + 1 : 0;
  const to = Math.min(stickerState.offset + stickerState.limit, stickerState.total);
  if (info) info.textContent = `${formatNumber(from)}-${formatNumber(to)} / ${formatNumber(stickerState.total)}`;
  if (prev) prev.disabled = stickerState.offset <= 0;
  if (next) next.disabled = stickerState.offset + stickerState.limit >= stickerState.total;
}

function updateStickerSelectionUi() {
  const count = stickerState.selected.size;
  const label = document.getElementById("sticker-selection-count");
  const deleteButton = document.getElementById("sticker-delete");
  if (label) label.textContent = `已选择 ${formatNumber(count)}`;
  if (deleteButton) deleteButton.disabled = count === 0;
}

function stickerTitle(item) {
  const data = item.data || {};
  return String(item.summary || data.summary || data.name || data.text || data.id || item.key || "表情");
}

function stickerSource(item) {
  const data = item.data || {};
  const src = String(data.url || data.preview || data.thumb || item.uri || "");
  if (/^(https?:\/\/|data:image\/|\/)/i.test(src)) return src;
  return "";
}

function stickerThumbHtml(item) {
  const src = stickerSource(item);
  const title = stickerTitle(item);
  if (src) {
    return `<img src="${escapeHtml(src)}" alt="${escapeHtml(title)}" loading="lazy" referrerpolicy="no-referrer">`;
  }
  const data = item.data || {};
  const id = data.id || data.emoji_id || data.file || item.type || "-";
  return `<div class="sticker-fallback"><strong>${escapeHtml(String(id).slice(0, 10))}</strong><span>${escapeHtml(item.type || "unknown")}</span></div>`;
}

function stickerTypeTagHtml(item) {
  return `<span class="ant-tag">${escapeHtml(item.type || "unknown")}</span>`;
}

function openStickerPreview(item) {
  const preview = document.getElementById("sticker-preview");
  const media = document.getElementById("sticker-preview-media");
  const meta = document.getElementById("sticker-preview-meta");
  const editor = document.getElementById("sticker-summary-editor");
  if (!preview || !media || !meta) return;
  stickerState.previewKey = item.key || "";
  document.getElementById("sticker-preview-title").textContent = stickerTitle(item);
  if (editor) editor.value = String(item.summary || "");
  setNotice("sticker-preview-status", "");
  media.innerHTML = stickerThumbHtml(item);
  const rows = [
    ["key", item.key],
    ["type", item.type],
    ["summary", item.summary || ""],
    ["count", item.count],
    ["first_seen", formatTimestamp(item.first_seen)],
    ["last_seen", formatTimestamp(item.last_seen)],
    ["group_id", item.group_id || ""],
    ["user_id", item.user_id || ""],
    ["uri", item.uri || ""],
    ["data", JSON.stringify(item.data || {}, null, 2)],
  ];
  meta.innerHTML = rows.map(([key, value]) =>
    `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value ?? ""))}</dd>`
  ).join("");
  preview.classList.remove("hidden");
}

function closeStickerPreview() {
  const preview = document.getElementById("sticker-preview");
  if (preview) preview.classList.add("hidden");
  stickerState.previewKey = "";
  setNotice("sticker-preview-status", "");
}

async function saveStickerSummary() {
  const key = stickerState.previewKey;
  const editor = document.getElementById("sticker-summary-editor");
  const saveButton = document.getElementById("sticker-summary-save");
  if (!key || !editor) return;
  try {
    if (saveButton) saveButton.disabled = true;
    setNotice("sticker-preview-status", "保存中...");
    const result = await fetchJson("/api/stickers/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, summary: editor.value }),
    });
    if (!result.updated) throw new Error("表情不存在或未更新");
    const updated = result.item || { key, summary: editor.value.trim() };
    const index = stickerState.items.findIndex(item => item.key === key);
    if (index >= 0) {
      stickerState.items[index] = { ...stickerState.items[index], ...updated };
      renderStickerTable();
      openStickerPreview(stickerState.items[index]);
    }
    setNotice("sticker-preview-status", "提示词已保存", "ok");
    setNotice("sticker-status", "提示词已保存", "ok");
  } catch (error) {
    setNotice("sticker-preview-status", String(error), "error");
  } finally {
    if (saveButton) saveButton.disabled = false;
  }
}

async function deleteSelectedStickers() {
  const keys = Array.from(stickerState.selected);
  if (!keys.length) return;
  if (!confirm(`确认从可用表情仓库删除 ${keys.length} 个表情？历史聊天样本不会被修改。`)) return;
  try {
    setNotice("sticker-status", "删除中...");
    const result = await fetchJson("/api/stickers/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keys }),
    });
    stickerState.selected.clear();
    setNotice("sticker-status", `已删除 ${formatNumber(result.deleted)} / ${formatNumber(result.requested)} 个`, "ok");
    await loadStickers(false);
    if (!stickerState.items.length && stickerState.offset > 0) {
      stickerState.offset = Math.max(0, stickerState.offset - stickerState.limit);
      await loadStickers(false);
    }
  } catch (error) {
    setNotice("sticker-status", String(error), "error");
  }
}

let stickerGalleryReload = null;

function mountStickerGalleryApp() {
  const root = document.getElementById("sticker-app");
  if (!root) return false;
  if (!window.React || !window.ReactDOM || !window.antd) {
    root.innerHTML = `<div class="panel"><div class="panel-body"><div class="notice error">Ant Design 资源加载失败，请检查 React / ReactDOM / antd 本地资源是否可访问。</div></div></div>`;
    return true;
  }

  const React = window.React;
  const ReactDOM = window.ReactDOM;
  const AntD = window.antd;
  const h = React.createElement;
  const Table = AntD.Table;
  const Modal = AntD.Modal;
  const Search = AntD.Input.Search;
  const TextArea = AntD.Input.TextArea;
  const Select = AntD.Select;
  const Button = AntD.Button;
  const Card = AntD.Card;
  const Space = AntD.Space;
  const Tag = AntD.Tag;
  const Typography = AntD.Typography;
  const Pagination = AntD.Pagination;
  const Empty = AntD.Empty;
  const Image = AntD.Image;
  const Descriptions = AntD.Descriptions;
  const Alert = AntD.Alert;
  const messageApi = AntD.message;

  function thumbNode(item, onOpen) {
    const src = stickerSource(item);
    const title = stickerTitle(item);
    const content = src
      ? h(Image, {
          src,
          alt: title,
          preview: false,
          fallback: "",
          referrerPolicy: "no-referrer",
        })
      : h("div", { className: "sticker-fallback" },
          h("strong", null, String((item.data || {}).id || (item.data || {}).emoji_id || (item.data || {}).file || item.type || "-").slice(0, 10)),
          h("span", null, item.type || "unknown")
        );
    return h("button", {
      className: "sticker-thumb",
      type: "button",
      onClick: onOpen,
      "aria-label": `预览 ${title}`,
    }, content);
  }

  function StickerGalleryApp() {
    const [items, setItems] = React.useState([]);
    const [total, setTotal] = React.useState(0);
    const [limit, setLimit] = React.useState(80);
    const [offset, setOffset] = React.useState(0);
    const [query, setQuery] = React.useState("");
    const [type, setType] = React.useState("");
    const [typeCounts, setTypeCounts] = React.useState({});
    const [selectedKeys, setSelectedKeys] = React.useState([]);
    const [loading, setLoading] = React.useState(false);
    const [notice, setNoticeState] = React.useState({ type: "info", text: "" });
    const [preview, setPreview] = React.useState(null);
    const [summaryText, setSummaryText] = React.useState("");
    const [saving, setSaving] = React.useState(false);

    const load = React.useCallback(async () => {
      const params = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
      });
      if (query.trim()) params.set("q", query.trim());
      if (type) params.set("type", type);
      setLoading(true);
      setNoticeState({ type: "info", text: "读取表情库..." });
      try {
        const payload = await fetchJson(`/api/stickers?${params.toString()}`);
        setItems(payload.items || []);
        setTotal(Number(payload.total || 0));
        setTypeCounts(payload.type_counts || {});
        setNoticeState({
          type: "success",
          text: `已加载 ${formatNumber((payload.items || []).length)} 个，仓库共 ${formatNumber(payload.total)} 个`,
        });
      } catch (error) {
        setItems([]);
        setNoticeState({ type: "error", text: String(error) });
      } finally {
        setLoading(false);
      }
    }, [limit, offset, query, type]);

    React.useEffect(() => {
      stickerGalleryReload = load;
      load();
      return () => {
        if (stickerGalleryReload === load) stickerGalleryReload = null;
      };
    }, [load]);

    const typeEntries = React.useMemo(
      () => Object.entries(typeCounts).sort((a, b) => Number(b[1]) - Number(a[1]) || String(a[0]).localeCompare(String(b[0]))),
      [typeCounts]
    );
    const typeOptions = React.useMemo(
      () => [
        { value: "", label: "全部类型" },
        ...typeEntries.map(([value, count]) => ({ value, label: `${value} (${formatNumber(count)})` })),
      ],
      [typeEntries]
    );
    const totalByType = React.useMemo(
      () => Object.values(typeCounts).reduce((sum, count) => sum + Number(count || 0), 0),
      [typeCounts]
    );

    function openPreview(item) {
      setPreview(item);
      setSummaryText(String(item.summary || ""));
    }

    async function saveSummary() {
      if (!preview) return;
      setSaving(true);
      try {
        const result = await fetchJson("/api/stickers/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key: preview.key, summary: summaryText }),
        });
        if (!result.updated) throw new Error("表情不存在或未更新");
        const updated = result.item || { ...preview, summary: summaryText.trim() };
        setItems(current => current.map(item => item.key === preview.key ? { ...item, ...updated } : item));
        setPreview(current => current ? { ...current, ...updated } : current);
        setNoticeState({ type: "success", text: "提示词已保存" });
        messageApi.success("提示词已保存");
      } catch (error) {
        setNoticeState({ type: "error", text: String(error) });
        messageApi.error(String(error));
      } finally {
        setSaving(false);
      }
    }

    function deleteSelected() {
      if (!selectedKeys.length) return;
      Modal.confirm({
        title: "删除所选表情",
        content: `确认从可用表情仓库删除 ${selectedKeys.length} 个表情？历史聊天样本不会被修改。`,
        okText: "删除",
        okType: "danger",
        cancelText: "取消",
        async onOk() {
          const result = await fetchJson("/api/stickers/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ keys: selectedKeys }),
          });
          setSelectedKeys([]);
          setNoticeState({
            type: "success",
            text: `已删除 ${formatNumber(result.deleted)} / ${formatNumber(result.requested)} 个`,
          });
          messageApi.success(`已删除 ${formatNumber(result.deleted)} 个表情`);
          await load();
        },
      });
    }

    const columns = [
      {
        title: "表情 / 提示词",
        dataIndex: "summary",
        key: "summary",
        width: 360,
        render: (_value, item) => h("div", { className: "sticker-table-media" },
          thumbNode(item, () => openPreview(item)),
          h("div", { style: { minWidth: 0 } },
            h("div", { className: "sticker-title", title: stickerTitle(item) }, stickerTitle(item)),
            h("div", { className: "sticker-subtitle", title: String(item.key || "") }, item.key || "")
          )
        ),
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 110,
        render: value => h(Tag, { color: "blue" }, value || "unknown"),
      },
      {
        title: "次数",
        dataIndex: "count",
        key: "count",
        width: 90,
        sorter: (a, b) => Number(a.count || 0) - Number(b.count || 0),
        render: value => formatNumber(value),
      },
      {
        title: "最近使用",
        dataIndex: "last_seen",
        key: "last_seen",
        width: 168,
        render: value => formatTimestamp(value),
      },
      {
        title: "操作",
        key: "actions",
        width: 96,
        render: (_value, item) => h(Button, { size: "small", onClick: () => openPreview(item) }, "编辑"),
      },
    ];

    const previewMeta = preview ? [
      { key: "key", label: "key", children: preview.key || "" },
      { key: "type", label: "type", children: preview.type || "" },
      { key: "summary", label: "summary", children: preview.summary || "" },
      { key: "count", label: "count", children: formatNumber(preview.count) },
      { key: "first_seen", label: "first_seen", children: formatTimestamp(preview.first_seen) },
      { key: "last_seen", label: "last_seen", children: formatTimestamp(preview.last_seen) },
      { key: "group_id", label: "group_id", children: preview.group_id || "" },
      { key: "user_id", label: "user_id", children: preview.user_id || "" },
      { key: "uri", label: "uri", children: preview.uri || "" },
      { key: "data", label: "data", children: h("pre", { style: { whiteSpace: "pre-wrap", margin: 0 } }, JSON.stringify(preview.data || {}, null, 2)) },
    ] : [];

    return h(React.Fragment, null,
      h("div", { className: "dashboard-hero" },
        h("div", null,
          h("p", { className: "page-kicker" }, "Sticker Gallery"),
          h("h2", { id: "stickers-title", className: "page-title" }, "表情包仓库"),
          h("p", { className: "page-copy" }, "用 Ant Design 表格管理已学习的表情记录，点开即可修改提示词。")
        ),
        h("div", { className: "hero-actions" },
          h("span", { className: "selection-count" }, `已选择 ${formatNumber(selectedKeys.length)}`),
          h(Button, {
            onClick: () => setSelectedKeys(items.map(item => item.key).filter(Boolean)),
            disabled: !items.length,
          }, "全选本页"),
          h(Button, { onClick: () => setSelectedKeys([]), disabled: !selectedKeys.length }, "取消选择"),
          h(Button, { danger: true, onClick: deleteSelected, disabled: !selectedKeys.length }, "删除所选")
        )
      ),
      h(Card, { className: "sticker-gallery-card" },
        h("div", { className: "sticker-gallery-toolbar" },
          h(Search, {
            allowClear: true,
            placeholder: "搜索 key、摘要、群、用户或协议数据",
            value: query,
            onChange: event => { setQuery(event.target.value); setOffset(0); },
            onSearch: value => { setQuery(value); setOffset(0); },
          }),
          h(Select, {
            ariaLabel: "表情类型",
            value: type,
            options: typeOptions,
            onChange: value => { setType(value || ""); setOffset(0); },
          }),
          h(Select, {
            ariaLabel: "每页数量",
            value: limit,
            options: [
              { value: 48, label: "每页 48" },
              { value: 80, label: "每页 80" },
              { value: 120, label: "每页 120" },
            ],
            onChange: value => { setLimit(Number(value)); setOffset(0); },
          })
        ),
        h("div", { className: "sticker-type-strip", "aria-label": "表情类型筛选" },
          h(Button, {
            size: "small",
            type: type ? "default" : "primary",
            onClick: () => { setType(""); setOffset(0); },
          }, `全部 ${formatNumber(totalByType)}`),
          ...typeEntries.map(([itemType, count]) => h(Button, {
            key: itemType,
            size: "small",
            type: type === itemType ? "primary" : "default",
            onClick: () => { setType(itemType); setOffset(0); },
          }, `${itemType} ${formatNumber(count)}`))
        ),
        notice.text ? h(Alert, { type: notice.type, message: notice.text, showIcon: true }) : null,
        h(Table, {
          className: "sticker-gallery-table",
          size: "small",
          rowKey: "key",
          loading,
          dataSource: items,
          columns,
          pagination: false,
          rowSelection: {
            selectedRowKeys: selectedKeys,
            onChange: keys => setSelectedKeys(keys),
            preserveSelectedRowKeys: true,
          },
          scroll: { x: 880, y: "calc(min(760px, 100dvh - 178px) - 278px)" },
          locale: { emptyText: h(Empty, { description: "没有匹配的表情记录" }) },
        }),
        h("div", { className: "sticker-gallery-pager" },
          h(Pagination, {
            current: Math.floor(offset / limit) + 1,
            pageSize: limit,
            total,
            showSizeChanger: false,
            showTotal: value => `共 ${formatNumber(value)} 个`,
            onChange: page => setOffset((page - 1) * limit),
          })
        )
      ),
      h(Modal, {
        open: Boolean(preview),
        title: preview ? stickerTitle(preview) : "表情预览",
        width: 980,
        onCancel: () => setPreview(null),
        destroyOnClose: false,
        footer: [
          h(Button, { key: "cancel", onClick: () => setPreview(null) }, "关闭"),
          h(Button, {
            key: "save",
            id: "sticker-summary-save",
            type: "primary",
            loading: saving,
            onClick: saveSummary,
          }, "保存提示词"),
        ],
      }, preview ? h("div", { className: "sticker-preview-layout" },
        h("div", { className: "preview-media" }, thumbNode(preview, () => {})),
        h("div", { className: "preview-detail" },
          h(Space, { direction: "vertical", size: 12, style: { width: "100%" } },
            h("label", { htmlFor: "sticker-summary-editor" },
              h(Typography.Text, { strong: true }, "提示词 / summary")
            ),
            h(TextArea, {
              id: "sticker-summary-editor",
              value: summaryText,
              maxLength: 200,
              autoSize: { minRows: 4, maxRows: 8 },
              spellCheck: false,
              onChange: event => setSummaryText(event.target.value),
            }),
            h(Descriptions, {
              column: 1,
              size: "small",
              bordered: true,
              items: previewMeta,
            })
          )
        )
      ) : null)
    );
  }

  const renderer = ReactDOM.createRoot
    ? ReactDOM.createRoot(root)
    : { render: component => ReactDOM.render(component, root) };
  renderer.render(h(StickerGalleryApp));
  return true;
}

function fieldType(field) {
  if (field.kind === "bool") return "bool";
  if (field.kind === "int" || field.kind === "float") return "number";
  if (field.kind === "json") return "json";
  return field.multiline ? "textarea" : "text";
}

function renderSettings() {
  const query = document.getElementById("search").value.trim().toLowerCase();
  const runtime = currentRuntime();
  const sections = new Map();
  for (const field of fields) {
    if (!fieldVisible(field, runtime)) continue;
    const haystack = `${field.name} ${field.title} ${field.description} ${field.env} ${field.category}`.toLowerCase();
    if (query && !haystack.includes(query)) continue;
    if (!sections.has(field.category)) sections.set(field.category, []);
    sections.get(field.category).push(field);
  }

  const nav = document.getElementById("settings-nav");
  nav.innerHTML = "";
  for (const category of categoryOrder) {
    if (!sections.has(category)) continue;
    const a = document.createElement("a");
    a.href = `#${category}`;
    a.textContent = `${category} (${sections.get(category).length})`;
    nav.appendChild(a);
  }

  const root = document.getElementById("settings-sections");
  root.innerHTML = "";
  for (const category of categoryOrder) {
    const items = sections.get(category);
    if (!items || !items.length) continue;
    const section = document.createElement("section");
    section.className = "panel";
    section.id = category;
    section.innerHTML = `<div class="panel-head"><h2>${escapeHtml(category)}</h2></div>`;
    for (const field of items) section.appendChild(renderField(field));
    root.appendChild(section);
  }
}

function currentRuntime() {
  const input = document.querySelector('[data-name="qwen_persona_runtime"]');
  return input ? input.value : (values.qwen_persona_runtime || "llama_cpp");
}

function fieldVisible(field, runtime) {
  const allowed = field.visibleWhen && field.visibleWhen.runtime;
  return !allowed || !allowed.length || allowed.includes(runtime);
}

function renderField(field) {
  const row = document.createElement("div");
  row.className = "field";
  row.dataset.name = field.name;
  const label = document.createElement("label");
  label.className = "field-label";
  label.textContent = field.title || field.name;
  const env = document.createElement("span");
  env.className = "env";
  env.textContent = field.env;
  label.appendChild(env);
  if (field.description) {
    const desc = document.createElement("span");
    desc.className = "desc";
    desc.textContent = field.description;
    label.appendChild(desc);
  }
  row.appendChild(label);

  const editor = document.createElement("div");
  editor.className = "editor";
  const type = fieldType(field);
  const value = values[field.name];
  if (field.widget === "combo" && !field.allowCustom) {
    const select = document.createElement("select");
    select.dataset.name = field.name;
    for (const option of field.options || []) {
      const node = document.createElement("option");
      node.value = option.value;
      node.textContent = option.label || option.value;
      node.selected = option.value === value;
      select.appendChild(node);
    }
    if (field.name === "qwen_persona_runtime") {
      select.addEventListener("change", () => {
        values[field.name] = select.value;
        renderSettings();
      });
    }
    editor.appendChild(select);
  } else if (field.widget === "combo") {
    const input = document.createElement("input");
    const listId = `list-${field.name}`;
    input.type = "text";
    input.value = value ?? "";
    input.setAttribute("list", listId);
    input.dataset.name = field.name;
    const list = document.createElement("datalist");
    list.id = listId;
    for (const option of field.options || []) {
      const node = document.createElement("option");
      node.value = option.value;
      node.label = option.label || option.value;
      node.title = option.description || "";
      list.appendChild(node);
    }
    editor.appendChild(input);
    editor.appendChild(list);
  } else if (type === "bool") {
    const wrap = document.createElement("label");
    wrap.className = "switch";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value);
    input.dataset.name = field.name;
    const state = document.createTextNode(input.checked ? "开启" : "关闭");
    input.addEventListener("change", () => { state.textContent = input.checked ? "开启" : "关闭"; });
    wrap.appendChild(input);
    wrap.appendChild(state);
    editor.appendChild(wrap);
  } else if (type === "number") {
    const input = document.createElement("input");
    input.type = "number";
    input.step = field.kind === "float" ? "0.01" : "1";
    input.value = value ?? "";
    input.dataset.name = field.name;
    editor.appendChild(input);
  } else if (type === "json") {
    const textarea = document.createElement("textarea");
    textarea.spellcheck = false;
    textarea.value = JSON.stringify(value ?? (field.default ?? []), null, 2);
    textarea.dataset.name = field.name;
    editor.appendChild(textarea);
  } else if (type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.spellcheck = false;
    textarea.value = value ?? "";
    textarea.dataset.name = field.name;
    editor.appendChild(textarea);
  } else {
    const input = document.createElement("input");
    input.type = field.secret ? "password" : "text";
    input.value = value ?? "";
    input.dataset.name = field.name;
    editor.appendChild(input);
  }
  row.appendChild(editor);
  return row;
}

function readValues() {
  const output = {};
  for (const field of fields) {
    const input = document.querySelector(
      `input[data-name="${field.name}"], textarea[data-name="${field.name}"], select[data-name="${field.name}"]`
    );
    if (!input) continue;
    const type = fieldType(field);
    if (type === "bool") output[field.name] = input.checked;
    else if (type === "number") output[field.name] = field.kind === "int" ? parseInt(input.value || "0", 10) : parseFloat(input.value || "0");
    else if (type === "json") output[field.name] = input.value.trim() ? JSON.parse(input.value) : [];
    else output[field.name] = input.value;
  }
  return output;
}

async function loadConfig() {
  try {
    setNotice("settings-status", "读取设置中...");
    const payload = await fetchJson("/api/config");
    fields = payload.fields;
    values = payload.values;
    renderSettings();
    setNotice("settings-status", `已加载 ${fields.length} 个设置项，覆盖文件：${payload.override_file}`, "ok");
  } catch (error) {
    setNotice("settings-status", String(error), "error");
  }
}

async function saveConfig() {
  try {
    setNotice("settings-status", "保存中...");
    const payload = { values: readValues() };
    const result = await fetchJson("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    fields = result.fields;
    values = result.values;
    renderSettings();
    const restart = result.restart_required && result.restart_required.length ? `；需重启：${result.restart_required.join(", ")}` : "";
    setNotice("settings-status", `已保存，变更 ${result.changed.length} 项${restart}`, "ok");
  } catch (error) {
    setNotice("settings-status", String(error), "error");
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bindEvent(id, eventName, handler) {
  const node = document.getElementById(id);
  if (node) node.addEventListener(eventName, handler);
}

document.getElementById("refresh").addEventListener("click", () => {
  if (page === "settings") loadConfig();
  else if (page === "stickers") {
    if (stickerGalleryReload) stickerGalleryReload();
    else if (document.getElementById("sticker-grid")) loadStickers(false);
  }
  else {
    loadDashboardStats();
    loadLlamaStatus();
    loadLlamaLogs();
    loadVisionStatus();
    loadVisionLogs();
  }
});
document.getElementById("save").addEventListener("click", saveConfig);
document.getElementById("search").addEventListener("input", renderSettings);
document.getElementById("training-run").addEventListener("click", triggerTraining);
document.getElementById("training-run-external").addEventListener("click", () => triggerTraining({ fullExternalCorpus: true }));
document.getElementById("llama-start").addEventListener("click", () => controlLlama("start"));
document.getElementById("llama-stop").addEventListener("click", () => controlLlama("stop"));
document.getElementById("llama-restart").addEventListener("click", () => controlLlama("restart"));
document.getElementById("llama-reload").addEventListener("click", () => controlLlama("reload"));
document.getElementById("llama-switch").addEventListener("click", switchModel);
document.getElementById("llama-model-input").addEventListener("input", () => { modelFieldsDirty = true; });
document.getElementById("llama-model-file-input").addEventListener("input", () => { modelFieldsDirty = true; });
document.getElementById("vision-start").addEventListener("click", () => controlVision("start"));
document.getElementById("vision-stop").addEventListener("click", () => controlVision("stop"));
document.getElementById("vision-restart").addEventListener("click", () => controlVision("restart"));
document.getElementById("logs-pause").addEventListener("click", () => {
  logsPaused = !logsPaused;
  document.getElementById("logs-pause").textContent = logsPaused ? "继续刷新" : "暂停刷新";
});
document.getElementById("logs-copy").addEventListener("click", async () => {
  const text = logText("llama-logs");
  if (navigator.clipboard) await navigator.clipboard.writeText(text);
  setNotice("dashboard-status", "日志已复制", "ok");
});
document.getElementById("vision-logs-pause").addEventListener("click", () => {
  visionLogsPaused = !visionLogsPaused;
  document.getElementById("vision-logs-pause").textContent = visionLogsPaused ? "继续刷新" : "暂停刷新";
});
document.getElementById("vision-logs-copy").addEventListener("click", async () => {
  const text = logText("vision-logs");
  if (navigator.clipboard) await navigator.clipboard.writeText(text);
  setNotice("vision-notice", "日志已复制", "ok");
});
bindEvent("sticker-search", "input", () => {
  clearTimeout(stickerSearchTimer);
  stickerSearchTimer = setTimeout(() => loadStickers(true), 180);
});
bindEvent("sticker-type", "change", () => loadStickers(true));
bindEvent("sticker-limit", "change", () => loadStickers(true));
bindEvent("sticker-select-all", "click", () => {
  for (const item of stickerState.items) stickerState.selected.add(item.key);
  renderStickerTable();
  updateStickerSelectionUi();
});
bindEvent("sticker-clear-selection", "click", () => {
  stickerState.selected.clear();
  renderStickerTable();
  updateStickerSelectionUi();
});
bindEvent("sticker-delete", "click", deleteSelectedStickers);
bindEvent("sticker-prev", "click", () => {
  stickerState.offset = Math.max(0, stickerState.offset - stickerState.limit);
  loadStickers(false);
});
bindEvent("sticker-next", "click", () => {
  if (stickerState.offset + stickerState.limit < stickerState.total) {
    stickerState.offset += stickerState.limit;
    loadStickers(false);
  }
});
bindEvent("sticker-preview-close", "click", closeStickerPreview);
bindEvent("sticker-summary-save", "click", saveStickerSummary);
bindEvent("sticker-preview", "click", event => {
  if (event.target.id === "sticker-preview") closeStickerPreview();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeStickerPreview();
});
document.querySelectorAll("[data-chart-mode]").forEach(button => {
  button.addEventListener("click", () => {
    chartMode = button.dataset.chartMode || "learning";
    document.querySelectorAll("[data-chart-mode]").forEach(item => item.classList.toggle("active", item === button));
    renderDashboardChart();
  });
});

if (page === "settings") {
  loadConfig();
} else if (page === "stickers") {
  if (!mountStickerGalleryApp() && document.getElementById("sticker-grid")) loadStickers(true);
} else {
  loadDashboardStats();
  loadLlamaStatus();
  loadLlamaLogs();
  loadVisionStatus();
  loadVisionLogs();
  setInterval(loadLlamaStatus, 4000);
  setInterval(loadLlamaLogs, 2500);
  setInterval(loadVisionStatus, 5000);
  setInterval(loadVisionLogs, 3000);
  setInterval(loadDashboardStats, 12000);
}
</script>
</body>
</html>
"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class AdminWebServer:
    def __init__(
        self,
        *,
        config: Any,
        save_callback: Callable[[dict[str, Any]], dict[str, Any]],
        logger: Any,
        model_manager: Any | None = None,
        vision_manager: Any | None = None,
        store: Any | None = None,
        qwen_client: Any | None = None,
        trainer: Any | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.config = config
        self.save_callback = save_callback
        self.logger = logger
        self.model_manager = model_manager
        self.vision_manager = vision_manager
        self.store = store
        self.qwen_client = qwen_client
        self.trainer = trainer
        self.event_loop = event_loop
        self._httpd: ReusableThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._pending_action = ""
        self._pending_action_future: Any | None = None
        self._pending_action_lock = threading.Lock()
        self._pending_training_action = ""
        self._pending_training_reason = ""
        self._pending_training_future: Any | None = None
        self._pending_training_lock = threading.Lock()
        self._last_training_result: dict[str, Any] | None = None

    def set_event_loop(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def set_model_manager(self, model_manager: Any) -> None:
        self.model_manager = model_manager

    def set_vision_manager(self, vision_manager: Any) -> None:
        self.vision_manager = vision_manager

    def set_store(self, store: Any) -> None:
        self.store = store

    def set_qwen_client(self, qwen_client: Any) -> None:
        self.qwen_client = qwen_client

    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def start(self) -> None:
        if not getattr(self.config, "qwen_persona_admin_enabled", True):
            return
        if self._httpd:
            return

        host = str(getattr(self.config, "qwen_persona_admin_host", "127.0.0.1"))
        port = int(getattr(self.config, "qwen_persona_admin_port", 8082))
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner._handle_get(self)

            def do_POST(self) -> None:
                owner._handle_post(self)

            def log_message(self, fmt: str, *args: Any) -> None:
                owner.logger.debug("qwen persona admin: " + fmt % args)

        self._httpd = ReusableThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        if not getattr(self.config, "qwen_persona_admin_token", "") and host not in {"127.0.0.1", "localhost", "::1"}:
            self.logger.warning("qwen persona admin web has no token and is not bound to localhost")
        self.logger.info(f"qwen persona admin web listening on http://{host}:{port}")

    def stop(self) -> None:
        if not self._httpd:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        path = parsed.path
        if path.startswith("/assets/vendor/"):
            try:
                payload = self._vendor_asset_payload(path)
            except FileNotFoundError:
                self._send_text(handler, "asset not found", status=404)
                return
            self._send_bytes(handler, payload["data"], payload["content_type"])
            return
        if path in {"/", "/settings", "/stickers"}:
            self._send_html(handler, ADMIN_HTML)
            return
        if path == "/api/config":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            self._send_json(handler, self._config_payload())
            return
        if path == "/api/llama/status":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            self._send_json(handler, self._llama_status_payload())
            return
        if path == "/api/llama/logs":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            try:
                tail = int(parse_qs(parsed.query).get("tail", ["200"])[0] or "200")
            except ValueError:
                tail = 200
            self._send_json(handler, self._llama_logs_payload(tail=tail))
            return
        if path == "/api/vision/status":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            self._send_json(handler, self._vision_status_payload())
            return
        if path == "/api/vision/logs":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            try:
                tail = int(parse_qs(parsed.query).get("tail", ["200"])[0] or "200")
            except ValueError:
                tail = 200
            self._send_json(handler, self._vision_logs_payload(tail=tail))
            return
        if path == "/api/dashboard/stats":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            self._send_json(handler, self._dashboard_stats_payload())
            return
        if path == "/api/stickers":
            if not self._authorized(handler):
                self._send_json(handler, {"error": "unauthorized"}, status=401)
                return
            self._send_json(handler, self._sticker_gallery_payload(parse_qs(parsed.query)))
            return
        self._send_text(handler, "not found", status=404)

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        path = urlparse(handler.path).path
        if path.startswith("/api/llama/"):
            self._handle_llama_post(handler, path)
            return
        if path.startswith("/api/vision/"):
            self._handle_vision_post(handler, path)
            return
        if path == "/api/training/run":
            self._handle_training_post(handler)
            return
        if path == "/api/stickers/delete":
            self._handle_sticker_delete_post(handler)
            return
        if path == "/api/stickers/update":
            self._handle_sticker_update_post(handler)
            return
        if path != "/api/config":
            self._send_text(handler, "not found", status=404)
            return
        if not self._authorized(handler):
            self._send_json(handler, {"error": "unauthorized"}, status=401)
            return

        try:
            length = int(handler.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ValueError("request body too large")
            data = json.loads(handler.rfile.read(length).decode("utf-8") or "{}")
            values = data.get("values", data)
            if not isinstance(values, dict):
                raise ValueError("values must be an object")
            result = self.save_callback(values)
            payload = {**self._config_payload(), **result}
            self._send_json(handler, payload)
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=400)

    def _handle_sticker_delete_post(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._authorized(handler):
            self._send_json(handler, {"error": "unauthorized"}, status=401)
            return
        try:
            self._send_json(handler, self._delete_stickers_payload(self._read_json_body(handler)))
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=400)

    def _handle_sticker_update_post(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._authorized(handler):
            self._send_json(handler, {"error": "unauthorized"}, status=401)
            return
        try:
            self._send_json(handler, self._update_sticker_summary_payload(self._read_json_body(handler)))
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=400)

    def _handle_training_post(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._authorized(handler):
            self._send_json(handler, {"error": "unauthorized"}, status=401)
            return
        if not self.trainer:
            self._send_json(handler, {"error": "trainer unavailable"}, status=503)
            return
        try:
            payload = self._read_json_body(handler)
            self._send_json(handler, self._submit_training_action(payload))
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=400)

    def _handle_llama_post(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if not self._authorized(handler):
            self._send_json(handler, {"error": "unauthorized"}, status=401)
            return
        action = path.removeprefix("/api/llama/")
        if action not in {"start", "stop", "restart", "reload", "switch"}:
            self._send_text(handler, "not found", status=404)
            return
        if not self.model_manager:
            self._send_json(handler, {"error": "model manager unavailable"}, status=503)
            return
        try:
            payload = self._read_json_body(handler) if action == "switch" else {}
            self._send_json(handler, self._submit_model_action(action, payload))
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=400)

    def _handle_vision_post(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if not self._authorized(handler):
            self._send_json(handler, {"error": "unauthorized"}, status=401)
            return
        action = path.removeprefix("/api/vision/")
        if action not in {"start", "stop", "restart"}:
            self._send_text(handler, "not found", status=404)
            return
        if not self.vision_manager:
            self._send_json(handler, {"error": "vision manager unavailable"}, status=503)
            return
        try:
            self._send_json(handler, self._submit_vision_action(action))
        except Exception as exc:
            self._send_json(handler, {"error": str(exc)}, status=400)

    async def _run_model_coro(self, coro: Any) -> Any:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self.event_loop or self.event_loop is None:
            return await coro
        if self.event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
            return await asyncio.wrap_future(future)
        return await coro

    def _run_model_coro_sync(self, coro: Any) -> Any:
        if self.event_loop and self.event_loop.is_running():
            try:
                return asyncio.run_coroutine_threadsafe(coro, self.event_loop).result(timeout=240)
            except FutureTimeoutError as exc:
                raise TimeoutError("llama operation timed out") from exc
        return asyncio.run(coro)

    def _read_json_body(self, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 200_000:
            raise ValueError("request body too large")
        data = json.loads(handler.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be an object")
        return data

    def _submit_model_action(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.model_manager:
            return {"status": "model_manager_unavailable", "action": action}
        payload = payload or {}
        with self._pending_action_lock:
            if self._pending_action_future and not self._pending_action_future.done():
                return {"status": "action_in_progress", "action": self._pending_action}
            self._pending_action = ""
            self._pending_action_future = None

            coro = self._model_action_coro(action, payload)
            if self.event_loop and self.event_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
                self._pending_action = action
                self._pending_action_future = future
                future.add_done_callback(lambda done: self._finish_model_action(action, done))
                return {"status": "action_scheduled", "action": action}

        result = asyncio.run(coro)
        payload = self._model_result_payload(result)
        payload["action"] = action
        return payload

    def _model_action_coro(self, action: str, payload: dict[str, Any]) -> Any:
        if action == "switch":
            model = str(payload.get("model", "") or "").strip()
            model_file = str(payload.get("model_file", "") or "").strip()
            if not model and not model_file:
                raise ValueError("model or model_file is required")
            values: dict[str, Any] = {"qwen_persona_llama_model_file": model_file}
            if model:
                values["qwen_persona_llama_model"] = model
            self.save_callback(values)
            return self.model_manager.switch_model(
                model or str(getattr(self.config, "qwen_persona_llama_model", "")),
                model_file,
            )
        return getattr(self.model_manager, action)()

    def _finish_model_action(self, action: str, future: Any) -> None:
        try:
            future.result()
        except Exception as exc:
            if self.model_manager and hasattr(self.model_manager, "_append_log"):
                self.model_manager._append_log(
                    "system",
                    f"admin action {action} failed: {type(exc).__name__}: {exc}",
                )
        finally:
            with self._pending_action_lock:
                if self._pending_action_future is future:
                    self._pending_action = ""
                    self._pending_action_future = None

    def _submit_training_action(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.trainer:
            return {"status": "trainer_unavailable", "action": "training"}
        payload = payload or {}
        reason = str(payload.get("reason", "") or "admin_web").strip()[:120] or "admin_web"
        include_external_corpus = payload.get("include_external_corpus")
        if include_external_corpus is not None:
            include_external_corpus = bool(include_external_corpus)
        full_external_corpus = bool(payload.get("full_external_corpus"))
        with self._pending_training_lock:
            if self._pending_training_future and not self._pending_training_future.done():
                return {
                    "status": "action_in_progress",
                    "action": self._pending_training_action,
                    "reason": self._pending_training_reason,
                }
            self._pending_training_action = ""
            self._pending_training_reason = ""
            self._pending_training_future = None

            coro = self.trainer.run_training(
                reason=reason,
                include_external_corpus=include_external_corpus,
                full_external_corpus=full_external_corpus,
            )
            if self.event_loop and self.event_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
                self._pending_training_action = "training"
                self._pending_training_reason = reason
                self._pending_training_future = future
                future.add_done_callback(lambda done: self._finish_training_action(done))
                return {"status": "action_scheduled", "action": "training", "reason": reason}

        result = asyncio.run(coro)
        self._last_training_result = self._training_result_payload(result)
        payload = dict(self._last_training_result)
        payload["action"] = "training"
        payload["reason"] = reason
        return payload

    def _finish_training_action(self, future: Any) -> None:
        try:
            result = future.result()
            payload = self._training_result_payload(result)
            self._last_training_result = payload
            self.logger.info(
                "qwen persona admin training finished: "
                f"{payload.get('status', 'unknown')}, samples={payload.get('sample_count', '-')}"
            )
        except Exception as exc:
            self.logger.exception(f"qwen persona admin training failed: {exc!r}")
        finally:
            with self._pending_training_lock:
                if self._pending_training_future is future:
                    self._pending_training_action = ""
                    self._pending_training_reason = ""
                    self._pending_training_future = None

    def _llama_status_payload(self) -> dict[str, Any]:
        if not self.model_manager:
            return {
                "runtime": "unavailable",
                "state": "model_manager_unavailable",
                "healthy": False,
                "control": {
                    "can_start": False,
                    "can_stop": False,
                    "can_restart": False,
                    "reason": "model manager unavailable",
                },
            }
        payload = self._run_model_coro_sync(self.model_manager.status_snapshot())
        with self._pending_action_lock:
            pending = self._pending_action if self._pending_action_future and not self._pending_action_future.done() else ""
        if pending:
            payload["operation"] = pending
            control = payload.setdefault("control", {})
            control["can_start"] = False
            control["can_stop"] = False
            control["can_restart"] = False
            control["can_reload"] = False
            control["can_switch"] = False
            control["reason"] = f"{pending} is running"
        payload["model_presets"] = LLAMA_MODEL_PRESETS
        return payload

    def _llama_logs_payload(self, *, tail: int = 200) -> dict[str, Any]:
        if not self.model_manager:
            return {"logs": []}
        tail = max(0, min(int(tail), 1000))
        return {"logs": self.model_manager.logs(tail=tail)}

    def _vision_status_payload(self) -> dict[str, Any]:
        if not self.vision_manager:
            return {
                "state": "vision_manager_unavailable",
                "healthy": False,
                "control": {
                    "can_start": False,
                    "can_stop": False,
                    "can_restart": False,
                    "reason": "vision manager unavailable",
                },
            }
        return self._run_model_coro_sync(self.vision_manager.status_snapshot())

    def _vision_logs_payload(self, *, tail: int = 200) -> dict[str, Any]:
        if not self.vision_manager:
            return {"logs": []}
        tail = max(0, min(int(tail), 1000))
        return {"logs": self.vision_manager.logs(tail=tail)}

    def _submit_vision_action(self, action: str) -> dict[str, Any]:
        if not self.vision_manager:
            return {"status": "vision_manager_unavailable", "action": action}
        if action == "restart" and not hasattr(self.vision_manager, "restart"):
            async def restart_fallback() -> Any:
                await self.vision_manager.stop()
                return await self.vision_manager.start()

            coro = restart_fallback()
        else:
            coro = getattr(self.vision_manager, action)()
        result = self._run_model_coro_sync(coro)
        payload = self._vision_result_payload(result)
        payload["action"] = action
        return payload

    @staticmethod
    def _vision_result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        if is_dataclass(result):
            return asdict(result)
        return {
            "status": getattr(result, "status", "unknown"),
            "endpoint": getattr(result, "endpoint", ""),
            "log": getattr(result, "log", ""),
        }

    def _sticker_gallery_payload(self, query: dict[str, list[str]] | None = None) -> dict[str, Any]:
        query = query or {}
        if not self.store or not hasattr(self.store, "list_stickers"):
            return {
                "items": [],
                "total": 0,
                "limit": self._query_int(query, "limit", 80),
                "offset": self._query_int(query, "offset", 0),
                "query": self._query_str(query, "q", ""),
                "type": self._query_str(query, "type", ""),
                "type_counts": {},
            }
        limit = self._query_int(query, "limit", 80)
        offset = self._query_int(query, "offset", 0)
        search = self._query_str(query, "q", "")
        segment_type = self._query_str(query, "type", "")
        return self._run_model_coro_sync(
            self.store.list_stickers(
                limit=limit,
                offset=offset,
                query=search,
                segment_type=segment_type,
            )
        )

    def _delete_stickers_payload(self, payload: dict[str, Any] | None = None) -> dict[str, int]:
        payload = payload or {}
        raw_keys = payload.get("keys", [])
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        if not isinstance(raw_keys, list):
            raise ValueError("keys must be a list")
        keys = [str(key) for key in raw_keys]
        if not self.store or not hasattr(self.store, "delete_stickers"):
            return {"requested": len([key for key in keys if key.strip()]), "deleted": 0}
        return self._run_model_coro_sync(self.store.delete_stickers(keys))

    def _update_sticker_summary_payload(
        self,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        key = str(payload.get("key") or "").strip()
        if not key:
            raise ValueError("key is required")
        summary = str(payload.get("summary") or "")
        if not self.store or not hasattr(self.store, "update_sticker_summary"):
            return {"updated": 0, "item": None}
        return self._run_model_coro_sync(self.store.update_sticker_summary(key, summary))

    def _dashboard_stats_payload(self) -> dict[str, Any]:
        if self.store and hasattr(self.store, "dashboard_stats"):
            payload = self._run_model_coro_sync(self.store.dashboard_stats())
        else:
            payload = {
                "totals": {
                    "learned_messages": 0,
                    "trainable_samples": 0,
                    "pending_training_samples": 0,
                    "stickers": 0,
                    "training_runs": 0,
                    "groups": 0,
                    "users": 0,
                    "samples_with_stickers": 0,
                    "today_samples": 0,
                    "last_24h_samples": 0,
                    "first_sample_at": None,
                    "last_sample_at": None,
                    "avg_training_chars": None,
                },
                "training": {"last_success": None, "last_run": None, "pending_since": None},
                "charts": {"learning_trend": [], "group_contribution": []},
            }

        client_metrics = (
            self.qwen_client.metrics_snapshot()
            if self.qwen_client and hasattr(self.qwen_client, "metrics_snapshot")
            else {"last": None, "history": []}
        )
        llama_metrics = (
            self.model_manager.performance_snapshot()
            if self.model_manager and hasattr(self.model_manager, "performance_snapshot")
            else {"last": None, "history": []}
        )
        last_tps = client_metrics.get("last") or llama_metrics.get("last")
        performance_history = [
            *list(client_metrics.get("history") or []),
            *list(llama_metrics.get("history") or []),
        ]
        performance_history.sort(key=lambda item: str(item.get("time", "")))
        payload.setdefault("charts", {})["performance_trend"] = performance_history[-40:]
        payload["inference"] = {
            "last_tps": last_tps,
            "client": client_metrics,
            "llama_log": llama_metrics,
        }
        payload["training_operation"] = self._training_operation_payload()
        train_target = bool(
            str(getattr(self.config, "qwen_persona_lora_api_url", "") or "").strip()
            or str(getattr(self.config, "qwen_persona_lora_train_command", "") or "").strip()
        )
        payload["training_config"] = {
            "target_configured": train_target,
            "helper_configured": bool(
                str(getattr(self.config, "qwen_persona_lora_helper_api_url", "") or "").strip()
            ),
            "min_train_samples": int(getattr(self.config, "qwen_persona_min_train_samples", 0) or 0),
            "dataset_limit": int(getattr(self.config, "qwen_persona_dataset_limit", 0) or 0),
            "external_corpus_enabled": bool(
                getattr(self.config, "qwen_persona_external_corpus_enabled", False)
            ),
            "external_corpus_limit": int(
                getattr(self.config, "qwen_persona_external_corpus_limit", 0) or 0
            ),
        }
        return payload

    @staticmethod
    def _query_str(query: dict[str, list[str]], key: str, default: str = "") -> str:
        values = query.get(key, [default])
        return str(values[0] if values else default).strip()

    @classmethod
    def _query_int(cls, query: dict[str, list[str]], key: str, default: int) -> int:
        try:
            return int(cls._query_str(query, key, str(default)) or default)
        except ValueError:
            return default

    def _model_result_payload(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            payload = dict(result)
        elif is_dataclass(result):
            payload = asdict(result)
        else:
            payload = {
                "status": getattr(result, "status", "unknown"),
                "model_dir": getattr(result, "model_dir", ""),
                "repo_id": getattr(result, "repo_id", ""),
                "log": getattr(result, "log", ""),
            }
        if self.model_manager and hasattr(self.model_manager, "_redact_text"):
            payload["log"] = self.model_manager._redact_text(payload.get("log", ""))
        return payload

    def _training_operation_payload(self) -> dict[str, Any]:
        with self._pending_training_lock:
            running = bool(self._pending_training_future and not self._pending_training_future.done())
            progress = (
                self.trainer.progress_snapshot()
                if self.trainer and hasattr(self.trainer, "progress_snapshot")
                else None
            )
            return {
                "running": running,
                "action": self._pending_training_action if running else "",
                "reason": self._pending_training_reason if running else "",
                "last_result": self._last_training_result,
                "available": bool(self.trainer),
                "progress": progress,
            }

    def _training_result_payload(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            payload = dict(result)
        elif is_dataclass(result):
            payload = asdict(result)
        else:
            payload = {
                "status": getattr(result, "status", "unknown"),
                "sample_count": getattr(result, "sample_count", 0),
                "dataset_path": getattr(result, "dataset_path", ""),
                "output_dir": getattr(result, "output_dir", ""),
                "log": getattr(result, "log", ""),
            }
        if self.model_manager and hasattr(self.model_manager, "_redact_text"):
            payload["log"] = self.model_manager._redact_text(payload.get("log", ""))
        if isinstance(payload.get("log"), str):
            payload["log"] = payload["log"][-4000:]
        return payload

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        token = str(getattr(self.config, "qwen_persona_admin_token", "") or "")
        if not token:
            return True
        parsed = urlparse(handler.path)
        query_token = parse_qs(parsed.query).get("token", [""])[0]
        header_token = handler.headers.get("X-Admin-Token", "")
        return token in {query_token, header_token}

    def _config_payload(self) -> dict[str, Any]:
        values = config_to_dict(self.config)
        fields = [self._field_payload(name, value) for name, value in values.items()]
        fields.sort(key=lambda item: (self._category_rank(item["category"]), item["order"], item["name"]))
        return {
            "fields": fields,
            "values": values,
            "override_file": self._override_file_text(),
        }

    def _field_payload(self, name: str, value: Any) -> dict[str, Any]:
        return {
            "name": name,
            "env": name.upper(),
            "title": CONFIG_LABELS.get(name, name),
            "description": CONFIG_DESCRIPTIONS.get(name, ""),
            "category": self._category(name),
            "kind": self._kind(value),
            "widget": self._widget(name),
            "options": self._options(name),
            "allowCustom": self._allow_custom(name),
            "secret": any(marker in name for marker in SECRET_MARKERS),
            "multiline": self._multiline(name, value),
            "default": value,
            "restart_hint": name in RESTART_HINTS,
            "order": self._field_order(name),
            "visibleWhen": self._visible_when(name),
        }

    def _override_file_text(self) -> str:
        from .runtime_config import admin_config_path

        return str(admin_config_path(self.config))

    @staticmethod
    def _kind(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, (list, tuple, set, dict)):
            return "json"
        return "str"

    @staticmethod
    def _multiline(name: str, value: Any) -> bool:
        if isinstance(value, str) and len(value) > 80:
            return True
        return any(part in name for part in ("prompt", "command", "regex", "blacklist"))

    @staticmethod
    def _widget(name: str) -> str:
        if name in {"qwen_persona_runtime", "qwen_persona_llama_model"}:
            return "combo"
        return ""

    @staticmethod
    def _options(name: str) -> list[dict[str, str]]:
        if name == "qwen_persona_runtime":
            return [
                {"value": "llama_cpp", "label": "自管 llama.cpp"},
                {"value": "external_llama", "label": "外部 llama.cpp 服务"},
                {"value": "external_api", "label": "外部 OpenAI-compatible API"},
            ]
        if name == "qwen_persona_llama_model":
            return LLAMA_MODEL_PRESETS
        return []

    @staticmethod
    def _allow_custom(name: str) -> bool:
        return name == "qwen_persona_llama_model"

    @staticmethod
    def _category(name: str) -> str:
        if "admin" in name:
            return "后台网页"
        if "lora" in name or "train" in name:
            return "LoRA 训练"
        if "vision" in name:
            return "视觉理解"
        if "sticker" in name:
            return "表情包"
        if "blacklist" in name:
            return "黑名单"
        if "llama" in name or name == "qwen_persona_runtime" or name == "qwen_persona_model_dir":
            return "模型部署"
        if "api" in name or name.endswith("model") or "temperature" in name or "top_p" in name or "max_tokens" in name:
            return "模型调用"
        if "learning" in name or "reply" in name or "context" in name or "group" in name or "sample" in name or "system_prompt" in name:
            return "学习与回复"
        return "其他"

    @staticmethod
    def _category_rank(category: str) -> int:
        try:
            return ["后台网页", "模型部署", "模型调用", "视觉理解", "学习与回复", "表情包", "LoRA 训练", "黑名单", "其他"].index(category)
        except ValueError:
            return 99

    @staticmethod
    def _field_order(name: str) -> int:
        order = {
            "qwen_persona_runtime": 0,
            "qwen_persona_llama_host": 10,
            "qwen_persona_llama_port": 11,
            "qwen_persona_llama_model": 12,
            "qwen_persona_llama_model_file": 13,
            "qwen_persona_llama_server_path": 20,
            "qwen_persona_llama_auto_install": 21,
            "qwen_persona_llama_ctx_size": 22,
            "qwen_persona_llama_extra_args": 23,
            "qwen_persona_vision_enabled": 0,
            "qwen_persona_vision_auto_start": 1,
            "qwen_persona_vision_host": 2,
            "qwen_persona_vision_port": 3,
            "qwen_persona_vision_model": 4,
            "qwen_persona_vision_python": 5,
        }
        return order.get(name, 100)

    @staticmethod
    def _visible_when(name: str) -> dict[str, list[str]]:
        self_managed = ["llama_cpp"]
        llama_modes = ["llama_cpp", "external_llama"]
        external_api = ["external_api"]
        visibility = {
            "qwen_persona_api_base": external_api,
            "qwen_persona_model": external_api,
            "qwen_persona_llama_server_path": self_managed,
            "qwen_persona_llama_auto_install": self_managed,
            "qwen_persona_llama_model_file": self_managed,
            "qwen_persona_llama_ctx_size": self_managed,
            "qwen_persona_llama_extra_args": self_managed,
            "qwen_persona_model_dir": self_managed,
            "qwen_persona_llama_host": llama_modes,
            "qwen_persona_llama_port": llama_modes,
            "qwen_persona_llama_model": llama_modes,
        }
        runtimes = visibility.get(name)
        return {"runtime": runtimes} if runtimes else {}

    @staticmethod
    def _vendor_asset_payload(path: str) -> dict[str, Any]:
        asset = VENDOR_ASSETS.get(path)
        if not asset:
            raise FileNotFoundError(path)
        relative_path, content_type = asset
        return {
            "data": (PROJECT_ROOT / relative_path).read_bytes(),
            "content_type": content_type,
        }

    @staticmethod
    def _send_html(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def _send_bytes(
        handler: BaseHTTPRequestHandler,
        data: bytes,
        content_type: str,
        status: int = 200,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "public, max-age=31536000, immutable")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def _send_text(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
        data = escape(body).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/plain; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)




