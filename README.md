# nonebot-plugin-qwen-persona

Local development package extracted from `fallen_rainy`.

## Install From Git Clone

```bash
git clone <your-repo-url>
cd nonebot-plugin-qwen-persona
uv pip install -e .
```

Then keep the host bot project loading:

```toml
[tool.nonebot.plugins]
nonebot_plugin_qwen_persona = ["nonebot_plugin_qwen_persona"]
```

## Notes

- Default admin WebUI stays on `127.0.0.1:8082`.
- Default self-managed llama.cpp endpoint stays on `127.0.0.1:18080`.
- Default self-managed vision endpoint stays on `127.0.0.1:18081`.
- External llama/API modes remain read-only from WebUI.
- Image/sticker understanding is served by the built-in vision service. The default model is
  CLIP zero-shot image classification; you can switch the model setting to BLIP or another
  transformers image-to-text model if you want caption-style descriptions.
- Install the optional `vision` extra for real transformers inference. Without it, captions
  safely fall back to generic text and never store raw OneBot media identifiers.
- Runtime data is not included in this package. Existing `data/qwen_persona/` remains in the host bot project.

## Verify

```bash
python -m py_compile nonebot_plugin_qwen_persona/*.py tests/test_qwen_persona_observability.py
python -m unittest tests/test_qwen_persona_observability.py -v
```
