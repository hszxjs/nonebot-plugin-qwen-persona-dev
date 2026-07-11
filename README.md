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
- Chat and image/sticker understanding run through embedded `mlx-vlm`; there is no model or
  vision sidecar process.
- The default model is `mlx-community/Qwen3.5-9B-MLX-4bit`.
- LoRA training defaults to `python -m mlx_vlm.lora` and exports both local chat samples and,
  when enabled, the full external corpus. Multimodal samples are whitelist-only media assets;
  ordinary images are used for immediate reply context and are not persisted as training assets.
- Runtime data is not included in this package. Existing `data/qwen_persona/` remains in the host bot project.

## Verify

```bash
python -m py_compile nonebot_plugin_qwen_persona/*.py
python -m unittest discover -s tests
```
