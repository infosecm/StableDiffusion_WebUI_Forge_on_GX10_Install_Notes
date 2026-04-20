# Stable Diffusion WebUI Forge on Asus Ascent GX10 (NVIDIA GB10 / CUDA 13.0 / aarch64)

This guide documents a working installation of [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) on the **Asus Ascent GX10**, which uses the **NVIDIA GB10 Blackwell** chip (compute capability sm_121), running **Ubuntu 24.04 on aarch64**, with **CUDA 13.0** and **Python 3.12**.

This setup is unusually complex because the GB10 is an edge AI workstation chip — not a consumer GPU — and its software ecosystem (PyTorch, gradio, starlette, transformers) lags behind mainstream x86 CUDA support. Every dependency version choice in this guide is deliberate.

---

## Table of Contents

- [Stable Diffusion WebUI Forge on Asus Ascent GX10 (NVIDIA GB10 / CUDA 13.0 / aarch64)](#stable-diffusion-webui-forge-on-asus-ascent-gx10-nvidia-gb10--cuda-130--aarch64)
  - [Table of Contents](#table-of-contents)
  - [Automated Installation](#automated-installation)
  - [System Specs](#system-specs)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Clone Forge](#step-1--clone-forge)
  - [Step 2 — Create an Isolated Virtual Environment](#step-2--create-an-isolated-virtual-environment)
  - [Step 3 — Install PyTorch with CUDA 13.0 Support](#step-3--install-pytorch-with-cuda-130-support)
  - [Step 4 — Install All Dependencies](#step-4--install-all-dependencies)
  - [Step 5 — Apply Required Code Patches](#step-5--apply-required-code-patches)
    - [5.1 — Fix gradio\_client: `bool` is not iterable](#51--fix-gradio_client-bool-is-not-iterable)
    - [5.2 — Fix gradio oauth: `HfFolder` removed from huggingface\_hub 1.x](#52--fix-gradio-oauth-hffolder-removed-from-huggingface_hub-1x)
    - [5.3 — Fix gradio blocks: localhost accessibility check](#53--fix-gradio-blocks-localhost-accessibility-check)
    - [5.4 — Fix gradio routes: api\_info TypeError + TemplateResponse signature](#54--fix-gradio-routes-api_info-typeerror--templateresponse-signature)
  - [Step 6 — Link Models from ComfyUI (Optional)](#step-6--link-models-from-comfyui-optional)
  - [Step 7 — Create the Launch Script](#step-7--create-the-launch-script)
  - [Step 8 — Launch](#step-8--launch)
  - [Quick Reinstall Reference](#quick-reinstall-reference)
  - [Summary of Patched Files](#summary-of-patched-files)
  - [References and Links](#references-and-links)
    - [Core Projects](#core-projects)
    - [Python Libraries](#python-libraries)
    - [Model Sources](#model-sources)
    - [NVIDIA / Hardware References](#nvidia--hardware-references)

---

## Automated Installation

For convenience, a fully automated installer script is provided alongside this README. It performs all steps below interactively, asking only for the install directory and whether to link an existing ComfyUI model folder.

**Usage:**

```bash
bash install_forge_gx10.sh
```

The script will:
1. Run preflight checks (OS, Python version, GPU, CUDA, git)
2. Ask for the Forge installation directory (default: `~/stable-diffusion-webui-forge`)
3. Ask whether to share models with an existing ComfyUI installation
4. Clone Forge, create the venv, install all dependencies
5. Apply all four code patches automatically
6. Create symlinks to ComfyUI models if requested
7. Generate `~/start_forge.sh`
8. Optionally launch Forge immediately

> **Note:** The manual steps below are documented for transparency, troubleshooting, and reproducibility. The automated script performs the exact same operations.

---

## System Specs

| Component | Details |
|---|---|
| Device | Asus Ascent GX10 |
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| VRAM | 124 GB unified memory |
| OS | Ubuntu 24.04 (aarch64) |
| CUDA | 13.0 |
| Driver | 580.82.09 |
| Python | 3.12.3 |

Verify your setup:
```bash
nvidia-smi
python3 --version
```

---

## Prerequisites

```bash
sudo apt install -y git libjpeg-dev zlib1g-dev libpng-dev
```

The image libraries (`libjpeg-dev`, `zlib1g-dev`, `libpng-dev`) are required for Pillow to compile from source when no prebuilt aarch64 wheel is available.

---

## Step 1 — Clone Forge

```bash
cd ~
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git
cd stable-diffusion-webui-forge
```

---

## Step 2 — Create an Isolated Virtual Environment

We use a **separate venv** named `.venv_forge` so this installation never touches your existing ComfyUI environment.

```bash
python3 -m venv .venv_forge
source .venv_forge/bin/activate
pip install --upgrade pip
```

---

## Step 3 — Install PyTorch with CUDA 13.0 Support

The GB10 requires `cu130` wheels. As of early 2026, these are available at the PyTorch index:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

Verify GPU detection:
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True / NVIDIA GB10
```

> **Why cu130?** The GB10 reports CUDA 13.0 (`nvidia-smi` confirms this). The `cu128` wheels that work on consumer Blackwell RTX cards will not work here. The `cu130` index was added to PyTorch in late 2025 specifically for GB10/DGX Spark support.

---

## Step 4 — Install All Dependencies

This installs the full dependency stack with versions carefully chosen for compatibility with Python 3.12, aarch64, CUDA 13.0, and Forge f2.0.1.

```bash
pip install --prefer-binary \
  "setuptools==71.0.0" \
  "fastapi==0.104.1" \
  "starlette==0.27.0" \
  "pydantic==1.10.21" \
  "gradio==4.44.1" \
  "gradio-client==1.3.0" \
  "uvicorn" \
  "transformers==4.49.0" \
  "diffusers==0.32.0" \
  "accelerate" \
  "safetensors" \
  "einops" \
  "omegaconf" \
  "kornia" \
  "opencv-python-headless" \
  "GitPython" \
  "tqdm" \
  "psutil" \
  "Pillow" \
  "requests" \
  "pytorch_lightning" \
  "torchdiffeq" \
  "torchsde" \
  "lark" \
  "inflection" \
  "diskcache" \
  "blendmodes" \
  "clean-fid" \
  "resize-right" \
  "piexif" \
  "facexlib" \
  "joblib" \
  "open-clip-torch" \
  "scikit-image>=0.22.0" \
  "pillow-avif-plugin" \
  "sentencepiece"
```

Then install packages that have conflicting declared dependencies, bypassing the resolver:

```bash
# gradio-rangeslider requires gradio>=4.0 but declares it as a hard dep —
# install without dep resolution since we already have a compatible gradio
pip install gradio-rangeslider --no-deps

# CLIP from OpenAI — old setup.py requires --no-build-isolation on Python 3.12
pip install git+https://github.com/openai/CLIP.git --no-build-isolation
```

> **Version notes:**
> - `setuptools==71.0.0` — torch 2.11+cu130 requires `setuptools<82`
> - `fastapi==0.104.1` + `starlette==0.27.0` + `pydantic==1.10.21` — Forge's internal routing code uses pydantic v1 `FieldInfo.in_` attribute, which was removed in pydantic v2. starlette 1.x changed `TemplateResponse` signature incompatibly.
> - `gradio==4.44.1` — Forge f2.0.1 requires gradio 4.x (`gradio.component_meta` does not exist in 3.x)
> - `transformers==4.49.0` + `diffusers==0.32.0` — transformers 5.x removed `no_init_weights` used by Forge; diffusers 0.37+ requires `Dinov2WithRegistersConfig` only available in transformers>=4.49
> - `scikit-image>=0.22.0` — Forge pins 0.21.0 which has no prebuilt aarch64 wheel and fails to compile under meson on Ubuntu 24.04

---

## Step 5 — Apply Required Code Patches

These patches fix incompatibilities between newer library versions and Forge's expected APIs. All patches are applied to files inside the venv.

### 5.1 — Fix gradio_client: `bool` is not iterable

Forge's gradio schema introspection passes `bool` values where `gradio_client` expects `dict`. Every `in schema` and `.get()` call in `get_type()` must guard against non-dict input:

```bash
UTILS=".venv_forge/lib/python3.12/site-packages/gradio_client/utils.py"

sed -i 's/if "const" in schema:/if isinstance(schema, dict) and "const" in schema:/g' $UTILS
sed -i 's/if "enum" in schema:/if isinstance(schema, dict) and "enum" in schema:/g' $UTILS
sed -i 's/elif schema\.get("\$ref"):/elif isinstance(schema, dict) and schema.get("\$ref"):/g' $UTILS
sed -i 's/elif schema\.get("oneOf"):/elif isinstance(schema, dict) and schema.get("oneOf"):/g' $UTILS
sed -i 's/elif schema\.get("anyOf"):/elif isinstance(schema, dict) and schema.get("anyOf"):/g' $UTILS
sed -i 's/elif schema\.get("allOf"):/elif isinstance(schema, dict) and schema.get("allOf"):/g' $UTILS
sed -i 's/elif "type" not in schema:/elif not isinstance(schema, dict) or "type" not in schema:/g' $UTILS
sed -i 's/if "json" in schema\.get("description", {}):/if isinstance(schema, dict) and "json" in schema.get("description", {}):/g' $UTILS
```

### 5.2 — Fix gradio oauth: `HfFolder` removed from huggingface_hub 1.x

`huggingface_hub` 1.x removed `HfFolder`. Gradio 4.44.1's oauth module still imports it:

```bash
OAUTH=".venv_forge/lib/python3.12/site-packages/gradio/oauth.py"

sed -i 's/from huggingface_hub import HfFolder, whoami/from huggingface_hub import whoami\ntry:\n    from huggingface_hub import HfFolder\nexcept ImportError:\n    HfFolder = None/' $OAUTH
```

### 5.3 — Fix gradio blocks: localhost accessibility check

On the GX10, gradio's internal check `networking.url_ok(self.local_url)` fails even though `127.0.0.1` is reachable, causing a `ValueError` that prevents launch. Bypass the check:

```bash
BLOCKS=".venv_forge/lib/python3.12/site-packages/gradio/blocks.py"

# Find the exact line number (should be around 2462)
BLOCKS_LINE=$(grep -n "and not networking.url_ok(self.local_url)" "$BLOCKS" | head -1 | cut -d: -f1)

# Replace the condition on that line
sed -i "${BLOCKS_LINE}s/and not networking.url_ok(self.local_url)/and False  # patched: skip localhost check/" $BLOCKS
```

### 5.4 — Fix gradio routes: api_info TypeError + TemplateResponse signature

Two issues in `routes.py`:

1. `api_info(False)` throws `TypeError` due to the bool-in-schema issue
2. starlette 0.27.0's `TemplateResponse` uses positional args, but newer starlette changed the signature, causing `unhashable type: 'dict'`

```bash
ROUTES=".venv_forge/lib/python3.12/site-packages/gradio/routes.py"

# Patch 1: wrap api_info call in try/except
python3 -c "
content = open('$ROUTES').read()
old = '                gradio_api_info = api_info(False)'
new = '''                try:
                    gradio_api_info = api_info(False)
                except Exception:
                    gradio_api_info = {\"version\": \"0.0\", \"named_endpoints\": {}, \"unnamed_endpoints\": {}}'''
open('$ROUTES', 'w').write(content.replace(old, new, 1))
print('Patch 1 OK')
"

# Patch 2: fix TemplateResponse call signature
python3 -c "
content = open('$ROUTES').read()
old = '''                return templates.TemplateResponse(
                    template,
                    {
                        \"request\": request,
                        \"config\": config,
                        \"gradio_api_info\": gradio_api_info,
                    },
                )'''
new = '''                return templates.TemplateResponse(
                    name=template,
                    request=request,
                    context={
                        \"config\": config,
                        \"gradio_api_info\": gradio_api_info,
                    },
                )'''
result = content.replace(old, new, 1)
open('$ROUTES', 'w').write(result)
print('Patch 2 OK' if 'request=request' in result else 'FAILED')
"
```

---

## Step 6 — Link Models from ComfyUI (Optional)

If you already have models in a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installation, you can share them with Forge via symlinks — no duplication needed:

```bash
# Remove the empty placeholder directories Forge created
rm -r ~/stable-diffusion-webui-forge/models/Stable-diffusion
rm -r ~/stable-diffusion-webui-forge/models/VAE
rm -r ~/stable-diffusion-webui-forge/models/Lora

# Create symlinks pointing to your ComfyUI model folders
ln -s ~/ComfyUI/models/checkpoints      ~/stable-diffusion-webui-forge/models/Stable-diffusion
ln -s ~/ComfyUI/models/vae              ~/stable-diffusion-webui-forge/models/VAE
ln -s ~/ComfyUI/models/loras            ~/stable-diffusion-webui-forge/models/Lora
ln -s ~/ComfyUI/models/controlnet       ~/stable-diffusion-webui-forge/models/ControlNet
ln -s ~/ComfyUI/models/embeddings       ~/stable-diffusion-webui-forge/models/embeddings
ln -s ~/ComfyUI/models/upscale_models   ~/stable-diffusion-webui-forge/models/ESRGAN
ln -s ~/ComfyUI/models/clip             ~/stable-diffusion-webui-forge/models/CLIP
ln -s ~/ComfyUI/models/unet             ~/stable-diffusion-webui-forge/models/unet
ln -s ~/ComfyUI/models/text_encoders    ~/stable-diffusion-webui-forge/models/text_encoder
ln -s ~/ComfyUI/models/diffusion_models ~/stable-diffusion-webui-forge/models/diffusion_models
```

---

## Step 7 — Create the Launch Script

```bash
cat > ~/start_forge.sh << 'EOF'
#!/bin/bash
cd ~/stable-diffusion-webui-forge
source .venv_forge/bin/activate
python launch.py \
  --listen \
  --skip-python-version-check \
  --skip-install \
  --skip-version-check \
  --enable-insecure-extension-access \
  --always-gpu
EOF
chmod +x ~/start_forge.sh
```

**Flag explanations:**
- `--listen` — bind to `0.0.0.0:7860` so the UI is accessible from the network
- `--skip-python-version-check` — Forge expects Python 3.10.6; we're on 3.12.3
- `--skip-install` — prevents Forge from trying to reinstall dependencies (which would overwrite our pinned versions)
- `--skip-version-check` — suppresses the gradio version mismatch warning
- `--enable-insecure-extension-access` — required for extensions to load properly
- `--always-gpu` — keeps the model loaded in VRAM between generations (with 124 GB unified memory this is always safe)

---

## Step 8 — Launch

```bash
~/start_forge.sh
```

Open your browser at `http://127.0.0.1:7860`

The UI should load with your models available in the **Checkpoint** dropdown. Select a model and generate.

---

## Quick Reinstall Reference

If you need to rebuild the venv from scratch:

```bash
cd ~/stable-diffusion-webui-forge
rm -rf .venv_forge
python3 -m venv .venv_forge
source .venv_forge/bin/activate
```

Then repeat Steps 3–5.

---

## Summary of Patched Files

| File | Patch | Reason |
|---|---|---|
| `gradio_client/utils.py` | `isinstance(schema, dict)` guards in `get_type()` | Forge passes `bool` where dict is expected |
| `gradio/oauth.py` | Try/except around `HfFolder` import | Removed from `huggingface_hub` 1.x |
| `gradio/blocks.py` | Skip `url_ok` localhost check | False negative on GX10 network stack |
| `gradio/routes.py` | Wrap `api_info()` + fix `TemplateResponse` signature | starlette 0.27 API mismatch |

---

## References and Links

### Core Projects

| Project | GitHub | Description |
|---|---|---|
| Stable Diffusion WebUI Forge | [lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) | The WebUI this guide installs |
| ComfyUI | [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Node-based UI, installed alongside Forge |
| AUTOMATIC1111 WebUI | [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | Original WebUI that Forge forks from |

### Python Libraries

| Library | GitHub | Version Used | Notes |
|---|---|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | 2.11.0+cu130 | CUDA 13.0 build for GB10 |
| Gradio | [gradio-app/gradio](https://github.com/gradio-app/gradio) | 4.44.1 | UI framework for Forge |
| Hugging Face Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | 4.49.0 | Required by Forge model loading |
| Diffusers | [huggingface/diffusers](https://github.com/huggingface/diffusers) | 0.32.0 | Flux pipeline support |
| FastAPI | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) | 0.104.1 | Forge's API server |
| Starlette | [encode/starlette](https://github.com/encode/starlette) | 0.27.0 | ASGI framework under FastAPI |
| Pydantic | [pydantic/pydantic](https://github.com/pydantic/pydantic) | 1.10.21 | v1 required by Forge routing code |
| CLIP | [openai/CLIP](https://github.com/openai/CLIP) | latest | OpenAI CLIP model |
| open_clip | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | latest | Open-source CLIP implementation |
| PyTorch Lightning | [Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) | latest | Training framework used by Forge |
| k-diffusion | [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion) | bundled | Sampling algorithms (bundled in Forge) |
| ControlNet (Forge built-in) | [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) | built-in | ControlNet extension, bundled with Forge |
| facexlib | [xinntao/facexlib](https://github.com/xinntao/facexlib) | latest | Face detection and restoration utilities |
| kornia | [kornia/kornia](https://github.com/kornia/kornia) | latest | Computer vision library |
| safetensors | [huggingface/safetensors](https://github.com/huggingface/safetensors) | latest | Safe model file format |
| torchdiffeq | [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) | latest | ODE solvers for samplers |
| torchsde | [google-research/torchsde](https://github.com/google-research/torchsde) | latest | SDE solvers for samplers |


### Model Sources

| Source | URL | Description |
|---|---|---|
| CivitAI | [civitai.com](https://civitai.com) | Community models |
| Hugging Face | [huggingface.co](https://huggingface.co) | Official model hub |
| SDXL VAE | [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae) | Recommended VAE for all SDXL-based models |

### NVIDIA / Hardware References

| Resource | URL | Description |
|---|---|---|
| NVIDIA Developer Forums — GB10/DGX Spark | [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663) | SM121 software support discussion |
| PyTorch CUDA 13.0 issue | [pytorch/pytorch#159779](https://github.com/pytorch/pytorch/issues/159779) | Tracking issue for CUDA 13.0 binary support |
