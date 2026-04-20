#!/bin/bash
# =============================================================================
# Stable Diffusion WebUI Forge — Installer for Asus Ascent GX10
# NVIDIA GB10 / CUDA 13.0 / aarch64 / Ubuntu 24.04 / Python 3.12
# =============================================================================
# Usage: bash install_forge_gx10.sh
# =============================================================================

set -e  # Exit on any unhandled error

# --- Colors ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Helpers -----------------------------------------------------------------
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
section() { echo -e "\n${BOLD}========== $1 ==========${NC}"; }

ask_yes_no() {
    # Usage: ask_yes_no "Question?" "y"  (second arg = default)
    local prompt="$1"
    local default="${2:-y}"
    local yn
    if [[ "$default" == "y" ]]; then
        read -rp "$prompt [Y/n] " yn
        yn="${yn:-y}"
    else
        read -rp "$prompt [y/N] " yn
        yn="${yn:-n}"
    fi
    [[ "$yn" =~ ^[Yy]$ ]]
}

ask_path() {
    # Usage: ask_path "Question?" "/default/path"
    local prompt="$1"
    local default="$2"
    local result
    read -rp "$prompt [$default]: " result
    echo "${result:-$default}"
}

# =============================================================================
# BANNER
# =============================================================================
clear
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Stable Diffusion WebUI Forge — GX10 Installer               ║"
echo "║     NVIDIA GB10 · CUDA 13.0 · aarch64 · Python 3.12             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "This script will install Forge alongside your existing ComfyUI"
echo "installation, using a completely isolated Python virtual environment."
echo ""
echo "Estimated time: 10–20 minutes depending on download speed."
echo ""

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================
section "Preflight Checks"

# Check OS
if [[ "$(uname -m)" != "aarch64" ]]; then
    warn "This script is designed for aarch64 (GX10). Detected: $(uname -m)"
    ask_yes_no "Continue anyway?" "n" || exit 0
fi

# Check Python 3.12
PYTHON=$(command -v python3 || true)
if [[ -z "$PYTHON" ]]; then
    error "python3 not found. Please install Python 3.12."
fi
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
info "Python version: $PY_VERSION"
if [[ ! "$PY_VERSION" =~ ^3\.12 ]]; then
    warn "Python 3.12 is recommended. Found $PY_VERSION."
    ask_yes_no "Continue with $PY_VERSION?" "y" || exit 0
fi

# Check NVIDIA driver and CUDA
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Please install NVIDIA drivers first."
fi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
info "GPU: $GPU_NAME"
info "CUDA: $CUDA_VERSION"
if [[ ! "$CUDA_VERSION" =~ ^13 ]]; then
    warn "CUDA 13.0 is expected. Found $CUDA_VERSION."
    ask_yes_no "Continue?" "y" || exit 0
fi

# Check git
if ! command -v git &>/dev/null; then
    error "git not found. Run: sudo apt install git"
fi

success "Preflight checks passed."

# =============================================================================
# CONFIGURATION
# =============================================================================
section "Configuration"

echo ""
echo "Where would you like to install Forge?"
FORGE_DIR=$(ask_path "Installation directory" "$HOME/stable-diffusion-webui-forge")

echo ""
echo "Do you have an existing ComfyUI installation to share models with?"
if ask_yes_no "Share models with ComfyUI?" "y"; then
    COMFY_DIR=$(ask_path "ComfyUI directory" "$HOME/ComfyUI")
    if [[ ! -d "$COMFY_DIR/models" ]]; then
        warn "Directory $COMFY_DIR/models not found."
        ask_yes_no "Continue without model linking?" "y" || exit 0
        COMFY_DIR=""
    else
        success "ComfyUI models found at $COMFY_DIR/models"
    fi
else
    COMFY_DIR=""
fi

echo ""
echo "Summary:"
echo "  Forge install dir : $FORGE_DIR"
echo "  ComfyUI dir       : ${COMFY_DIR:-"(none — skipping model links)"}"
echo ""
ask_yes_no "Proceed with installation?" "y" || exit 0

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
section "System Dependencies"

info "Installing system libraries required to compile Pillow from source..."
sudo apt install -y libjpeg-dev zlib1g-dev libpng-dev
success "System dependencies installed."

# =============================================================================
# CLONE FORGE
# =============================================================================
section "Cloning Forge Repository"

if [[ -d "$FORGE_DIR/.git" ]]; then
    warn "Forge directory already exists at $FORGE_DIR"
    if ask_yes_no "Re-use existing clone?" "y"; then
        info "Using existing clone."
    else
        info "Removing and re-cloning..."
        rm -rf "$FORGE_DIR"
        git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git "$FORGE_DIR"
    fi
else
    git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git "$FORGE_DIR"
fi

cd "$FORGE_DIR"
success "Forge cloned to $FORGE_DIR"

# =============================================================================
# VIRTUAL ENVIRONMENT
# =============================================================================
section "Creating Virtual Environment"

VENV_DIR="$FORGE_DIR/.venv_forge"

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR"
    if ask_yes_no "Delete and recreate venv?" "n"; then
        rm -rf "$VENV_DIR"
    else
        info "Using existing venv."
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

# Activate venv for the rest of the script
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
success "Virtual environment activated."

# =============================================================================
# PYTORCH WITH CUDA 13.0
# =============================================================================
section "Installing PyTorch (CUDA 13.0 / cu130)"

info "This may take several minutes..."
pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Verify GPU detection
info "Verifying GPU detection..."
GPU_DETECTED=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "FAILED")
if [[ "$GPU_DETECTED" == "FAILED" ]]; then
    error "PyTorch cannot detect the GPU. Check your CUDA installation."
fi
success "PyTorch installed. GPU detected: $GPU_DETECTED"

# =============================================================================
# PYTHON DEPENDENCIES
# =============================================================================
section "Installing Python Dependencies"

info "Installing core dependencies with pinned versions..."
info "This will take several minutes..."

pip install --prefer-binary --quiet \
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

success "Core dependencies installed."

info "Installing gradio-rangeslider (bypassing dep resolver — incompatible declaration)..."
pip install gradio-rangeslider --no-deps --quiet
success "gradio-rangeslider installed."

info "Installing CLIP (requires --no-build-isolation on Python 3.12)..."
pip install git+https://github.com/openai/CLIP.git --no-build-isolation --quiet
success "CLIP installed."

# =============================================================================
# CODE PATCHES
# =============================================================================
section "Applying Code Patches"

UTILS="$VENV_DIR/lib/python3.12/site-packages/gradio_client/utils.py"
OAUTH="$VENV_DIR/lib/python3.12/site-packages/gradio/oauth.py"
BLOCKS="$VENV_DIR/lib/python3.12/site-packages/gradio/blocks.py"
ROUTES="$VENV_DIR/lib/python3.12/site-packages/gradio/routes.py"

# --- Patch 1: gradio_client/utils.py -----------------------------------------
# Forge passes bool values where gradio_client expects dict in its JSON schema
# parser. Every branch in get_type() and _json_schema_to_python_type() that
# uses `in schema` or `schema.get()` must guard with isinstance(schema, dict).
info "Patching gradio_client/utils.py (bool-in-schema TypeError)..."

sed -i 's/if "const" in schema:/if isinstance(schema, dict) and "const" in schema:/g'          "$UTILS"
sed -i 's/if "enum" in schema:/if isinstance(schema, dict) and "enum" in schema:/g'             "$UTILS"
sed -i 's/elif schema\.get("\$ref"):/elif isinstance(schema, dict) and schema.get("\$ref"):/g'  "$UTILS"
sed -i 's/elif schema\.get("oneOf"):/elif isinstance(schema, dict) and schema.get("oneOf"):/g'  "$UTILS"
sed -i 's/elif schema\.get("anyOf"):/elif isinstance(schema, dict) and schema.get("anyOf"):/g'  "$UTILS"
sed -i 's/elif schema\.get("allOf"):/elif isinstance(schema, dict) and schema.get("allOf"):/g'  "$UTILS"
sed -i 's/elif "type" not in schema:/elif not isinstance(schema, dict) or "type" not in schema:/g' "$UTILS"
sed -i 's/if "json" in schema\.get("description", {}):/if isinstance(schema, dict) and "json" in schema.get("description", {}):/g' "$UTILS"

success "gradio_client/utils.py patched."

# --- Patch 2: gradio/oauth.py -------------------------------------------------
# huggingface_hub 1.x removed HfFolder. gradio 4.44.1 still imports it.
# We wrap the import in a try/except so gradio loads without authentication.
info "Patching gradio/oauth.py (HfFolder removed from huggingface_hub 1.x)..."

if grep -q "from huggingface_hub import HfFolder, whoami" "$OAUTH"; then
    sed -i 's/from huggingface_hub import HfFolder, whoami/from huggingface_hub import whoami\ntry:\n    from huggingface_hub import HfFolder\nexcept ImportError:\n    HfFolder = None/' "$OAUTH"
    success "gradio/oauth.py patched."
else
    warn "gradio/oauth.py — patch already applied or line not found, skipping."
fi

# --- Patch 3: gradio/blocks.py ------------------------------------------------
# On the GX10, gradio's networking.url_ok(self.local_url) returns False even
# though 127.0.0.1 is reachable, causing Forge to refuse to start unless
# share=True. We bypass the check entirely.
info "Patching gradio/blocks.py (localhost accessibility false negative)..."

BLOCKS_LINE=$(grep -n "and not networking.url_ok(self.local_url)" "$BLOCKS" | head -1 | cut -d: -f1)
if [[ -n "$BLOCKS_LINE" ]]; then
    sed -i "${BLOCKS_LINE}s/and not networking.url_ok(self.local_url)/and False  # patched: skip localhost check/" "$BLOCKS"
    success "gradio/blocks.py patched (line $BLOCKS_LINE)."
else
    warn "gradio/blocks.py — patch target not found, may already be applied."
fi

# --- Patch 4: gradio/routes.py ------------------------------------------------
# Two issues:
# (a) api_info(False) throws TypeError — wrap in try/except returning a stub
# (b) starlette 0.27.0 TemplateResponse expects positional args but the newer
#     starlette API changed the signature, causing "unhashable type: dict"
info "Patching gradio/routes.py (api_info TypeError + TemplateResponse signature)..."

python3 -c "
import sys
routes = '$ROUTES'
content = open(routes).read()

# Patch (a): wrap api_info call
old_a = '                gradio_api_info = api_info(False)'
new_a = '''                try:
                    gradio_api_info = api_info(False)
                except Exception:
                    gradio_api_info = {\"version\": \"0.0\", \"named_endpoints\": {}, \"unnamed_endpoints\": {}}'''
if old_a in content:
    content = content.replace(old_a, new_a, 1)
    print('  api_info patch applied')
else:
    print('  api_info patch already applied or not found')

# Patch (b): fix TemplateResponse call
old_b = '''                return templates.TemplateResponse(
                    template,
                    {
                        \"request\": request,
                        \"config\": config,
                        \"gradio_api_info\": gradio_api_info,
                    },
                )'''
new_b = '''                return templates.TemplateResponse(
                    name=template,
                    request=request,
                    context={
                        \"config\": config,
                        \"gradio_api_info\": gradio_api_info,
                    },
                )'''
if old_b in content:
    content = content.replace(old_b, new_b, 1)
    print('  TemplateResponse patch applied')
else:
    print('  TemplateResponse patch already applied or not found')

open(routes, 'w').write(content)
"

success "gradio/routes.py patched."

# =============================================================================
# MODEL SYMLINKS
# =============================================================================
section "Model Directory Setup"

MODELS_DIR="$FORGE_DIR/models"

if [[ -n "$COMFY_DIR" ]]; then
    info "Creating symlinks from Forge models/ to ComfyUI models/..."

    # Helper: remove forge placeholder dir and create symlink
    link_model() {
        local forge_name="$1"
        local comfy_path="$2"
        local dest="$MODELS_DIR/$forge_name"
        if [[ ! -d "$comfy_path" ]]; then
            warn "  Skipping $forge_name — $comfy_path not found"
            return
        fi
        if [[ -L "$dest" ]]; then
            info "  $forge_name already linked, skipping."
        elif [[ -d "$dest" ]]; then
            rm -rf "$dest"
            ln -s "$comfy_path" "$dest"
            success "  Linked $forge_name → $comfy_path"
        else
            ln -s "$comfy_path" "$dest"
            success "  Linked $forge_name → $comfy_path"
        fi
    }

    link_model "Stable-diffusion" "$COMFY_DIR/models/checkpoints"
    link_model "VAE"              "$COMFY_DIR/models/vae"
    link_model "Lora"             "$COMFY_DIR/models/loras"
    link_model "ControlNet"       "$COMFY_DIR/models/controlnet"
    link_model "embeddings"       "$COMFY_DIR/models/embeddings"
    link_model "ESRGAN"           "$COMFY_DIR/models/upscale_models"
    link_model "CLIP"             "$COMFY_DIR/models/clip"
    link_model "unet"             "$COMFY_DIR/models/unet"
    link_model "text_encoder"     "$COMFY_DIR/models/text_encoders"
    link_model "diffusion_models" "$COMFY_DIR/models/diffusion_models"
else
    info "Skipping model links (no ComfyUI directory configured)."
    info "Place your .safetensors models in: $MODELS_DIR/Stable-diffusion/"
fi

# =============================================================================
# LAUNCH SCRIPT
# =============================================================================
section "Creating Launch Script"

LAUNCH_SCRIPT="$HOME/start_forge.sh"

cat > "$LAUNCH_SCRIPT" << EOF
#!/bin/bash
# Stable Diffusion WebUI Forge — Launch script for Asus Ascent GX10
# Generated by install_forge_gx10.sh

cd $FORGE_DIR
source .venv_forge/bin/activate

python launch.py \\
  --listen \\
  --skip-python-version-check \\
  --skip-install \\
  --skip-version-check \\
  --enable-insecure-extension-access \\
  --always-gpu
EOF

chmod +x "$LAUNCH_SCRIPT"
success "Launch script created at $LAUNCH_SCRIPT"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
section "Installation Complete"

echo ""
echo -e "${GREEN}${BOLD}Forge has been successfully installed!${NC}"
echo ""
echo "  Installation directory : $FORGE_DIR"
echo "  Virtual environment    : $VENV_DIR"
echo "  Launch script          : $LAUNCH_SCRIPT"
echo "  Web UI URL             : http://127.0.0.1:7860"
echo ""
echo -e "${BOLD}To start Forge:${NC}"
echo "  ~/start_forge.sh"
echo ""
echo -e "${BOLD}Notes:${NC}"
echo "  - ComfyUI continues to run independently on its own port (default 8188)"
echo "  - Both can run simultaneously — they share model files via symlinks"
echo "  - 'No API found' popups in the UI are cosmetic and do not affect generation"
echo "  - For best uncensored results, download Pony XL or Illustrious XL from CivitAI"
echo "    and place them in your ComfyUI checkpoints folder"
echo ""

if ask_yes_no "Launch Forge now?" "y"; then
    exec "$LAUNCH_SCRIPT"
fi
