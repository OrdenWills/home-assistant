.PHONY: all install-kernel install-pip download-wheels install-pytorch \
       uninstall-pytorch verify clean help

ROCM_VERSION     := 7.2
ROCM_REL         := rocm-rel-$(ROCM_VERSION)
REPO_BASE_URL    := https://repo.radeon.com/rocm/manylinux/$(ROCM_REL)
PYTHON_VERSION   := cp312

TORCH_WHL        := torch-2.9.1+rocm$(ROCM_VERSION).0.lw.git7e1940d4-$(PYTHON_VERSION)-$(PYTHON_VERSION)-linux_x86_64.whl
TORCHVISION_WHL  := torchvision-0.24.0+rocm$(ROCM_VERSION).0.gitb919bd0c-$(PYTHON_VERSION)-$(PYTHON_VERSION)-linux_x86_64.whl
TORCHAUDIO_WHL   := torchaudio-2.9.0+rocm$(ROCM_VERSION).0.gite3c6ee2b-$(PYTHON_VERSION)-$(PYTHON_VERSION)-linux_x86_64.whl
TRITON_WHL       := triton-3.5.1+rocm$(ROCM_VERSION).0.gita272dfa8-$(PYTHON_VERSION)-$(PYTHON_VERSION)-linux_x86_64.whl

WHEELS           := $(TORCH_WHL) $(TORCHVISION_WHL) $(TORCHAUDIO_WHL) $(TRITON_WHL)
WHEEL_URLS       := $(addprefix $(REPO_BASE_URL)/,$(WHEELS))

WHEEL_DIR        := .wheels

all: install-pytorch verify ## Full install: download wheels, install, and verify

# ---------- System preparation --------------------------------------------------

install-kernel: ## Install the required OEM kernel (requires reboot)
	sudo apt update && sudo apt install -y linux-oem-24.04c
	@echo "------------------------------------------------------------"
	@echo "Reboot your system and verify the kernel with: uname -r"
	@echo "Expected: 6.14-1018 or newer"
	@echo "------------------------------------------------------------"

install-pip: ## Install / upgrade pip and wheel
	sudo apt install -y python3-pip
	pip3 install --upgrade pip wheel

# ---------- PyTorch install -----------------------------------------------------

$(WHEEL_DIR):
	mkdir -p $(WHEEL_DIR)

download-wheels: $(WHEEL_DIR) ## Download ROCm PyTorch wheels from repo.radeon.com
	@for whl in $(WHEELS); do \
		if [ ! -f "$(WHEEL_DIR)/$$whl" ]; then \
			echo "Downloading $$whl ..."; \
			wget -q --show-progress -P $(WHEEL_DIR) "$(REPO_BASE_URL)/$$whl"; \
		else \
			echo "Already downloaded: $$whl"; \
		fi; \
	done

uninstall-pytorch: ## Uninstall any existing torch packages
	pip3 uninstall -y torch torchvision triton torchaudio 2>/dev/null || true

install-pytorch: download-wheels uninstall-pytorch ## Download wheels, remove old versions, and install PyTorch for ROCm
	pip3 install $(addprefix $(WHEEL_DIR)/,$(WHEELS))

# ---------- Verification --------------------------------------------------------

verify: ## Verify the PyTorch + ROCm installation
	@echo "=== Import check ==="
	python3 -c 'import torch' 2>/dev/null && echo 'Success' || echo 'Failure'
	@echo ""
	@echo "=== GPU available ==="
	python3 -c 'import torch; print(torch.cuda.is_available())'
	@echo ""
	@echo "=== Device name ==="
	python3 -c "import torch; print('device name [0]:', torch.cuda.get_device_name(0))"
	@echo ""
	@echo "=== Environment info ==="
	python3 -m torch.utils.collect_env

# ---------- AOTriton experimental kernels ---------------------------------------

enable-aotriton: ## Print the env var needed to enable AOTriton experimental kernels
	@echo "Run the following before your PyTorch scripts:"
	@echo ""
	@echo "  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"

# ---------- Cleanup -------------------------------------------------------------

clean: ## Remove downloaded wheel files
	rm -rf $(WHEEL_DIR)

# ---------- Help ----------------------------------------------------------------

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
