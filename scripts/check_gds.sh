#!/bin/bash
# check_gds.sh — Verify GPUDirect Storage Readiness
#
# This script checks all prerequisites for cuFile/GPUDirect Storage.
# Run before developing or deploying any cuFile application.
#
# Usage:
#   bash check_gds.sh [nvme_mountpoint]
#
# If no mountpoint is specified, checks platform-level requirements only.

set -o pipefail  # catch failures inside pipelines

MOUNTPOINT="${1:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0
warn_count=0

pass() {
    echo -e "  [${GREEN}PASS${NC}] $1"
    ((pass_count++))
}

fail() {
    echo -e "  [${RED}FAIL${NC}] $1"
    echo -e "         ${YELLOW}→ $2${NC}"
    ((fail_count++))
}

warn() {
    echo -e "  [${YELLOW}WARN${NC}] $1"
    echo -e "         → $2"
    ((warn_count++))
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
}

echo "============================================"
echo "  GPUDirect Storage (GDS) Readiness Check"
echo "============================================"
echo ""

# ─── Section 1: GPU ─────────────────────────────────────────

echo "─── GPU ───"

if check_cmd nvidia-smi; then
    pass "nvidia-smi available"

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    echo "       GPU: ${GPU_NAME:-unknown}"

    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
    CC_MAJOR=$(echo "${CC:-0}" | cut -d. -f1)
    if [ "$CC_MAJOR" -ge 6 ]; then
        pass "Compute Capability $CC (≥ 6.0 required)"
    else
        fail "Compute Capability $CC (< 6.0)" \
             "GDS requires Pascal (SM 6.0) or newer GPU"
    fi

    DRV_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    DRV_MAJOR=$(echo "${DRV_VER:-0}" | cut -d. -f1)
    if [ "$DRV_MAJOR" -ge 470 ]; then
        pass "Driver version $DRV_VER (≥ 470.57 required)"
    else
        fail "Driver version $DRV_VER (< 470.57)" \
             "Upgrade to NVIDIA driver R525+ for best results"
    fi
else
    fail "nvidia-smi not found" \
         "Install NVIDIA driver and CUDA toolkit"
fi

echo ""

# ─── Section 2: CUDA Toolkit ────────────────────────────────

echo "─── CUDA Toolkit ───"

# Find nvcc: search CUDA toolkit paths (not just PATH)
NVCC_BIN=""
for cuda_base in /usr/local/cuda-* /usr/local/cuda /opt/cuda; do
    if [ -x "$cuda_base/bin/nvcc" ]; then
        NVCC_BIN="$cuda_base/bin/nvcc"
        break
    fi
done
# Also check PATH as fallback
if [ -z "$NVCC_BIN" ]; then
    NVCC_BIN=$(command -v nvcc 2>/dev/null || true)
fi

if [ -n "$NVCC_BIN" ] && [ -x "$NVCC_BIN" ]; then
    NVCC_VER=$("$NVCC_BIN" --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || true)
    if [ -n "$NVCC_VER" ]; then
        pass "nvcc found ($NVCC_BIN, version $NVCC_VER)"
        CUDA_MAJOR=$(echo "${NVCC_VER:-0}" | cut -d. -f1)
        if [ "${CUDA_MAJOR:-0}" -ge 12 ]; then
            pass "CUDA ${CUDA_MAJOR}.x (≥ 11.4 required)"
        elif [ "${CUDA_MAJOR:-0}" -ge 11 ]; then
            warn "CUDA ${CUDA_MAJOR}.x (≥ 11.4 required, 12+ recommended)"
        elif [ "${CUDA_MAJOR:-0}" -gt 0 ]; then
            fail "CUDA ${CUDA_MAJOR}.x (< 11.4)" \
                 "Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads"
        fi
    else
        warn "nvcc found at $NVCC_BIN but --version failed"
    fi
else
    fail "nvcc not found in PATH or /usr/local/cuda*" \
         "Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
fi

# Check for libcufile: search all CUDA paths
CUFILE_LIB=$(find /usr/local/cuda* /opt/cuda* -name "libcufile*" 2>/dev/null | head -1 || true)
if [ -n "$CUFILE_LIB" ]; then
    pass "libcufile found: $CUFILE_LIB"
else
    warn "libcufile not found" \
         "libcufile is bundled with CUDA Toolkit. Check CUDA installation."
fi

echo ""

# ─── Section 3: nvidia-fs Kernel Module ─────────────────────

echo "─── nvidia-fs Kernel Module ───"

if lsmod | grep -q nvidia_fs; then
    pass "nvidia-fs kernel module loaded"

    if [ -r /proc/driver/nvidia-fs/gds ]; then
        GDS_STATUS=$(cat /proc/driver/nvidia-fs/gds 2>/dev/null || echo "unknown")
        echo "       GDS status: $GDS_STATUS"
    fi
else
    fail "nvidia-fs not loaded" \
         "Run: sudo modprobe nvidia-fs"
fi

echo ""

# ─── Section 4: NVMe Devices ────────────────────────────────

echo "─── NVMe Devices ───"

NVME_COUNT=$(lspci 2>/dev/null | grep -ci "Non-Volatile" || echo 0)
if [ "$NVME_COUNT" -gt 0 ]; then
    pass "$NVME_COUNT NVMe device(s) found"

    lspci 2>/dev/null | grep "Non-Volatile" | while read -r line; do
        BDF=$(echo "$line" | awk '{print $1}')
        echo "       $BDF: $line"

        # Check PCIe link speed
        if [ -e "/sys/bus/pci/devices/0000:$BDF" ]; then
            SPEED=$(cat "/sys/bus/pci/devices/0000:$BDF/current_link_speed" 2>/dev/null || echo "unknown")
            WIDTH=$(cat "/sys/bus/pci/devices/0000:$BDF/current_link_width" 2>/dev/null || echo "unknown")
            echo "             PCIe $SPEED x$WIDTH"
        fi
    done || true
else
    fail "No NVMe devices found via lspci" \
         "NVMe PCIe SSD required for GDS"
fi

echo ""

# ─── Section 4a: PCIe Topology & ACS ─────────────────────────

echo "─── PCIe Topology & ACS ───"

# Check PCIe topology: GPU and NVMe on same root complex
GPU_BDF_LIST=$(lspci 2>/dev/null | grep -i "VGA\|3D\|Display" | grep -i nvidia | awk '{print $1}' || true)
NVME_BDF_LIST=$(lspci 2>/dev/null | grep "Non-Volatile" | awk '{print $1}' || true)

if [ -n "$GPU_BDF_LIST" ] && [ -n "$NVME_BDF_LIST" ]; then
    echo "       GPUs found:  $(echo $GPU_BDF_LIST | wc -w | tr -d ' ')"
    echo "       NVMe found:  $(echo $NVME_BDF_LIST | wc -w | tr -d ' ')"

    # For each GPU-NVMe pair, check if same PCIe root complex
    for gpu_bdf in $GPU_BDF_LIST; do
        GPU_ROOT=$(lspci -t -v 2>/dev/null | grep -B10 "$gpu_bdf" | grep "Root Complex" | tail -1 || echo "")
        for nvme_bdf in $NVME_BDF_LIST; do
            NVME_ROOT=$(lspci -t -v 2>/dev/null | grep -B10 "$nvme_bdf" | grep "Root Complex" | tail -1 || echo "")
            if [ "$GPU_ROOT" = "$NVME_ROOT" ] && [ -n "$GPU_ROOT" ]; then
                pass "GPU $gpu_bdf and NVMe $nvme_bdf: same root complex ✓"
            else
                warn "GPU $gpu_bdf and NVMe $nvme_bdf: may be on DIFFERENT root complexes" \
                     "GDS requires GPU + NVMe on the same PCIe root complex. Verify with: lspci -t -v"
            fi
        done
    done

    # ─── ACS (Access Control Services) Check ───
    echo ""
    echo "       --- ACS (Access Control Services) Status ---"
    echo "       ACS must be DISABLED for PCIe P2P between GPU and NVMe."
    echo "       If ACS is enabled (+) for any GPU or NVMe, GDS will NOT work."
    echo ""

    ACS_ISSUE_FOUND=0
    # ACS check requires root. Detect permission once; skip if unavailable.
    FIRST_BDF=$(echo "$GPU_BDF_LIST" | awk '{print $1}')
    HAVE_ACS_ACCESS=0
    if [ -n "$FIRST_BDF" ] && [ -r "/sys/bus/pci/devices/0000:$FIRST_BDF/acs_ctrl" ]; then
        HAVE_ACS_ACCESS=1
    fi
    if [ "$HAVE_ACS_ACCESS" -eq 0 ]; then
        warn "ACS check requires root — skipping" \
             "Re-run with: sudo bash $(basename "$0") $MOUNTPOINT"
    else
        for bdf in $GPU_BDF_LIST $NVME_BDF_LIST; do
            ACS_PATH="/sys/bus/pci/devices/0000:$bdf/acs_ctrl"
            if [ -r "$ACS_PATH" ]; then
                ACS_VAL=$(cat "$ACS_PATH" 2>/dev/null || echo "0")
                if [ "${ACS_VAL:-0}" -gt 0 ] 2>/dev/null; then
                    fail "ACS ENABLED on $bdf (acs_ctrl=$ACS_VAL)" \
                         "Disable ACS in BIOS or with kernel param: pci=disable_acs"
                    ACS_ISSUE_FOUND=1
                else
                    pass "ACS disabled on $bdf"
                fi
            fi
        done
    fi

    if [ "$ACS_ISSUE_FOUND" -eq 1 ]; then
        echo ""
        echo "       ┌─────────────────────────────────────────────────────┐"
        echo "       │  ⚠️  ACS is ENABLED — GDS P2P DMA WILL BE BLOCKED   │"
        echo "       │                                                     │"
        echo "       │  To disable ACS:                                    │"
        echo "       │  1. BIOS: Check for 'ACS Control' or 'PCIe ACS'     │"
        echo "       │  2. Kernel: Add 'pci=disable_acs' to kernel cmdline │"
        echo "       │     sudo grubby --update-kernel=ALL \\               │"
        echo "       │       --args='pci=disable_acs'                     │"
        echo "       │  3. Reboot after changes                            │"
        echo "       └─────────────────────────────────────────────────────┘"
    fi

elif [ -z "$GPU_BDF_LIST" ]; then
    warn "No NVIDIA GPU found via lspci" \
         "Skipping PCIe topology check"
elif [ -z "$NVME_BDF_LIST" ]; then
    warn "No NVMe devices found via lspci" \
         "Skipping PCIe topology check"
fi

echo ""

# ─── Section 5: GDS Tools ───────────────────────────────────

echo "─── GDS Tools ───"

if check_cmd gdscheck; then
    pass "gdscheck available"
else
    warn "gdscheck not found" \
         "Install cuda-toolkit or nvidia-gds package"
fi

if check_cmd gdsio; then
    pass "gdsio available"
else
    warn "gdsio not found" \
         "Install cuda-toolkit or nvidia-gds package"
fi

echo ""

# ─── Section 6: Filesystem Check (if mountpoint provided) ───

if [ -n "$MOUNTPOINT" ]; then
    echo "─── Filesystem: $MOUNTPOINT ───"

    if [ -d "$MOUNTPOINT" ]; then
        FS_TYPE=$(df -T "$MOUNTPOINT" 2>/dev/null | tail -1 | awk '{print $2}')
        echo "       Type: $FS_TYPE"

        case "$FS_TYPE" in
            ext4|xfs|gpfs)
                pass "Filesystem type: $FS_TYPE (GDS supported)"
                ;;
            nfs*|cifs|smb*|tmpfs|fuse*)
                fail "Filesystem type: $FS_TYPE (NOT GDS-compatible)" \
                     "GDS requires ext4, xfs, or GPFS on local NVMe storage"
                ;;
            *)
                warn "Filesystem type: $FS_TYPE (unknown GDS compatibility)" \
                     "Verify with: gdscheck -f $MOUNTPOINT"
                ;;
        esac

        # Check if O_DIRECT works
        TESTFILE="$MOUNTPOINT/.gds_test_$$"
        if python3 -c "
import os
try:
    fd = os.open('$TESTFILE', os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
    os.close(fd)
    os.unlink('$TESTFILE')
    print('OK')
except Exception as e:
    print('FAIL: ' + str(e))
" 2>/dev/null | grep -q "OK"; then
            pass "O_DIRECT supported on $MOUNTPOINT"
        else
            fail "O_DIRECT NOT supported on $MOUNTPOINT" \
                 "GDS requires O_DIRECT. Check filesystem mount options."
        fi

        # Run gdscheck if available
        if check_cmd gdscheck; then
            echo ""
            echo "       Running gdscheck -f $MOUNTPOINT..."
            gdscheck -f "$MOUNTPOINT" 2>&1 | sed 's/^/       /' || true
        fi
    else
        fail "Mount point $MOUNTPOINT does not exist" \
             "Specify a valid NVMe mount point"
    fi

    echo ""
fi

# ─── Section 7: cufile.json ─────────────────────────────────

echo "─── Configuration ───"

if [ -f /etc/cufile.json ]; then
    pass "/etc/cufile.json exists"

    # Check key settings (cufile.json may use non-standard JSON with comments;
    # use grep as a tolerant parser rather than strict python3 -m json.tool)
    if grep -q '"enable_compat_mode"' /etc/cufile.json 2>/dev/null; then
        if grep '"enable_compat_mode"' /etc/cufile.json 2>/dev/null | grep -q 'true\|1'; then
            echo "       Compat mode: ENABLED"
        elif grep '"enable_compat_mode"' /etc/cufile.json 2>/dev/null | grep -q 'false\|0'; then
            warn "Compat mode is DISABLED (hard errors on GDS fallback)" \
                 "Set enable_compat_mode: true in /etc/cufile.json for production"
        fi
    fi

    # Try strict JSON validation as a bonus check (non-fatal)
    if python3 -m json.tool /etc/cufile.json > /dev/null 2>&1; then
        pass "cufile.json is valid strict JSON"
    else
        warn "cufile.json is not valid strict JSON (may contain comments or trailing commas)" \
             "This is usually fine — cuFile accepts non-standard JSON. Verify with gdscheck."
    fi
else
    warn "/etc/cufile.json not found" \
         "Create it or set CUFILE_CONFIG env var. See references/configuration.md"
fi

echo ""

# ─── Summary ────────────────────────────────────────────────

echo "============================================"
echo "  Summary"
echo "============================================"
echo -e "  Passed:  ${GREEN}$pass_count${NC}"
echo -e "  Warnings: ${YELLOW}$warn_count${NC}"
echo -e "  Failed:  ${RED}$fail_count${NC}"
echo ""

if [ "$fail_count" -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed. GDS should be operational.${NC}"

    if [ -n "$MOUNTPOINT" ] && check_cmd gdsio; then
        echo ""
        echo "Run a quick performance test:"
        echo "  gdsio -f $MOUNTPOINT/testfile -s 1G -i 1M -x 0"
    fi
else
    echo -e "${RED}❌ $fail_count critical check(s) failed. GDS may not work.${NC}"
    echo "   Fix the FAIL items above before developing cuFile applications."
fi

echo ""
echo "For detailed hardware requirements, see:"
echo "  references/hardware-requirements.md"
