# cuFile Hardware Requirements

## Overview

GPUDirect Storage has specific hardware requirements. If any component in the chain doesn't meet them, cuFile falls back to compatibility mode (CPU bounce buffer) — silently, unless you check `cuFileDriverGetProperties()`.

## GPU Requirements

### Supported Architectures

| Architecture     | SM Version | Example GPUs               | GDS Support | Notes                                       |
| ---------------- | ---------- | -------------------------- | ----------- | ------------------------------------------- |
| **Pascal**       | 6.0, 6.1   | P100, P40, GP102           | ✅ Minimum  | cuFile 1.0+. First arch with GDS support    |
| **Volta**        | 7.0        | V100                       | ✅ Full     | cuFile 1.0+. NVLink support                 |
| **Turing**       | 7.5        | T4, RTX 2080 Ti            | ✅ Full     | cuFile 1.0+                                 |
| **Ampere**       | 8.0, 8.6   | A100, A40, RTX 3090, A6000 | ✅ Full     | cuFile 1.2+. PCIe Gen4 support              |
| **Ada Lovelace** | 8.9        | RTX 4090, L40, L40S        | ✅ Full     | cuFile 1.3+. PCIe Gen4                      |
| **Hopper**       | 9.0        | H100, H800                 | ✅ Full     | cuFile 1.4+. PCIe Gen5. CUDA graph + cuFile |
| **Blackwell**    | 10.0       | B100, B200                 | ✅ Full     | cuFile 1.5+ (expected)                      |
| **Maxwell**      | 5.x        | M40, GTX 980               | ❌ None     | No GDS support                              |
| **Kepler**       | 3.x        | K80, K40                   | ❌ None     | No GDS support                              |

**Check your GPU:**

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## NVMe SSD Requirements

### Minimum Requirements

- **Interface**: PCIe NVMe (not SATA, not SAS)
- **PCIe Generation**: Gen3 x4 minimum; Gen4 x4 recommended
- **Controller**: Must support PCIe Bus Mastering and DMA
- **Queue Depth**: NVMe queue depth ≥ 32 (higher = better for batch I/O)

### Recommended NVMe SSDs for GDS

| Drive                 | Form Factor | PCIe    | GDS Performance     | Notes                             |
| --------------------- | ----------- | ------- | ------------------- | --------------------------------- |
| Samsung PM1733/1735   | HHHL / U.2  | Gen4 x8 | Excellent (28 GB/s) | Enterprise, high endurance        |
| Samsung PM9A3         | M.2 / U.2   | Gen4 x4 | Excellent (7 GB/s)  | Datacenter, balanced              |
| Intel P5800X (Optane) | U.2         | Gen4 x4 | Excellent (7 GB/s)  | Ultra-low latency, high endurance |
| Kioxia CM6            | U.2         | Gen4 x4 | Good (6.5 GB/s)     | Enterprise, high capacity         |
| WD SN840              | U.2         | Gen3 x4 | Good (3.2 GB/s)     | Enterprise, cost-effective        |
| Micron 9300 MAX       | U.2         | Gen3 x4 | Good (3.2 GB/s)     | Enterprise                        |

### NVMe Requirements for GDS

- The NVMe controller must support all standard NVM command set features
- Controller Memory Buffer (CMB) is recommended but NOT required
- The NVMe device must be on a PCIe root complex that supports P2P

**Check NVMe details:**

```bash
# List NVMe devices
lspci | grep "Non-Volatile"

# Detailed NVMe info
nvme list
nvme id-ctrl /dev/nvme0

# Check PCIe link speed/width
lspci -vv -s 17:00.0 | grep -E "LnkSta:|LnkCap:"
```

## PCIe Topology Requirements

### The Critical Rule

**The GPU and NVMe SSD MUST be on the same PCIe root complex for P2P DMA.**

```text
WORKING (same root complex):
  PCIe Root Complex 0
  ├── GPU 0 (17:00.0)
  └── NVMe 0 (65:00.0)
  ─ P2P DMA possible ✓

NOT WORKING (different root complexes):
  PCIe Root Complex 0          PCIe Root Complex 1
  ├── GPU 0 (17:00.0)          ├── GPU 1 (a1:00.0)
                                └── NVMe 0 (c0:00.0)
  ─ P2P DMA between GPU 0 and NVMe 0 NOT possible ✗
  ─ P2P DMA between GPU 1 and NVMe 0 IS possible ✓
```

### ACS (Access Control Services)

ACS must be DISABLED for PCIe P2P between GPU and NVMe devices. ACS is a security feature that blocks P2P transactions.

```bash
# Check ACS status for a PCIe device
lspci -vv -s 17:00.0 | grep "ACSCtl"

# If ACS is enabled, GDS will NOT work for this device pair
# Disable ACS in BIOS (if supported) or via kernel parameter:
#   pci=disable_acs
```

### Multi-Socket Systems

On multi-socket (NUMA) systems, each CPU socket has its own PCIe root complex:

```text
Socket 0                     Socket 1
├── PCIe RC 0                ├── PCIe RC 1
│   ├── GPU 0                │   ├── GPU 2
│   └── NVMe 0               │   └── NVMe 2
└── PCIe RC 0a               └── PCIe RC 1a
    ├── GPU 1                    └── GPU 3
    └── NVMe 1
```

GDS works within each root complex pair (GPU 0 ↔ NVMe 0, GPU 1 ↔ NVMe 1, etc.) but NOT across sockets. Check with:

```bash
# See PCIe topology
gdscheck -p

# Manual check
lspci -t -v
```

## Filesystem Requirements

| Filesystem | GDS Support | Mount Options | Notes                                     |
| ---------- | ----------- | ------------- | ----------------------------------------- |
| ext4       | ✅          | Default       | dax mount option recommended if supported |
| xfs        | ✅          | Default       | Best performance for large files          |
| GPFS       | ✅          | Default       | IBM Spectrum Scale parallel FS            |
| NFS        | ❌          | —             | Network filesystem                        |
| CIFS/SMB   | ❌          | —             | Network filesystem                        |
| tmpfs      | ❌          | —             | RAM-backed, no NVMe                       |
| ZFS        | ⚠️ Partial  | —             | Limited/tested configurations only        |

**Check filesystem:**

```bash
df -T /mnt/nvme
gdscheck -f /mnt/nvme
```

## Software Stack Requirements

| Component     | Minimum Version  | Recommended Version | Check Command                             |
| ------------- | ---------------- | ------------------- | ----------------------------------------- |
| NVIDIA Driver | 470.57.02 (R470) | 525+ (R525)         | `nvidia-smi`                              |
| CUDA Toolkit  | 11.4             | 12.0+               | `nvcc --version`                          |
| libcufile     | Bundled          | Bundled (CUDA 12.x) | `find /usr/local/cuda -name "libcufile*"` |
| nvidia-fs.ko  | Matching driver  | Matching driver     | `modinfo nvidia-fs`                       |
| Linux Kernel  | 4.18+            | 5.15+ (LTS)         | `uname -r`                                |

### nvidia-fs Kernel Module

```bash
# Check if module is loaded
lsmod | grep nvidia_fs

# Load it
sudo modprobe nvidia-fs

# Check module info
modinfo nvidia-fs

# Persistent load across reboots
echo "nvidia-fs" | sudo tee /etc/modules-load.d/nvidia-fs.conf
```

## GDS Verification Checklist

Run this checklist before developing any cuFile application:

```bash
# 1. GPU check
nvidia-smi --query-gpu=name,compute_cap --format=csv
# Must show compute capability ≥ 6.0

# 2. Driver check
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Must show ≥ 470.57

# 3. NVMe check
lspci | grep "Non-Volatile"
# Must show at least one NVMe device

# 4. PCIe topology check (same root complex?)
lspci -t -v | grep -A5 "GPU\|NVMe"

# 5. nvidia-fs module check
lsmod | grep nvidia_fs
# Must show nvidia_fs loaded

# 6. GDS platform check (NVIDIA's tool)
gdscheck -p
# All checks must pass

# 7. GDS filesystem check
gdscheck -f /mnt/nvme
# Must pass for your data mount point

# 8. Functional test
gdsio -f /mnt/nvme/test_gds -s 1G -i 1M -x 0
# Must complete without errors, should show GDS throughput

# 9. In-code verification
# Always add to your application:
CUfileDrvProps_t props;
cuFileDriverGetProperties(&props);
assert(props.is_gds_enabled && "GDS must be enabled for this workload");
```

## Common Hardware Issues

| Issue                                | Symptom                           | Resolution                                                  |
| ------------------------------------ | --------------------------------- | ----------------------------------------------------------- |
| GPU on wrong PCIe root complex       | GDS capable=false                 | Move GPU or NVMe to same PCIe switch, or use different pair |
| ACS enabled                          | P2P DMA blocked                   | Disable ACS in BIOS or kernel cmdline                       |
| NVMe is SATA (not PCIe)              | No NVMe in `lspci` output         | Replace with PCIe NVMe drive                                |
| nvidia-fs.ko not loaded              | `cuFileDriverOpen` fails          | `modprobe nvidia-fs`                                        |
| Old NVIDIA driver (< 470)            | `CU_FILE_NOT_SUPPORTED`           | Upgrade to R525+                                            |
| Filesystem not O_DIRECT capable      | GDS falls back to compat          | Use ext4/xfs, not NFS                                       |
| Shared NVMe across VMs (SR-IOV)      | GDS performance highly variable   | Dedicate NVMe or use compat mode                            |
| GPU with < 4GB BAR size (older GPUs) | Large buffer registration limited | Register smaller buffer chunks                              |

## System Validation Script

```bash
#!/bin/bash
# Quick GDS hardware validation
echo "=== GDS Hardware Check ==="

echo -n "GPU (SM ≥ 6.0): "
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
[ "$SM" -ge 6 ] && echo "PASS (SM $SM)" || echo "FAIL (SM $SM)"

echo -n "Driver (≥ 470): "
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$DRV" -ge 470 ] && echo "PASS ($DRV)" || echo "FAIL ($DRV)"

echo -n "NVMe device: "
NVME=$(lspci | grep "Non-Volatile" | wc -l)
[ "$NVME" -gt 0 ] && echo "PASS ($NVME found)" || echo "FAIL (none found)"

echo -n "nvidia-fs module: "
lsmod | grep -q nvidia_fs && echo "PASS" || echo "FAIL (not loaded)"

echo -n "gdscheck available: "
which gdscheck > /dev/null 2>&1 && echo "PASS" || echo "FAIL (install cuda-toolkit)"

echo -n "libcufile available: "
find /usr/local/cuda -name "libcufile*" 2>/dev/null | grep -q . && echo "PASS" || echo "FAIL"
```

See also:

- `driver-lifecycle.md` for using these checks in code
- `configuration.md` for cufile.json settings that interact with hardware
- `comparison-spdk.md` for alternatives when GDS hardware isn't available
