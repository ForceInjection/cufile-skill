# cuFile Programming Skill

NVIDIA cuFile / GPUDirect Storage (GDS) 高性能编程 Agent Skill。

## 这是什么？

一个面向 Claude Code 的 Agent Skill，提供 cuFile / GPUDirect Storage (GDS) 编程的完整知识体系。cuFile 是 NVIDIA 的 API，支持 NVMe SSD 与 GPU 显存之间的直接 DMA 数据传输，绕过 CPU 内存，实现最低延迟和最高吞吐。

## 涵盖内容

- **cuFile API 生命周期**：Driver → Buffer → File Handle → I/O → Cleanup
- **同步 / 异步 / 批量 I/O**：三种 I/O 模式的选择与优化
- **性能调优**：对齐要求、IO 大小优化、批量聚合、NUMA 亲和性
- **cufile.json 配置**：完整的配置参考和调优指导
- **集成模式**：双缓冲预取流水线、Checkpoint/Restore、多 GPU 多 NVMe 条带化
- **错误处理与诊断**：GDS 兼容性检测、故障排查工作流、常见性能陷阱

## 文件结构

```text
├── SKILL.md              # 主技能文件 (Claude Code 加载)
├── CLAUDE.md             # AI Agent 指导文件
├── README.md             # 本文件
├── references/           # 13 个深入参考文件（按需加载）
├── examples/             # 6 个可编译的 CUDA C 示例
└── scripts/              # GDS 环境检测脚本
```

## 前置条件

- NVIDIA GPU (Pascal SM 6.0+)
- NVIDIA 驱动 470.57.02+ (推荐 525+)
- CUDA Toolkit 12.0+
- nvidia-fs 内核模块
- 支持 GDS 的 NVMe SSD (PCIe Gen3 x4+)
- ext4 / xfs / GPFS 文件系统

## 快速验证 GDS 是否就绪

```bash
# 运行环境检测脚本
bash scripts/check_gds.sh

# 或手动验证
gdscheck -p                     # 平台检查
gdscheck -f /mnt/nvme           # 文件系统检查
gdsio -f /mnt/nvme/testfile     # 功能验证
```

## 相关技能

- `nvme-programming` — NVMe 协议基础知识
- `cuda-knowledge` — CUDA API 参考
- `cuda-samples` — CUDA 代码模式
