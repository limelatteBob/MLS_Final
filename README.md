# FedCrop

面向异构边缘设备的番茄叶片病害联邦分类研究原型。项目包含真实 PyTorch 本地训练、Opacus 图像记录级 DP-SGD、Flower SecAgg+ 安全聚合、设备与网络仿真、ONNX 静态量化、公开数据蒸馏，以及本地监控界面。

本项目是单机综合仿真。客户端执行真实优化和密码协议，但共享一个进程，不构成操作系统隔离；设备延迟与能耗分别是主机校准的预测和代理指标。性能目标需要实验验证。

已完成端到端运行、33 项自动化测试和原 21 次三种子短预算预实验。新增公共预训练、隐私分类头微调、CPU 专用环境测量以及保留测试侧的分组修正，原因和结果见 [改进记录](docs/IMPROVEMENTS.md)，新增对照见 [实测结果](docs/IMPROVEMENT_RESULTS.md)。原始结果见 [预实验报告](docs/REPORT.md)。完整正式实验及 15% 达标时间目标尚未验证。

## 推荐的改进流程

当前改进配置为 `configs/improved.json`，保留 `main.json` 作为历史基线。新版数据清单、公共初始化和 CPU 配置实测已另存。训练只计算验证指标，配置冻结后再显式评估测试：

```powershell
.\.venv\Scripts\python -m fedcrop train --config configs/improved.json --output runs/improved_new
.\.venv\Scripts\python -m fedcrop evaluate-run runs/improved_new
.\.venv\Scripts\python -m fedcrop export runs/improved_new --distill-epochs 120 --distill-augment
.\.venv\Scripts\python scripts/summarize_improvements.py
```

从数据开始复现的完整命令见 [改进记录](docs/IMPROVEMENTS.md)。`best_round=0` 表示公共模型未被私有微调超越，报告必须保留这个基线。三个新版联邦种子共享同一个公共初始化，衡量的是给定初始化条件下的联邦稳定性。

CPU 专用测量环境与 GPU 训练环境分开，避免把 CUDA 运行时内存当成边缘 CPU 必需内存：

```powershell
python -m venv --system-site-packages .venv_cpu
.\.venv_cpu\Scripts\python -m pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu --no-deps
.\.venv_cpu\Scripts\python -m pip install opacus==1.6.0 flwr==1.43.0 psutil==7.2.2
.\.venv_cpu\Scripts\python scripts/profile_cpu_client.py
.\.venv_cpu\Scripts\python scripts/probe_cpu_round.py
```

当前 CPU 环境复用了本机已有基础依赖，实际版本见 `artifacts/cpu_environment.txt`。其他机器需先补齐 NumPy、Pillow、Requests 等依赖；CPU 实测完成后应冻结 profile，不要在已有运行中途重测覆盖。

## 快速运行

在项目根目录打开 PowerShell。当前工作区已经建立 `.venv`，复用了本机可用的 CUDA PyTorch。

```powershell
.\.venv\Scripts\python.exe -m fedcrop doctor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m fedcrop train --config configs/smoke.json
.\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.address 127.0.0.1
```

训练输出目录已有结果时不会覆盖。重新做独立实验请加 `--output runs/smoke_new`；恢复相同实验使用 `--resume`，配置和数据必须相同。CLI 无需激活虚拟环境。

运行目录由操作系统文件锁保证单写入者。运行时异常默认最多自动恢复一次（可在配置中将 `max_retries` 设为 0—3）；恢复从已提交检查点继续，隐私账本始终保留已预留步骤。配置错误、数据校验失败和缺失账本不会被自动重置。

已有完整训练结果时，单独重新导出和评估部署模型：

```powershell
.\.venv\Scripts\python.exe -m fedcrop export runs/smoke
.\.venv\Scripts\python.exe -m fedcrop predict runs/smoke path\to\leaf.jpg --client 0
```

## 环境与数据准备

最低 Python 3.11。当前验证环境为 Windows、Python 3.11.15、PyTorch 2.14.0+cu130、RTX 5060 Laptop（约 8 GB 显存）、约 32 GB 系统内存。环境依赖见 `pyproject.toml`、`requirements-lock.txt` 和 `artifacts/environment.json`。

在已有可用 CUDA PyTorch 的同类机器上：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fedcrop doctor
```

独立环境可使用 `python -m venv .venv`，随后根据目标机器安装支持其 GPU 的 PyTorch，再安装本项目依赖。CPU 也可运行，使用配置 `"device": "cpu"`。精确环境复现请使用锁定文件，并为其中 CUDA PyTorch wheel 提供相应的 PyTorch 官方 wheel 索引。不同 OS/CPU 的测量不能直接混用。

数据下载命令：

```powershell
# 较小但真实的十类开发子集，每类、每个原始 split 最多取 60 个组
.\.venv\Scripts\python.exe -m fedcrop prepare --max-groups-per-class 60 --policy audited

# 全量番茄数据；正式实验前冻结数据，不要在已有运行中途修改 manifest
.\.venv\Scripts\python.exe -m fedcrop prepare --root data_full --policy audited

# 只保留能匹配官方叶片映射的记录；目前不能覆盖全部十类
.\.venv\Scripts\python.exe -m fedcrop prepare --root data_strict --max-groups-per-class 60 --policy strict

.\.venv\Scripts\python.exe -m fedcrop profile --output artifacts/profile.json
```

下载脚本固定来源提交版本、保留原始文件名和训练/测试边界；划分验证集与公共池时使用完整组。SHA256 完全重复及 dHash 距离不超过 2 的近重复会并入同组；历史默认隔离整个跨边界组件，新版 `--preserve-test` 保留测试侧、移除训练侧。dHash 是审计启发式，不能保证识别全部近重复。

**实际核验发现官方 `leaf-map.json` 不包含番茄花叶病毒类别，且其他类别也存在缺失。**`audited` 模式将缺失记录标为 `leaf_verified=false`，使用规范化文件名加重复审计构建代理组，不声称完整物理叶片隔离；`strict` 模式排除这些记录。Target Spot 的 `Com.G_TgS_FL` 与映射中的 `Com.G_FL` 采用显式别名映射。数据清单包含验证覆盖、缺失路径和隔离路径。`configs/main.json` 显式允许审计模式；设置 `allow_unverified_groups=false` 会拒绝不完整映射的数据。

已有十类子集时，`python scripts/verified_subset.py` 可直接派生 `data/manifest_verified.json`，保留原来的四类 split，无需重新下载。`configs/strict_smoke.json` 验证严格子集运行。该子集缺少花叶病毒类；网络仍是十类输出，Macro-F1 固定按十类计算，因此须注明其类别覆盖，不能冒充完整十类评估。

## 实验与产物

```powershell
# 主配置：12 客户端、每轮 6 个、重建阈值 4、最多 50 轮
.\.venv\Scripts\python.exe -m fedcrop train --config configs/main.json

# 先生成三种子主比较配置，检查后执行同一命令并添加 --execute
.\.venv\Scripts\python.exe -m fedcrop suite --output runs/suite

# 加入单因素压力测试；不会运行全组合扫描
.\.venv\Scripts\python.exe -m fedcrop suite --output runs/stress --stress

.\.venv\Scripts\python.exe -m fedcrop report runs --output artifacts/analysis
```

`configs/pilot.json` 为短预算预实验；`configs/quality_pilot.json` 是更长预算的非隐私集中式优化诊断，显式选择 Adam 和最短训练轮次，不能与默认 SGD 配置混为同一组对照。默认方案保持 SGD 与五轮验证耐心；正式实验参数应根据验证集诊断统一冻结，禁止用测试结果挑选参数。

每个运行目录保存配置、主机信息、分区、轮次记录、检查点、最佳模型、隐私账本和结果摘要。导出后增加模型注册表、量化与蒸馏模型、独立进程 CPU 推理测量和设备分配结果。监控仅读取允许发布的记录，不读取私有账本、训练图片、梯度或个体更新。

三种子主比较包含 local-only、集中式、均匀无隐私 FedAvg、均匀 DP、均匀 DP+SecAgg+、速度优先和完整公平调度系统。集中式／本地训练是非隐私研究参照，不能用于证明隐私系统的保护能力。本地基线在各自合成农场测试集上评估，集中式与联邦模型同时保留全局测试指标。

正式比较前，应在独立 pilot 的验证集上确定并冻结 `target_f1`。默认 `null` 表示尚未设置，报告不会把它写成达标。三种子数量不足时标准差保留为空，不伪造统计结果。研究结果应基于相同数据清单、隐私设置与预算；同一 seed 的客户端可用性、掉线和网络抖动由调度前生成的共同轨迹确定。

## 动态部署

默认在三个导出变体中选择满足内存、预测 p95 延迟且验证 F1 相对教师下降不超过 0.03 的最高质量模型。没有可行模型则暂停推理。INT8 不保证在所有 CPU 上更快。

```powershell
# 修改 assignments.json 中 devices 数组后，另存为 devices_changed.json
.\.venv\Scripts\python.exe -m fedcrop assign runs/smoke --devices devices_changed.json --latency-budget-ms 50
```

该命令重新评估资源，记录分配变化和模拟模型下发字节。`predict` 使用当前分配的实际 ONNX 文件执行分类。资源变化文件只需包含设备字段，可参考 `assignments.json`；无真实设备时不声称完成设备端网络部署。

运行 `python scripts/replay_resources.py runs/smoke` 可重放正常资源、延迟收紧、内存压力与恢复四个场景，对比固定分配的预算违例和动态分配的暂停／切换；末尾恢复正常资源分配。

## 已知边界

- 512 MB 摄像头档无法容纳本机测得的完整 PyTorch CPU DP 训练进程，因此可作为推理设备，但不会强行纳入训练。敏感性实验可改变设备资源配置；不能把主机 GPU 显存当作边缘 RAM。
- Flower 1.43 在 setup/share-keys 阶段掉线时有缺失秘密份额的恢复边界；本实现提前中止该轮，不回退到明文。collect-masked-vectors 阶段允许两客户端掉线，三客户端掉线中止；unmask 阶段按真实协议恢复已上传贡献。
- CPU 推理在独立的 ONNX Runtime 进程测量，避免把训练进程已加载的 PyTorch/CUDA 内存计入部署模型。延迟默认不包含图片预处理。
- 模拟截止时间带有抖动余量，电池预留会在训练前限制可执行步数。密码运算在不同真实边缘芯片上的耗时尚未单独校准，不能把模拟时间视为实机时延保证。
- 固定随机种子的 DP 运行用于可复现实验，不能当作生产级安全随机性。详情见 [隐私说明](docs/PRIVACY.md)。
- PDF 实际包含课程主页与项目概述，没有详细评分量表；课程依据和交付映射见 [课程核对](docs/COURSE_REQUIREMENTS.md)。

## 代码导航

| 模块 | 职责 |
|---|---|
| `fedcrop/data.py` | 数据下载、分组与重复审计、客户端分区 |
| `fedcrop/training.py`、`privacy.py` | DP-SGD、物理微批次和持久隐私记账 |
| `fedcrop/federation.py` | Flower ClientApp、SecAgg+ 和序列化单机消息传输 |
| `fedcrop/resources.py`、`experiment.py` | 主机测量、资源调度、检查点与实验运行 |
| `fedcrop/deployment.py`、`benchmark.py` | 蒸馏、ONNX 导出、INT8、实际推理和部署分配 |
| `fedcrop/analysis.py`、`dashboard.py` | 对照实验、统计图表与监控 |

数据与算法出处见 [参考资料](docs/REFERENCES.md)。复用工具、生成式工具辅助和作者自己的设计／分析应在最终课程提交中如实注明。
