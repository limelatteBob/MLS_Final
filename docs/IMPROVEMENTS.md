# 分类质量与叶片映射改进记录

日期：2026-11-10。旧数据、旧模型、隐私账本和预实验结果均保留；本次新增数据清单与运行目录。

## 已定位的原因

1. **优化不足。** 原默认 SGD 学习率 0.03、每轮 20 步、耐心值 5，模型通常第 6 轮即早停，仍处在单类别预测阶段。已有集中式 Adam 长预算对照的测试 macro-F1 为 0.3545，说明问题并非数据完全不可学习。原短预算 DP 与 DP＋SecAgg+ 对照质量相同，尚无证据把主要质量问题归因于安全聚合。
2. **小客户端与隐私噪声叠加。** 原 12 客户端样本数为 83、146、29、151、91、141、178、17、137、51、75、104。最小客户端 Poisson 采样率为 0.5；按 50×20=1,000 步预留 epsilon=8 的噪声倍率约为 10.30。只运行少量轮次时，这种保守配置耗费了精度，却没有充分利用预定训练预算。
3. **类别和资源覆盖不均。** 按“每类相同组数”抽样不等于相同图片数，原训练类别数量为 48–192。加载 CUDA 版 PyTorch 的 CPU 进程内存超过 512 MiB，使四个摄像头客户端不可行，409/1,203 张训练图像没有参与联邦优化。
4. **测试样本被过度删除。** Target Spot 的明确命名别名恢复后，源训练侧有 568 张图片与源测试侧共享已知叶片。旧逻辑删除整个冲突组件，开发测试集中该类仅余 4 张，分类指标容易波动。新逻辑在抽样前处理源级冲突，并在下载后重复审计；保留测试侧，删除相关训练侧。
5. **映射缺失来自源元数据覆盖不足。** 官方映射由 CSV 的文件名与叶片编号汇总，不是每张图都有可验证身份。官方加载器对缺失项也有文件名回退；回退不能证明真实叶片隔离。[原始汇总脚本](https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/leaf_grouping/aggregate_map.py)、[官方数据加载代码](https://huggingface.co/datasets/mohanty/PlantVillage/blob/main/plant_village.py)。

| 类别 | 源图片数 | 未匹配数 | 主要原因 |
|---|---:|---:|---|
| Bacterial spot | 2,127 | 0 | 已覆盖 |
| Early blight | 1,000 | 1 | 映射键带 conflicted copy 注释，原图文件名无注释 |
| Late blight | 1,909 | 989 | GHLB 等来源批次缺失 |
| Leaf Mold | 952 | 0 | 已覆盖 |
| Septoria | 1,771 | 771 | JR_Sept.L.S、Keller.St_CG 批次缺失 |
| Spider mites | 1,676 | 0 | 已覆盖 |
| Target Spot | 1,404 | 0 | 显式 Com.G_TgS_FL → Com.G_FL 别名恢复 |
| Yellow Leaf Curl | 5,357 | 2,619 | UF.GRC_YLCV_Lab 批次缺失 |
| Mosaic virus | 373 | 373 | 整类没有官方映射 |
| Healthy | 1,591 | 592 | 主要是 GH_HL Leaf 批次缺失 |

上述数量来自固定源版本的全量番茄元数据，不是开发子集。逐前缀统计见 `artifacts/mapping_audit_v2.json`。没有据病害标签、相似图像或相邻编号猜测物理叶片身份。文件名中明确的 `Leaf 349`/`Leaf 349.1` 归入同一代理组，但仍标记 `leaf_verified=false`；严格模式仍会排除这些图片，因此仍不能覆盖完整十类。

新版保留 5,563 张已匹配、1,793 张未匹配图片；未匹配图片构成 1,703 个代理组，其中 79 组包含多张图片。扩大数据和修正类别抽样不会凭空提高官方映射覆盖。已输出 `artifacts/unmapped_leaf_review.csv`，列出全部 5,345 条源级缺失记录及待填写的叶片 ID、证据出处。真正补齐身份仍需原作者采集记录或有依据的人工核验；该模板不会被自动当作可信映射导入。

## 已实施的改动

- 新清单 `data/manifest_v2.json` 按每类训练源约 500 张、测试源最多约 300 张取完整组；组边界允许略超预算。小类使用实际可用图片。保留旧 `manifest.json`，两份清单的测试集合不同，不能把其分数差直接解释成模型算法收益。
- 新清单训练/验证/公共/测试数量为 **3,851 / 468 / 491 / 2,546**。训练各类约 383–404 张，花叶病毒因总量小为 245 张；Target Spot 测试 288 张。新版最小客户端为 73 张。
- 公共池预训练使用持久 AdamW、旋转/镜像增强、模型内输入归一化；训练仅访问 public，模型选择仅访问 validation。提供公共模型基线。
- 私有阶段冻结卷积特征，只对分类头执行原有 Opacus 隐私机制。最多 30 轮×5 步，最早第 20 轮触发早停；逐客户端重新校准噪声，epsilon 上限仍为 8。保存公共初始化为候选最佳模型，微调未改进时不发布更差的验证模型。
- 新配置使用 CPU 专用 PyTorch 的冻结特征训练实测，包含完整客户端依赖，并加 10% 内存余量；恢复摄像头在当前主机仿真中的可行性。仍上传完整同构模型，不宣称减少训练通信。
- 训练配置 `evaluate_test=false`，避免调参流程自动计算测试指标。`evaluate-run` 只对完成并匹配配置的运行生成独立测试结果，同时评估公共模型；不改动预算或训练配置。
- 新增分组冲突、类名精确匹配、未验证代理组、冻结特征的真实 DP 优化、公共初始化一致性、验证阶段不触碰测试指标等测试。

## 已完成的原划分对照

| 原数据划分上的模型 | 验证 macro-F1 | 测试 macro-F1 | 解释 |
|---|---:|---:|---|
| 原默认完整系统 `runs/main` | 0.0268 | 0.0230 | 原模型、原预算与早停 |
| 公共预训练＋隐私分类头流程 `runs/improved_legacy` | 0.5516 | 0.5688 | 最佳轮次为 0；收益来自公共预训练，私有微调未提高验证质量 |

该对照使用相同旧划分；它同时改变初始化、优化、增强和训练预算，不能分离各因素因果贡献。旧测试集合的 Target Spot 覆盖不足仍限制结论。新版对照采用共享公共初始化和三个联邦随机种子，并加入无隐私微调参考；其结果由 `scripts/summarize_improvements.py` 汇总。

新版三个隐私运行已全部完成，测试 macro-F1 为 **0.7161 ± 0.0004**，共享公共模型基线为 **0.7163**。两个种子和无隐私微调对照均保留第 0 轮公共模型；seed42 的验证 F1 有微小提升，测试却略降。因此不能宣称当前联邦更新提升分类质量，也不能把很小的跨种子标准差当作完整系统高度稳定的证明。三个种子只改变联邦阶段的随机性。12 个客户端均实际参与，单次运行最大 epsilon 不超过 **6.758**。

seed42 测试准确率为 **0.7223**。较弱类别是早疫病（召回 **0.5256**）和晚疫病（**0.6026**），健康叶片召回 **0.9402**。无隐私对照也没有超过公共初始化，说明新版质量瓶颈不能只归因于 DP 噪声；非 IID 条件下的更新偏移、分类头有限的适配能力以及公共特征泛化能力是下一步要验证的假设，不是此次实验已经证明的因果结论。后续应在新的、预先冻结的验证方案下比较局部解冻、公共数据回放和更长的优化预算，不能用本次测试结果继续择优调参。

逐运行结果、类别混淆图和实际压缩指标见 [实测结果](IMPROVEMENT_RESULTS.md)。原默认 5-epoch 学生模型在新版验证集 F1 仅 **0.1464**，其注册表保留于 `artifacts/student5_registry.json`；它未通过质量门槛，不会被分配给设备。已增加显式蒸馏预算、公共池增强、持久 AdamW 和验证早停，并保存完整学习曲线，避免学生模型同样因训练不足而失效。

120-epoch 蒸馏完成后，学生 FP32 最佳验证 F1 为 **0.7379**，学生 INT8 验证/测试 F1 为 **0.7305 / 0.6683**。质量已经改善，但相对教师的验证下降仍为 **4.08 个百分点**，故继续排除出自动分配，没有降低门槛来使结果达标。教师 INT8 压缩 **3.44 倍**，验证 F1 为 **0.7606**、测试 F1 为 **0.7095**，验证下降 **1.06 个百分点**，满足当前压缩质量目标。训练预算扩大、增强和优化器一起改变，不能将学生提升全部归因于单一因素。

## 硬件证据与局限

CPU 专用 PyTorch 2.14.0+cpu，导入 Flower、Opacus 和实验模块后，冻结特征的批量 8/16/32 训练峰值分别约 **396.5 / 431.9 / 482.2 MiB**。包含六模拟客户端、真实图片、账本与 SecAgg+ 的完整轮次探测峰值 **430.2 MiB**。记录为 `artifacts/profile_cpu_head.json` 与 `artifacts/cpu_round_probe.json`。这些是 Windows 主机环境的测量，并非真实摄像头部署证明；10% 余量也不是部署内存上界的形式保证。

公共池扩大和预训练都增加开发成本，并依赖代表性公开标注数据。少数类、未知叶片身份、受控背景、固定公开噪声种子及单进程仿真边界仍存在。多个研究运行的 epsilon 不应被误认为共同满足一个 epsilon=8；在真实私有数据上发布多次运行需另做跨运行组合。

## 复现

```powershell
.\.venv\Scripts\python -m fedcrop prepare --root data --policy audited --train-images-per-class 500 --test-images-per-class 300 --preserve-test --manifest-name manifest_v2.json
.\.venv\Scripts\python scripts/audit_mapping.py
.\.venv\Scripts\python -m fedcrop warmup --config configs/improved.json --output runs/public_v2 --epochs 120
.\.venv_cpu\Scripts\python scripts/profile_cpu_client.py
.\.venv_cpu\Scripts\python scripts/probe_cpu_round.py
.\.venv\Scripts\python scripts/run_improved_study.py
.\.venv\Scripts\python -m fedcrop export runs/improved_seed42 --distill-epochs 120 --distill-augment
.\.venv\Scripts\python scripts/summarize_improvements.py
.\.venv\Scripts\python -m pytest -q
```

已有初始化或运行目录不会被覆盖；需要重做时复制配置并指定新的输出路径。CPU 环境创建与安装命令见 README。本次结果不能替代课程所需的完整正式消融与真实边缘设备实验。
