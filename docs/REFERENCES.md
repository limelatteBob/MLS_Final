# 数据与工具来源

- [项目原始提案](../MAIE5532_Final_Project_Proposal_BAO_ZiWen_21386043.md)与[评估开发方案](../FedCrop_项目评估与开发方案.md)。
- [PlantVillage 数据仓库](https://github.com/spMohanty/PlantVillage-Dataset)：原始彩色图片与叶片映射，固定图片提交 `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`。
- [PlantVillage Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)：固定来源 revision 后下载 splits 和 leaf-map；revision 写入数据清单。数据集卡标注 CC BY-SA 3.0，发布数据衍生材料时保留来源与相应条款。
- Mohanty, Hughes, Salathé (2016), [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419)。
- McMahan et al. (2017), [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html)。本项目采用等客户端权重变体，区别于按样本数加权。
- [Flower SecAgg+ 官方示例](https://flower.ai/docs/examples/flower-secure-aggregation.html)：本项目使用其工作流和客户端 mod，未自行设计密码协议。
- [Opacus 官方文档](https://opacus.ai/api/privacy_engine.html)：逐样本裁剪、噪声、采样和记账；实现使用底层 GradSampleModule、DPOptimizer 与 DPDataLoader。
- [ONNX Runtime 量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)：静态 QDQ S8S8、公开校准集与 CPU 执行。

外部工具和数据的作者保留其版权。本项目自身实现应按课程要求说明生成式工具辅助范围；不能将上游协议或已有算法声明为个人原创贡献。
