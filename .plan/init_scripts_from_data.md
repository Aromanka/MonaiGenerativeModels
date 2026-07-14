现在我要求你实施下列微调计划，最终目的是构造一个“输入ehr latent feature作为控制condition，生成OCT眼底图像的生成模型”。
注意，部分数据和模型存储在服务器上，因此本地无法访问。如果需要更多信息，你可以向我索取。

### 数据来源
1. 主要数据
两个纯字典 `.pt` 文件，存储了两类患者：
- `data/ukbehr_ehr_with_oct.pt`：存在 EHR 与 OCT 的患者，内容为 `{patient_id(int): ehr_latent}`
- `data/ukbehr_ehr_only.pt`：存在 EHR 但是缺失 OCT 的患者，内容为 `{patient_id(int): ehr_latent}`
你应该使用 “`data/ukbehr_ehr_with_oct.pt`：存在 EHR 与 OCT 的患者”构造train split 和 val split。

2. OCT原始图像
encoder 文件夹下存储了OCT图像处理的代码，其中 @encoder\README.md 记录了如何获取得到OCT图像（已经处理完毕），以及其数据格式、如何使用等信息。
其中，最重要的数据字典位置为：`/data/home/wanglidi/code/encode_oct/data/OCT_eid.json`

### 原始模型和微调方案
基于“A. 推荐主方案 MONAI CXR LDM”（checkpoint已部署，load model测试均已通过）展开微调。

### 文档生成
最后，将文档写入 @README.md
