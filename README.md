# 仓库介绍

本分支为 [2024 CCF大数据与计算智能大赛（CCF BDCI）](https://www.datafountain.cn/special/BDCI2024) 中赛题 [基于航片的玉米异常情况识别](https://www.datafountain.cn/competitions/1064) 的决赛方案代码复现仓库，获奖名次为三等奖（树蛙玉米，ID:350477）

# 赛题介绍

## 赛题简介

本赛题聚焦于运用无人机航拍技术（航片）对玉米全生育期进行监测，通过图像处理与机器学习算法，精准识别玉米生长过程中的异常情况。这些异常可能包括病虫害、营养缺乏、环境压力等，它们对玉米的生长和产量有着直接影响。赛题的挑战在于如何利用先进的遥感技术和人工智能算法，从航拍图像中提取有价值的信息，以实现对玉米生长状况的实时监控和精准管理。


## 赛题背景

随着农业科技的迅猛发展，无人机在农业监测领域中的应用日益广泛。玉米作为重要的粮食作物，其生长状况直接影响农业产出和经济效益。然而，传统的人工监测方式往往耗时且覆盖范围有限。因此，通过无人机航片来识别和提取玉米生长过程中的异常情况，不仅可以大幅提升监测效率，还能实现精准农业的目标。此外，这种技术还能科学评估灾害对作物的影响程度及其分布，为农民提供决策支持和政策指导。


## 赛题任务

参赛者需要设计并实现一套完备的图像识别系统，该系统能够自动地从无人机航片中检测出玉米生长过程中的异常情况，并进行深入的数据分析与直观的可视化展示。具体任务包括但不限于以下几个方面：

1. **图像数据的预处理与增强**。
2. **利用深度学习技术训练先进的模型，以识别不同类型的玉米灾害**。
3. **提供详尽的可视化结果，清晰展示灾害的位置、损失程度和分类信息**。
4. **运用人工智能技术进行科学的数据分析，为准确掌握粮食生产状况和合理规划农事活动提供技术支撑**。


## 数据简介

本赛题提供的训练数据集为玉米地块无人机航拍图像以及多光谱数据等，数据多样且具有代表性，适用于模型的训练和测试。


## 数据说明

### 数据坐标系

- 坐标系：WGS_1984

### 数据可见范围

- 上：4518292.67016758
- 左：258121.027018591
- 下：4515734.82606758
- 右：260962.056118591

### 数据集包含以下文件

| 文件名称               | 文件大小   | 描述                                                                                       |
|--------------------|--------|------------------------------------------------------------------------------------------|
| dsm.tif            | 559MB  | 数字表面模型，任务区域的高程文件(DSM)，每个像素均包含经纬度和高程。                                                     |
| gsddsm.tif         | 35.6MB | 降采样为 5m 分辨率的 dsm，可在 M300 或 P4R 仿地飞行时导入使用。                                                |
| result.tif         | 2.40GB | 正射影像成果文件(DOM)，二维重建最主要的成果。                                                                |
| result_Green.tif   | 0.99GB | 各波段影像的正射镶嵌结果。                                                                            |
| result_NIR.tif     | 0.98GB | 各波段影像的正射镶嵌结果。                                                                            |
| result_Red.tif     | 0.99GB | 各波段影像的正射镶嵌结果。                                                                            |
| result_RedEdge.tif | 0.99GB | 各波段影像的正射镶嵌结果。                                                                            |
| standard.tif       | 1.59GB | 玉米分类结果文件，坐标系、空间范围（四至）与提供的数据 result.tif 保持一致，且只有0、1、2（0代表非玉米、1代表玉米未受灾、2代表玉米受灾）三种数值的单波段数据。 |

------

# 项目流程介绍

## 项目复现流程说明

### 1. **公开数据**

- 数据来源：使用组委会提供的公开数据集，应存储于 `raw_data` 文件夹内，包括训练集和测试集。
  - 训练集文件：
    - `result.tif`：原始影像数据。
    - `standard.tif`：标准分割掩码。
    - 其他数据：如 `dsm.tif`（数字高程模型），`result_NIR.tif`（近红外波段），`result_RED.tif`（红波段）。
  - 测试集文件：
    - `result.tif`：待预测的原始影像数据。
    - 供预测使用的其他数据：如 `dsm.tif`（数字高程模型），`result_NIR.tif`（近红外波段），`result_RED.tif`（红波段）。

------

### 2. **预训练模型**

- 使用的模型为 `Unet`，编码器为 `resnet34`，在 `imagenet` 数据集上进行了预训练。
- 权重文件存储路径：`user_data/`。
  - `weightsV0/best_model_epoch_50.pth`：无增强训练模型权重。
  - `weightsV1/best_model_epoch_50.pth`：含数据增强的三通道模型权重。
  - `weightsV2/best_model_epoch_50.pth`：五通道数据增强模型权重。
- 在保存训练结果方面，本项目只保存了模型权重，模型结构的定义通过调用 `segmentation_models_pytorch` 库实现。

------

### 3. **训练数据预处理与伪标签生成**

#### 数据预处理

1. **分块处理**
   - 代码路径：`tif2pngs.py`
   - 将影像切分为大小为 `256x256` 的块，处理时保留边界不完整的块，通过填充补齐。
   - 在`images/` 下存放影像块，`masks/` 下存放对应掩码块。
2. **去除纯黑块**
   - 在预处理后，对纯黑块进行统计，随机保留一部分（例如 100 块），其余删除。

#### 数据增强

在 `train_2.py` 和 `train_3.py` 中引入数据增强，使用 `albumentations` 库：

增强方法

- 水平翻转：`HorizontalFlip(p=0.5)`
- 垂直翻转：`VerticalFlip(p=0.5)`
- 随机仿射变换：`ShiftScaleRotate`。
- 随机裁剪和调整大小：`RandomResizedCrop`。

增强后图片和掩码分别存储在 `images/` 和 `masks/` 中，文件名带有 `_aug` 后缀。

#### 伪标签生成

- 伪标签生成与五通道合并处理通过 `combine_channels.py` 实现。
- 使用红、近红外波段生成 `NDVI`，并结合 `RGB` 和 `DSM` 形成五通道影像`output.tif`。目的是通过五通道结合多种信息，弥补单一数据源的不足，使模型具有更强的区分能力，有助于模型从更多的维度理解和分辨地物。

------

### 4. **B 榜预测数据预处理**

1. 分块
   - 类似训练数据预处理，将测试数据切分为小块存储。
2. 影像合并
   - 测试影像的五通道合并与训练一致。

------

### 5. **模型训练**

#### 超参数选择

使用网格搜索的方法确定了较优的超参数。

#### 模型结构

1. 使用 `segmentation_models_pytorch` 的 `Unet`。
2. 编码器为 `resnet34`。
3. 激活函数为 `softmax`。

#### **损失函数**

- **Focal Loss**：
  $$
  \text{FL}(p_t) = -\alpha(1-p_t)^\gamma \log(p_t)
  $$
  

  - $p_t$：预测概率。
  - $\alpha = 0.8$：平衡因子，减小正负样本不平衡的影响。
  - $\gamma = 2$：聚焦因子，降低简单样本的权重。

#### 训练参数

- 使用 `Adam` 优化器：
  - 学习率：`1e-4`。
- **批量大小**：
  - `train_1.py` 和 `train_2.py`：64。
  - `train_3.py`：32（五通道数据内存占用更高）。
- **训练轮次**：所有模型均训练 **50** 个 epoch。

#### 训练逻辑

- 多通道模型训练：
  - `train_1.py`：三通道无增强。
  - `train_2.py`：三通道数据增强。
  - `train_3.py`：五通道数据增强。
- 每个 epoch 后保存模型到 `weightsVx/` 文件夹，保存文件名格式为 `best_model_epoch_{epoch}.pth`。

------

### 6. **预测流程**

#### 分块推理

- `infer_1.py`、`infer_2.py` 和 `infer_3.py` 分别使用不同的模型进行推理。
- `infer_1.py` 分割图像块的步长为64，`infer_2.py` 和 `infer_3.py` 分割图像块的步长为128，以求使用更大的感受野覆盖同一小块（64 x 64 / 128 x 128）目标图像。
- 模型输出结果为 `(3, 256, 256)`，表示在该图像上各像素属于各三个标签的概率；推理过程中直接将输出结果累加到整体预测矩阵 `(3, H, W)` 上，以实现投票集成的效果；最后使用 `np.argmax` 得到整体预测结果。
- 分块预测后将结果合并，生成完整影像。

#### 模型融合

- 融合策略为每个像素上取标签最大值： $\text{Final Prediction} = \max(\text{Pred}_1, \text{Pred}_2, \text{Pred}_3)$
- 输出文件保存在 `prediction_result/submission.tif`。

------

### 7. **代码结构**

- 主程序入口：`main.py`
- 核心模块：
  - 数据处理：`combine_channels.py`, `tif2pngs.py`
  - 模型训练：`train_1.py`, `train_2.py`, `train_3.py`
  - 推理脚本：`infer_1.py`, `infer_2.py`, `infer_3.py`

------

```
.
├── raw_data/                           # 原始数据存储目录
│   ├── 训练集文件/                     # 训练数据
│   │   ├── result.tif                 # 原始影像
│   │   ├── standard.tif               # 标准分割掩码
│   │   ├── dsm.tif                    # 数字高程模型
│   │   ├── result_NIR.tif             # 近红外波段
│   │   └── result_RED.tif             # 红波段
│   └── 测试集文件/                     # 测试数据
│       └── result.tif                 # 待预测的影像
├── user_data/                         # 处理生成的数据和模型存储目录
│   ├── dataV0/                        # 阶段 0 数据（初始分块数据）
│   │   ├── train/                     # 训练数据分块
│   │   │   ├── images/               # 分块后的影像
│   │   │   └── masks/                # 分块后的掩码
│   │   ├── infer/                     # 推理中分块
│   │   │   └── images/               # 分块后的影像
│   ├── dataV1/                        # 阶段 1 数据（数据增强）
│   │   ├── train/                     # 增强后的训练数据
│   │   │   ├── images/               # 增强后的影像
│   │   │   └── masks/                # 增强后的掩码
│   │   ├── infer/                     # 推理中分块
│   │   │   └── images/               # 分块后的影像
│   ├── dataV2/                        # 阶段 2 数据（五通道合并）
│   │   ├── train/                     # 五通道训练数据
│   │   │   ├── blocks/               # 五通道数据块
│   │   │   └── masks/                # 对应掩码
│   │   ├── infer/                     # 推理中分块
│   │   │   └── images/               # 分块后的影像
│   ├── weightsV0/                     # 阶段 0 模型权重
│   ├── weightsV1/                     # 阶段 1 模型权重
│   └── weightsV2/                     # 阶段 2 模型权重（五通道）
├── prediction_result/
│   └── submission.tif					#最终提交文件
└── code/                               # 项目核心代码目录
    ├── train/ 
    |	├── combine_channels.py        # 五通道合并及 NDVI 计算
    │   ├── tif2pngs.py                # 分块与转格式
    │   ├── train_1.py                 # 三通道无增强训练
    │   ├── train_2.py                 # 三通道数据增强训练
    │   └── train_3.py                 # 五通道数据增强训练
    ├── inference/ 
    |	├── combine_channels.py        # 五通道合并及 NDVI 计算
    │   ├── tif2pngs.py                # 分块与转格式
    │   ├── infer_1.py                 # 三通道无增强推理
    │   ├── infer_2.py                 # 三通道数据增强推理
    │   └── infer_3.py                 # 五通道数据增强推理
    ├── main.py                        # 推理主程序入口
    └── train.py                       # 训练主程序入口

```

------

### 8. **运行注意事项**

- 本项目的训练和推理代码运行时会产生较多的临时文件，请确保有足够的磁盘空间。
- 为了避免磁盘空间不足，推理和训练主程序中会在每次推理或训练前删除所有临时文件，并使得推理或训练过程按顺序运行。如果想要并行运行推理或训练，请保证有足够的磁盘空间，并手动运行对应的推理或训练脚本。

------

# 运行环境相关

## 环境说明

- 项目具体依赖见 `ROOT/data/code/requirements.txt`。

- 需使用 GPU 进行模型训练和推理，Pytorch 版本为 2.0.1，CUDA 版本为 11.7，CUDNN 版本使用 `print(torch.backends.cudnn.version())` 查看为 `8500`，与提供的环境一致。

## 代码运行注意事项

- 项目的 `ROOT/data/code` 目录下包含了所有代码，其中 `main.py` 为推理脚本，`train.py` 为训练脚本。
  - `main.py` 中的推理过程包括依次使用三个模型进行推理，对应脚本为 `ROOT/data/code/inference` 下的三个脚本，并在最后进行结果融合。
  - `train.py` 中的训练过程包括依次使用三个模型进行训练，对应脚本为 `ROOT/data/code/train` 下的三个脚本。
- 本项目的训练和推理代码运行时会产生较多的临时文件，请确保有足够的磁盘空间。
- 为了避免磁盘空间不足，推理和训练入口中会在每次推理或训练前删除所有临时文件，并使得推理或训练过程按顺序运行。如果想要并行运行推理或训练，请保证有足够的磁盘空间，并手动运行对应的推理或训练脚本。
- 要将训练好的模型权重应用于推理，请注意 `weightV{num}` 文件夹中的权重与 `models` 文件夹中已有的模型文件的对应关系：
  - `weightV0` 中的权重对应 `models` 中的 `model_3c_no_aug.pth`。
  - `weightV1` 中的权重对应 `models` 中的 `model_3c_aug.pth`。
  - `weightV2` 中的权重对应 `models` 中的 `model_5c_aug.pth`。
- 由于仓库空间限制，没有上传 `models` 文件夹中的权重文件，**若要使用推理部分请自行执行训练脚本**。

## 模型训练过程

1. 数据预处理：在前两个模型的训练过程中，本项目直接将 `result.tif` 文件和 `standard.tif` 文件等大小切分为若干个 `256 x 256` 的 .png 文件，并分别放置于 `images` 和 `masks` 文件夹中。在第三个模型的训练过程中，本项目使用了重投影的方法生成分割模型的输入数据，同时使用了 `standard.tif` 文件等大小切分为若干个 `256 x 256` 的 .npy 文件 和 .png 文件，并分别放置于 `blocks` 和 `masks` 文件夹中。
2. 数据增强：在第二个和第三个模型的训练过程中，本项目利用 `albumentations` 库实现了数据增强，增强后图片和掩码分别带有 `_aug` 后缀存储在同目录下作为训练集：

   - 水平翻转：`HorizontalFlip(p=0.5)`
   - 垂直翻转：`VerticalFlip(p=0.5)`
   - 随机仿射变换：`ShiftScaleRotate`。
   - 随机裁剪和调整大小：`RandomResizedCrop`。

3. 模型训练：在训练过程中，本项目使用 `torch` 库实现了模型的训练，训练过程中会保存模型权重。
4. 模型训练细节请见 `ROOT/data/README.md`。

## 项目入口

- 推理入口：`ROOT/data/code/main.py`。
- 训练入口：`ROOT/data/code/train.py`。