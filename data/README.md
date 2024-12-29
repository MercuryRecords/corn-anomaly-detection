## 参赛选手说明

1. 李泰川 北京航空航天大学 计算机学院 taichuanli@buaa.edu.cn
2. 黄厚坤 北京航空航天大学 计算机学院 huanghoukun@buaa.edu.cn 
3. 王睿风 北京航空航天大学 计算机学院 22373180@buaa.edu.cn

------

## 项目复现流程说明

------

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

### 4. **B榜预测数据预处理**

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

### 8. **运行注意事项**

- 本项目的训练和推理代码运行时会产生较多的临时文件，请确保有足够的磁盘空间。
- 为了避免磁盘空间不足，推理和训练主程序中会在每次推理或训练前删除所有临时文件，并使得推理或训练过程按顺序运行。如果想要并行运行推理或训练，请保证有足够的磁盘空间，并手动运行对应的推理或训练脚本。