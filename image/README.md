## 环境说明

- 同目录下的 `corn.zip` 为打包好的 conda 环境，包含了本项目所需的所有依赖包，具体依赖见 `ROOT/data/code/requirements.txt`。

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