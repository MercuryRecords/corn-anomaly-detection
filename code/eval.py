import numpy as np
import torch
from torch.nn import Module

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cal_score(model: Module, val_loader):
    value_list = []
    model.eval()

    # 在验证集上计算
    # with torch.no_grad():
    #     for i, (image, mask) in enumerate(val_dataset):
    #         tmp = image.unsqueeze(0).to(device)
    #         output = model(tmp)
    #         result = (output.squeeze().cpu().numpy())
    #         y_pred = np.argmax(result, axis=0)
    #         y_true = np.argmax(mask, axis=0).cpu().numpy()
    #         # 计算值1：同时为0或者同时不为0的计数值
    #         value1 = np.sum(np.logical_and(y_pred == 0, y_true == 0)) + np.sum(np.logical_and(y_pred != 0, y_true != 0))
    #
    #         # 计算值2：相等的计数值
    #         value2 = np.sum(np.equal(y_pred, y_true))
    #
    #         value_list.append([value1, value2])
    #
    # value1 = np.average([value[0] for value in value_list]) / mask_size
    # value2 = np.average([value[1] for value in value_list]) / mask_size

    # return value1 * 40 + value2 * 60

    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            outputs = outputs.cpu().numpy()
            masks = torch.argmax(masks, dim=1).cpu().numpy()
            value1 = np.sum(np.logical_and(outputs == 0, masks == 0)) + np.sum(np.logical_and(outputs != 0, masks != 0))
            value2 = np.sum(np.equal(outputs, masks))
            value_list.append([masks.size, value1, value2])

    pixel_cnt = np.sum([value[0] for value in value_list])
    value1 = np.sum([value[1] for value in value_list]) / pixel_cnt
    value2 = np.sum([value[2] for value in value_list]) / pixel_cnt

    return value1 * 40 + value2 * 60

def cal_score_v2(model: Module, val_loader):
    value_list = []
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images = torch.tensor(images, dtype=torch.float32).to(device)
            # images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            outputs = outputs.cpu().numpy()
            masks = torch.argmax(masks, dim=1).cpu().numpy()
            value1 = np.sum(np.logical_and(outputs == 0, masks == 0)) + np.sum(np.logical_and(outputs != 0, masks != 0))
            value2 = np.sum(np.equal(outputs, masks))
            value_list.append([masks.size, value1, value2])

    pixel_cnt = np.sum([value[0] for value in value_list])
    value1 = np.sum([value[1] for value in value_list]) / pixel_cnt
    value2 = np.sum([value[2] for value in value_list]) / pixel_cnt

    return value1 * 40 + value2 * 60



if __name__ == '__main__':
    import torch
    from PIL import Image
    import os
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)


    class SegmentationDatasetV2(Dataset):
        def __init__(self,
                     root_dir,
                     transform=None,
                     array_prefix='block',
                     mask_prefix='standard',
                     array_suffix='.npy',
                     mask_suffix='.png',
                     inference=False,
                     ):
            self.root_dir = root_dir
            self.transform = transform
            self.array_prefix = array_prefix
            self.mask_prefix = mask_prefix
            self.array_suffix = array_suffix
            self.mask_suffix = mask_suffix
            self.arrays = os.listdir(os.path.join(root_dir, 'blocks'))
            self.inference = inference

        def __len__(self):
            return len(self.arrays)

        def __getitem__(self, idx):
            array_name = self.arrays[idx]
            array_path = os.path.join(self.root_dir, 'blocks', array_name)
            array = np.load(array_path)

            if self.inference:
                return array, array_name

            mask_path = os.path.join(self.root_dir, 'masks',
                                     array_name.replace(self.array_prefix, self.mask_prefix).replace(self.array_suffix,
                                                                                                     self.mask_suffix))
            mask = Image.open(mask_path)
            mask = np.array(mask)

            # 创建三个二进制掩码，每个类别一个
            mask_0 = (mask == 0).astype(np.uint8)  # 类别0的掩码
            mask_1 = (mask == 127).astype(np.uint8)  # 类别1的掩码
            mask_2 = (mask == 254).astype(np.uint8)  # 类别2的掩码

            # 将三个掩码堆叠成一个新的数组，形状为(3, 256, 256)
            mask = np.stack([mask_0, mask_1, mask_2], axis=0)

            # 将numpy数组转换为torch tensor
            mask = torch.tensor(mask, dtype=torch.float32)

            return array, mask

    import os
    from SegmentationDataset import ROOT
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    import torch
    import random

    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)
    random.seed(3407)

    # 阶段二，加载数据集
    train_dir = os.path.join(ROOT, 'dataV2', 'train')
    val_dir = os.path.join(ROOT, 'dataV2', 'val')

    # 定义变换
    transform = T.Compose([
        T.ToTensor(),
    ])

    # 创建数据集
    train_dataset = SegmentationDatasetV2(root_dir=train_dir)
    val_dataset = SegmentationDatasetV2(root_dir=val_dir)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    import torch
    from segmentation_models_pytorch import Unet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Unet(encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=5,
                 classes=3,
                 activation='softmax'
                 ).to(device)

    model.load_state_dict(torch.load(f'../model/best_model_epoch_{10}.pth', weights_only=True))
    model.eval()
    pass

    print(cal_score_v2(model, val_loader))