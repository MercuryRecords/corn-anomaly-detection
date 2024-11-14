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



if __name__ == '__main__':
    import os
    from SegmentationDataset import SegmentationDataset, ROOT
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    import torch
    import random

    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)
    random.seed(3407)

    # 阶段二，加载数据集
    train_dir = os.path.join(ROOT, 'data', 'train')
    val_dir = os.path.join(ROOT, 'data', 'val')

    # 定义变换
    transform = T.Compose([
        T.ToTensor(),
    ])

    # 创建数据集
    val_dataset = SegmentationDataset(root_dir=val_dir, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    import torch
    from segmentation_models_pytorch import Unet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Unet(encoder_name='resnet34',
                 encoder_weights='imagenet',
                 classes=3,
                 activation='softmax'
                 ).to(device)
    model.load_state_dict(torch.load(f'../model/best_model_epoch_{50}.pth', weights_only=True))

    # 开始计时
    import time
    start_time = time.time()
    score = cal_score(model, val_loader)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(score)