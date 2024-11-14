import numpy as np
import torch
from torch.nn import Module

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cal_score(model: Module, val_dataset):
    value_list = []
    model.eval()

    _, tmp_mask = val_dataset[0]
    mask_size = np.argmax(tmp_mask, axis=0).cpu().numpy().size

    # 在验证集上计算
    with torch.no_grad():
        for i, (image, mask) in enumerate(val_dataset):
            tmp = image.unsqueeze(0).to(device)
            output = model(tmp)
            result = (output.squeeze().cpu().numpy())
            y_pred = np.argmax(result, axis=0)
            y_true = np.argmax(mask, axis=0).cpu().numpy()
            # 计算值1：同时为0或者同时不为0的计数值
            value1 = np.sum(np.logical_and(y_pred == 0, y_true == 0)) + np.sum(np.logical_and(y_pred != 0, y_true != 0))

            # 计算值2：相等的计数值
            value2 = np.sum(np.equal(y_pred, y_true))

            value_list.append([value1, value2])

    value1 = np.average([value[0] for value in value_list]) / mask_size
    value2 = np.average([value[1] for value in value_list]) / mask_size

    return value1 * 40 + value2 * 60

