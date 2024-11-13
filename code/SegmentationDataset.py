import os
import random
from shutil import copyfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 code 的路径
parent_directory_path = os.path.dirname(current_file_path)
# 获取项目的根目录
ROOT = os.path.dirname(parent_directory_path)


# 定义测试集与训练集目录

def split_train_val(val_ratio=0.2, image_prefix='result', mask_prefix='standard'):
    source_dir = os.path.join(ROOT, 'data', 'train')
    target_dir = os.path.join(ROOT, 'data', 'val')

    files = os.listdir(os.path.join(source_dir, 'images'))

    num_files = len(files)
    num_val_files = int(num_files * val_ratio)
    val_files = random.sample(files, num_val_files)
    # 将抽取的文件复制到目标目录
    for file in tqdm(val_files):
        # 复制图像文件
        copyfile(os.path.join(source_dir, 'images', file), os.path.join(target_dir, 'images', file))
        # 复制掩码文件
        copyfile(os.path.join(source_dir, 'masks', file.replace(image_prefix, mask_prefix)),
                 os.path.join(target_dir, 'masks', file.replace(image_prefix, mask_prefix)))
        # 删除源目录下的文件
        os.remove(os.path.join(source_dir, 'images', file))
        os.remove(os.path.join(source_dir, 'masks', file.replace(image_prefix, mask_prefix)))


class SegmentationDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 image_prefix='result',
                 mask_prefix='standard',
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_prefix = image_prefix
        self.mask_prefix = mask_prefix
        self.images = os.listdir(os.path.join(root_dir, 'images'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        mask_path = os.path.join(self.root_dir, 'masks', img_name.replace(self.image_prefix, self.mask_prefix))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask



if __name__ == '__main__':
    split_train_val()

    train_dir = os.path.join(ROOT, 'data', 'train')
    val_dir = os.path.join(ROOT, 'data', 'val')

    # 定义变换
    transform = T.Compose([
        T.ToTensor(),
    ])

    # 创建数据集
    train_dataset = SegmentationDataset(root_dir=train_dir, transform=transform)
    val_dataset = SegmentationDataset(root_dir=val_dir, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
