import glob
import subprocess
import os
import time

import numpy as np
import rasterio

# 当前文件处于 /data/code 目录下
# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 code 的路径
code_directory_path = os.path.dirname(current_file_path)
# 获取根目录
ROOT = os.path.dirname(code_directory_path)

# 定义要执行的 Python 脚本列表
# scripts = ['train_1.py', 'train_2.py', 'train_3.py']
scripts = ['train_3.py']

def delete_files_with_suffix(root_dir, file_suffix):
    """
    删除指定目录及其子目录下所有以指定后缀结尾的文件。

    :param root_dir: 要搜索的根目录
    :param file_suffix: 文件后缀，例如 ".txt"
    """
    # 构建搜索模式，匹配所有以指定后缀结尾的文件
    pattern = os.path.join(root_dir, '**', f'*{file_suffix}')
    # 使用glob.glob递归查找所有匹配的文件
    for file_path in glob.glob(pattern, recursive=True):
        try:
            # 尝试删除找到的文件
            os.remove(file_path)
            # print(f"Deleted file: {file_path}")
        except Exception as e:
            # 如果文件删除失败，打印错误信息
            print(f"Failed to delete file: {file_path}. Reason: {e}")

for script in scripts:
    delete_files_with_suffix(os.path.join(ROOT, 'user_data'), '.npy')
    delete_files_with_suffix(os.path.join(ROOT, 'user_data'), '.png')
    print(f"正在执行 {script}...")
    # 记录开始时间
    start_time = time.time()
    result = subprocess.run(['python', os.path.join(ROOT, 'code', 'train', script)], capture_output=True, text=True, encoding='utf-8')
    # 记录结束时间
    end_time = time.time()
    print(f"{script} 执行完成，耗时 {end_time - start_time} 秒。")
    if result.returncode != 0:
        print(f"执行 {script} 出错，错误信息：\n{result.stderr}")
    else:
        print(f"{script} 执行完成。")