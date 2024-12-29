import os

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from tqdm import tqdm


# 重采样函数
def resample_to_reference(input_tif, reference_tif, output_tif):
    if os.path.exists(output_tif):
        print("output_tif已存在，跳过重采样。")
        return

    with rasterio.open(reference_tif) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(input_tif) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear #这个采样方式是双线性插值：通过距离权重对周围四个像素的值进行加权平均，生成平滑的结果。
                )


# 计算 NDVI（分块处理）
def calculate_ndvi_blockwise(nir_tif, red_tif, output_ndvi_tif, block_size=512):
    if os.path.exists(output_ndvi_tif):
        print("output_ndvi_tif 已存在，跳过 NDVI 计算。")
        return

    with rasterio.open(nir_tif) as nir_src, rasterio.open(red_tif) as red_src:
        meta = nir_src.meta.copy()
        width = nir_src.width
        height = nir_src.height

        # 更新输出元数据
        meta.update({"dtype": "float32", "count": 1})
        with rasterio.open(output_ndvi_tif, 'w', **meta) as dst:
            total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
            with tqdm(total=total_blocks, desc="Calculating NDVI") as pbar:
                for row_start in range(0, height, block_size):
                    for col_start in range(0, width, block_size):
                        # 动态调整窗口大小
                        row_size = min(block_size, height - row_start)
                        col_size = min(block_size, width - col_start)
                        window = Window(col_start, row_start, col_size, row_size)

                        nir_block = nir_src.read(1, window=window).astype('float32')
                        red_block = red_src.read(1, window=window).astype('float32')

                        # 计算 NDVI
                        denominator = nir_block + red_block
                        denominator[denominator == 0] = np.nan  # 避免除以零
                        ndvi_block = (nir_block - red_block) / denominator
                        ndvi_block[np.isnan(ndvi_block)] = 0  # 将 NaN 值处理为 0

                        # 写入块到输出文件
                        dst.write(ndvi_block, 1, window=window)
                        pbar.update(1)


# 合并通道（分块处理）
def combine_channels(rgb_tif, height_tif, ndvi_tif, output_tif, block_size=512):
    if os.path.exists(output_tif):
        print("output_tif 已存在，跳过通道合并。")
        return


    with rasterio.open(rgb_tif) as rgb_src, \
            rasterio.open(height_tif) as height_src, \
            rasterio.open(ndvi_tif) as ndvi_src:

        # 获取元数据并更新为5个通道
        meta = rgb_src.meta.copy()
        meta.update({"count": 5, "dtype": "float32"})

        with rasterio.open(output_tif, 'w', **meta) as dst:
            total_blocks = ((rgb_src.height + block_size - 1) // block_size) * \
                           ((rgb_src.width + block_size - 1) // block_size)

            with tqdm(total=total_blocks, desc="Combining Channels") as pbar:
                for row_start in range(0, rgb_src.height, block_size):
                    for col_start in range(0, rgb_src.width, block_size):
                        # 动态调整块大小
                        row_size = min(block_size, rgb_src.height - row_start)
                        col_size = min(block_size, rgb_src.width - col_start)
                        window = Window(col_start, row_start, col_size, row_size)

                        rgb_block = rgb_src.read(window=window)
                        height_block = height_src.read(1, window=window)
                        ndvi_block = ndvi_src.read(1, window=window)

                        # 写入到输出文件
                        dst.write(rgb_block[0], 1, window=window)  # Red
                        dst.write(rgb_block[1], 2, window=window)  # Green
                        dst.write(rgb_block[2], 3, window=window)  # Blue
                        dst.write(height_block, 4, window=window)  # Height
                        dst.write(ndvi_block, 5, window=window)  # NDVI

                        pbar.update(1)


# 主函数
def get_tif_metadata(tif):
    with rasterio.open(tif) as src:
        meta = src.meta
        return meta


def combine(root_path, reference_tif, dsm_tif, nir_tif, red_tif, output_tif):
    # 中间文件路径
    os.makedirs(os.path.join(root_path, 'user_data', 'tmp'), exist_ok=True)
    resampled_dsm = os.path.join(root_path, 'user_data', 'tmp', 'resampled_dsm.tif')  # 重采样后的 DSM
    resampled_nir = os.path.join(root_path, 'user_data', 'tmp', 'resampled_nir.tif')  # 重采样后的 NIR
    resampled_red = os.path.join(root_path, 'user_data', 'tmp', 'resampled_red.tif')  # 重采样后的 Red
    ndvi_tif = os.path.join(root_path, 'user_data', 'tmp', 'ndvi.tif')  # 计算得到的 NDVI

    # 删除可能已有的输出文件和中间文件
    for file in [output_tif, resampled_dsm, resampled_nir, resampled_red, ndvi_tif]:
        if os.path.exists(file):
            os.remove(file)

    # 步骤 1：重采样
    print("重采样 DSM 文件...")
    resample_to_reference(dsm_tif, reference_tif, resampled_dsm)

    print("重采样 NIR 波段...")
    resample_to_reference(nir_tif, reference_tif, resampled_nir)

    print("重采样 Red 波段...")
    resample_to_reference(red_tif, reference_tif, resampled_red)

    # 步骤 2：计算 NDVI
    print("计算 NDVI（分块处理）...")
    calculate_ndvi_blockwise(resampled_nir, resampled_red, ndvi_tif)

    # 步骤 3：合并通道
    print("合并通道到最终输出（分块处理）...")
    combine_channels(reference_tif, resampled_dsm, ndvi_tif, output_tif)

    print("处理完成！输出文件为：", output_tif)