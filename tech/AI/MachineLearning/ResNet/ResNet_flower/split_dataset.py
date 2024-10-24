"""
#
# 文件： split_dataset.py
#
# 功能： 数据集划分脚本。将原始数据集 flower_photos 划分为 train 和 test 两个数据集，并更改图片 size = 224x224。
#
#        数据集下载地址：http://download.tensorflow.org/example_images/flower_photos.tgz
#
"""

import os
import glob
import random
from PIL import Image

def filepath_filter(filepath):
    # Windows operation is 'nt' or 'windows'
    # print('os.name = ', os.name)
    # print('os.sep = ', os.sep)
    path_separator = os.sep
    # Whether is windows?
    if path_separator == '\\':
        new_filepath = filepath.replace('/', path_separator)
        return new_filepath
    else:
        return filepath

def split_dataset(root, split_rate, image_new_size):
    root = filepath_filter(root)
    root = os.path.expanduser(root)
    print('root = ', root)
    file_path = '{}/flower_photos'.format(root)    # 获取原始数据集路径
    file_path = filepath_filter(file_path)
    print('file_path = ', file_path)
    assert os.path.exists(file_path), "file {} does not exist.".format(file_path)

    # 找到文件中所有文件夹的目录，即类文件夹名
    dirs = glob.glob(os.path.join(file_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print("Totally {} classes: {}".format(len(dirs), dirs))    # 打印花类文件夹名称

    for path in dirs:
        # 对每个类别进行单独处理
        path = path.split(os.sep)[-1]  # -1表示以分隔符/保留后面的一段字符

        # 在根目录中创建两个文件夹，train/test
        os.makedirs(filepath_filter("{}/train/{}".format(root, path)), exist_ok=True)
        os.makedirs(filepath_filter("{}/test/{}".format(root, path)), exist_ok=True)

        # 读取原始数据集中path类中对应类型的图片，并添加到files中
        files = glob.glob(os.path.join(file_path, path, '*jpg'))
        files += glob.glob(os.path.join(file_path, path, '*jpeg'))
        files += glob.glob(os.path.join(file_path, path, '*png'))

        random.shuffle(files)    # 打乱图片顺序
        split_boundary = int(len(files) * split_rate)  # 训练集和测试集的划分边界

        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')

            # 更改原始图片尺寸
            old_size = img.size  # (wight, height)
            ratio = float(image_new_size) / max(old_size)  # 通过最长的size计算原始图片缩放比率
            # 把原始图片最长的size缩放到resize_pic，短的边等比率缩放，等比例缩放不会改变图片的原始长宽比
            new_size = tuple([int(x * ratio) for x in old_size])

            # im = img.resize(new_size, Image.ANTIALIAS)  # 更改原始图片的尺寸，并设置图片高质量，保存成新图片im
            # 新版本 pillow（10.0.0之后）Image.ANTIALIAS 被移除了，取而代之的是 Image.LANCZOS or Image.Resampling.LANCZOS
            # See: https://zhuanlan.zhihu.com/p/669460623
            im = img.resize(new_size, Image.LANCZOS)
            new_im = Image.new("RGB", (image_new_size, image_new_size))  # 创建一个resize_pic尺寸的黑色背景
            # 把新图片im贴到黑色背景上，并通过'地板除//'设置居中放置
            new_im.paste(im, ((image_new_size - new_size[0]) // 2, (image_new_size - new_size[1]) // 2))

            # 先划分0.1_rate的测试集，剩下的再划分为0.9_ate的训练集，同时直接更改图片后缀为.jpg
            assert new_im.mode == "RGB"
            print('file = ', file)
            # print('dir_name = ', filepath_filter(os.path.join("{}/test/{}".format(root, path))))
            # print('file_name = ', file.split(os.sep)[-1].split('.')[0] + '.jpg')
            if i < split_boundary:
                new_im.save(os.path.join(filepath_filter("{}/test/{}".format(root, path)),
                                         file.split(os.sep)[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(os.path.join(filepath_filter("{}/train/{}".format(root, path)),
                                         file.split(os.sep)[-1].split('.')[0] + '.jpg'))

    # 统计划分好的训练集和测试集中.jpg图片的数量
    train_files = glob.glob(filepath_filter(os.path.join(root, 'train', '*', '*.jpg')))
    test_files = glob.glob(filepath_filter(os.path.join(root, 'test', '*', '*.jpg')))

    print("Totally {} files for train".format(len(train_files)))
    print("Totally {} files for test".format(len(test_files)))

if __name__ == '__main__':
    split_rate = 0.1        # 训练集和验证集划分比率
    resize_image = 224      # 图片缩放后统一大小

    split_dataset('~/.datasets/flower_photos', split_rate, resize_image)
