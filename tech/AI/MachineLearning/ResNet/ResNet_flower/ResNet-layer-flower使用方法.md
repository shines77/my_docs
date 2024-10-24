# ResNet-layer for flower_photos 使用方法

## 1. 数据集下载

下载地址为：[http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

下载完成后，解压并放置在 `~\.datasets\flower_photos` 目录下，注意：这是当前用户目录，为了避免多个项目使用同一个数据集，故把数据集放到当前用户目录下，不同项目可以共享。

解压后，你应该看到 `~\.datasets\flower_photos\flower_photos` 的目录，因为 `~\.datasets\flower_photos` 目录下面还要放 `train`, `test` 目录。

注意：`~\.datasets` 目录不存在的话，请自行创建。

## 2. 预训练权重文件

ResNet 各种类型的预训练的文件下载列表为：

```python
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
```

这里选择的是 `resnet34`，下载地址是：[https://download.pytorch.org/models/resnet34-b627a593.pth](https://download.pytorch.org/models/resnet34-b627a593.pth)。

下载完成后，解压并放置在 `~\.resnet_models\` 目录下，并把文件更名为 `resnet34-pre-b627a593.pth` ，具体可以看 `train.py` 里的文件名。

注意：`~\.resnet_models` 目录不存在的话，请自行创建。

## 3. 分割 dataset 和 重新调整图形大小

进入 ResNet-layer 项目的根目录，执行下列命令：

```bash
python ./split_dataset.py
```

执行完，会在 `~\.datasets\flower_photos` 目录下面看到 `train` 和 `test` 目录。

## 4. 训练模型

进入 ResNet-layer 项目的根目录，执行下列命令：

```bash
python ./train.py
```

结果如下：

```bash
Using cpu device.
Using batch_size=4 dataloader workers every process.
Using 3306 train_images for training, 364 test_images for validation.

model_weight_path =  C:\Users\xxxxxxxx\.resnet_models\resnet34-pre-b627a593.pth

train epoch[1/10] loss:1.515.:  53%|███████████████               | 439/827 [16:00<11:32,  1.79s/it]
```
