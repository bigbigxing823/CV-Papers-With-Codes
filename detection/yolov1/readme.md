### 说明

这是一个关于Yolov1复现的仓库，采用的是NEU-DET数据集。

### 安装

```bash
$ cd yolov1 && pip install -r requirements.txt
```

### 预测

1. 下载[权重模型](https://pan.baidu.com/s/1F5MkhRt1kIzM7cuGzEQmwg )，提取码：l9vk。

```bash
$ mv yolov1.pt checkpoints
```

2. 给定一张图片，进行预测

```bash
$ python detect.py 
```

<img src="C:\Users\Administrator\Desktop\yolov1\resources\result.png" alt="result"  />

### 训练

1. 下载[NEU-DET](https://pan.baidu.com/s/1oPrXmDLKdxzcgRsHYndzPQ )，提取码：c594。

2. 转换labels。

```bash
$ mv NEU-DET data
$ cd data && python voc_label.py   # 注意修改为自己的路径
```

3.开始训练。

```
$ python train.py
```

![loss](resources\loss.png)

### 指标评估

1. Ground Truth标签转换。

```
$ cd mAP/scripts && python get_GT.py  # 注意修改为自己的路径
```

2. 批量预测并转换结果。

```
$ python detect_dir.py
$ cd mAP/scripts && python get_DR.py  # 注意修改为自己的路径  
```

3. 指标计算

```
# cd mAP && python main.py
```

![mAP](resources\mAP.png)

### 参考资料

[1] [abeardear/pytorch-YOLO-v1: an experiment for yolo-v1, including training and testing. (github.com)](https://github.com/abeardear/pytorch-YOLO-v1)

[2] [【YoloV1】损失函数最贴近公式的实现+解读（pytorch）_yolov1损失函数 pytorch-CSDN博客](https://blog.csdn.net/Jiangnan_Cai/article/details/132192813)

