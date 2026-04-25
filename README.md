#依据yolov8建筑面裂痕检测项目

![检测结果](“./result/val_batch0_pred.jpg”)

#项目简介

本项目基于 YOLOv8 detect模型实现建筑表面裂缝的目标检测任务.对于像素级的裂痕检测任务来说，更适合的模型是yolov8 segment。但是本项目未来将更新基于yolo检测的ROI标注，并且基于此通过传统图像处理方法进行像素级检测。

#项目结构

row_data包含原始图片./images和二值化groundTruth ./GT_images；yolov8n.pt是官方预训练模型；dataset中存放分类后的数据集及标注信息。run中包含yolo模型自动生成训练结果，包含损失函数，PR曲线以及mAP等信息。

#训练步骤

step1:将二值化groundTruth转化成yolo格式(label.py)。对其进行形态学处理使裂痕更连续，然后使用opencv的findContours函数对框选裂痕区域。

step2:划分数据集（dataPrepro.py），按照8：1：1的比例将有数据集分为train\val\test。

step3:加载预训练模型进行训练(train.py)，并且在测试集上验证模型(test.py)。
