from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    mosaic=1.0, #四张图拼接
    #mixup=0.2,  #两张图混合，适合分类任务
    fliplr=0.5, #翻转
    scale=0.5   #缩放
)