from ultralytics import YOLO

model=YOLO("./runs/detect/train8/weights/best.pt")
#使用label评估模型
metrics=model.val(
    data="./dataset.yaml",
    split="test",
    save_json=True #保存详细结果
)

print("\n====TEST RESULTS=====")
print(metrics)
#使用模型推理，不需要label
model.predict(
    source="./dataset/images/test",
    save=True,
    conf=0.25
)

print("\n预测结果已保存到 run/detect/predict")