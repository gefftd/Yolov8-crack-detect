import cv2
import os
import numpy as np
from ultralytics import YOLO
model =YOLO("yolov8n.pt")

mask_dir="./raw_data/GT_image"
save_dir="./dataset/raw_labels"
#show_label_dir="./dataset/labels_visualization"
os.makedirs(save_dir,exist_ok=True)  #会自动创建中间缺失目录
#os.makedirs(show_label_dir,exist_ok=True)
#定义标注函数
def mask_to_yolo(mask_path, save_path):
    #根据mask_path定位到对应的原图path
    image_path=mask_path.replace("GT_image","image").replace("png","jpg")

    mask = cv2.imread(mask_path, 0)  #0是读取灰度图的意思
    image=cv2.imread(image_path)
    #print(image_path)
    if mask is None or image is None:
        print("读取失败",mask_path)
        return
    
    h, w = mask.shape
    #形态学处理
    Kernel=np.ones((5,5),np.uint8)
    mask_morph=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,Kernel)
    mask_morph=cv2.dilate(mask_morph,Kernel,iterations=1)
    contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #这里面是寻找了很多框，现在可以合并成一个
    with open(save_path, 'w') as f:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1=int(x);y1=int(y)
            x2=int(x+bw);y2=int(y+bh)
            # 转YOLO格式
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            bw /= w
            bh /= h

            f.write(f"0 {xc} {yc} {bw} {bh}\n")
            
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("label visualization",image)
    
#主函数
for file in os.listdir(mask_dir):
    if not file.endswith((".png",".jpg")):
        continue
    mask_path=os.path.join(mask_dir,file)
    label_path=os.path.join(save_dir,file.replace(".png",".txt").replace(".jpg",".txt"))
    mask_to_yolo(mask_path,label_path)
    key=cv2.waitKey(0) 
    if key== 27:
        break;

print("转换完成")
