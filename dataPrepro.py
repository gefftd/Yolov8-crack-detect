import os
import random
import shutil

image_dir = "./raw_data/image"
label_dir = "./dataset/raw_labels"

out_img = "./dataset/images"
out_label = "./dataset/labels"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(out_img, split), exist_ok=True)
    os.makedirs(os.path.join(out_label, split), exist_ok=True)

files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(files)

n = len(files)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

train_files = files[:train_split]
val_files = files[train_split:val_split]
test_files = files[val_split:]

def move_files(file_list, split):
    for f in file_list:
        base = os.path.splitext(f)[0]  #分离名称和扩展名，并且取名称

        img_src = os.path.join(image_dir, f)
        label_src = os.path.join(label_dir, base + ".txt")

        if not os.path.exists(label_src):   #images比labels多，这里处理了
            continue

        shutil.copy(img_src, os.path.join(out_img, split, f))
        shutil.copy(label_src, os.path.join(out_label, split, base + ".txt"))

move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("数据划分完成！")

