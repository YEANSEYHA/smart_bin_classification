import os
import shutil
from sklearn.model_selection import train_test_split


# paths
src_dir = 'dataset'
dst_dir = 'dataset_yolo'


# Get all classes
classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,d))]

# Create folder strucuture

for split in ['train', 'val']:
    for cls in classes:
        os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

# Split and copy images
for cls in classes:
    cls_path = os.path.join(src_dir, cls)
    images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(dst_dir, 'train',cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(dst_dir, 'val',cls, img))

    print(f"{cls} : {len(train_imgs)} train, {len(val_imgs)} val")

print("\nDataset ready!")




