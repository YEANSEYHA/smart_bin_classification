import os
import shutil
import random

# Paths
SOURCE_DIR = "/home/monyratanak/smart_bin_classification/dataset"
DEST_DIR = "/home/monyratanak/smart_bin_classification/dataset_yolo"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

# Create split folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# Split per class
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # Create class folders
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(DEST_DIR, "train", class_name, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(DEST_DIR, "val", class_name, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(DEST_DIR, "test", class_name, img)
        )

    print(f"{class_name}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

print("✅ Dataset split completed successfully!")
