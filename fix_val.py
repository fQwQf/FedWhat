import os
import shutil
import argparse

def restructure_val(val_dir):
    annotation_file = 'val_annotations.txt'
    anno_path = os.path.join(val_dir, annotation_file)
    
    # 1. Check if path exists
    if not os.path.exists(val_dir):
        print(f"Error: Directory not found {val_dir}")
        print("Please check if the path is correct!")
        return
    
    if not os.path.exists(anno_path):
        print(f"Error: Annotation file not found {anno_path}")
        return

    print(f"Reading annotation file: {anno_path} ...")
    
    # 2. Read mappings
    with open(anno_path, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} validation images, starting restructuring...")

    count = 0
    # 3. Iterate and move images
    for line in lines:
        parts = line.strip().split('\t')
        filename = parts[0]
        class_label = parts[1]
        
        # Create class directory (e.g. val/n01443537)
        class_dir = os.path.join(val_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)
        
        # Define source and destination paths
        # Original images are usually in val/images/
        src = os.path.join(val_dir, 'images', filename)
        dst = os.path.join(class_dir, filename)
        
        # If image exists in images folder, move it
        if os.path.exists(src):
            shutil.move(src, dst)
            count += 1
        # If image is directly under val/ (some versions do not have images subdirectory), try moving it
        elif os.path.exists(os.path.join(val_dir, filename)):
            src_flat = os.path.join(val_dir, filename)
            shutil.move(src_flat, dst)
            count += 1

    print(f"Successfully processed: {count} images.")
            
    # 4. Clean up empty images folder
    images_folder = os.path.join(val_dir, 'images')
    if os.path.exists(images_folder) and not os.listdir(images_folder):
        os.rmdir(images_folder)
        print("Removed empty 'images' folder.")

    print("\n✅ Restructuring complete! The structure should now comply with PyTorch ImageFolder standard.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restructure Tiny-ImageNet validation set.")
    parser.add_argument('--dir', type=str, default='./data/tiny-imagenet-200/val', 
                        help='Path to the validation directory (default: ./data/tiny-imagenet-200/val)')
    args = parser.parse_args()
    
    restructure_val(args.dir)
