import os
import shutil

# --- 配置区域 ---
# 根据你的 log，你的数据集应该在这个路径
# 如果报错找不到路径，请手动修改这里
VAL_DIR = '/data1/tongjizhou/datasets/tiny-imagenet-200/val'
ANNOTATION_FILE = 'val_annotations.txt'
# ----------------

def restructure_val():
    anno_path = os.path.join(VAL_DIR, ANNOTATION_FILE)
    
    # 1. 检查路径是否存在
    if not os.path.exists(VAL_DIR):
        print(f"错误: 找不到文件夹 {VAL_DIR}")
        print("请检查路径是否正确！")
        return
    
    if not os.path.exists(anno_path):
        print(f"错误: 找不到标签文件 {anno_path}")
        return

    print(f"正在读取标签文件: {anno_path} ...")
    
    # 2. 读取映射关系
    with open(anno_path, 'r') as f:
        lines = f.readlines()

    print(f"找到 {len(lines)} 张验证集图片，开始重组结构...")

    count = 0
    # 3. 遍历并移动图片
    for line in lines:
        parts = line.strip().split('\t')
        filename = parts[0]
        class_label = parts[1]
        
        # 创建类别文件夹 (例如 val/n01443537)
        class_dir = os.path.join(VAL_DIR, class_label)
        os.makedirs(class_dir, exist_ok=True)
        
        # 定义源路径和目标路径
        # 原始图片通常混在 val/images/ 下
        src = os.path.join(VAL_DIR, 'images', filename)
        dst = os.path.join(class_dir, filename)
        
        # 如果 images 文件夹里有这张图，就移动它
        if os.path.exists(src):
            shutil.move(src, dst)
            count += 1
        # 如果图片已经在 val/ 下面（有些解压版本没有 images 子文件夹），尝试直接移动
        elif os.path.exists(os.path.join(VAL_DIR, filename)):
            src_flat = os.path.join(VAL_DIR, filename)
            shutil.move(src_flat, dst)
            count += 1

    print(f"成功处理: {count} 张图片。")
            
    # 4. 清理空的 images 文件夹
    images_folder = os.path.join(VAL_DIR, 'images')
    if os.path.exists(images_folder) and not os.listdir(images_folder):
        os.rmdir(images_folder)
        print("已删除空的 images 文件夹。")

    print("\n✅ 重构完成！现在的结构应该符合 PyTorch ImageFolder 标准了。")

if __name__ == '__main__':
    restructure_val()
