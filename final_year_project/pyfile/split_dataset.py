import os
import shutil

origin_img_dir = '../dataset/lfw_cropped/faces'
for txt_file in os.listdir('../dataset/lfw_cropped/lists'):
    number, train_test, category = txt_file.split('.')[0].split('_')
    if not number.isdigit():
        continue
    dest_img_dir = os.path.join('lfw_cropped/split_data', train_test, number)
    if not os.path.exists(dest_img_dir):
        os.mkdir(dest_img_dir)
    dest_img_dir = os.path.join(dest_img_dir, category)
    if not os.path.exists(dest_img_dir):
        os.mkdir(dest_img_dir)
    with open(os.path.join('lfw_cropped/lists', txt_file), 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            imgs = line.strip().split(' ')
            os.mkdir(os.path.join(dest_img_dir, str(i)))
            for img in imgs:
                img += '.jpg'
                shutil.copyfile(os.path.join(origin_img_dir, img), os.path.join(dest_img_dir, str(i), img))
