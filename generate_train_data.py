import shutil
import os 
import glob 
import cv2 
from PIL import Image
output_root = '/mnt/fillipo/yaowei/tower/annotation_data/train_data/yolo_11/fire'
data_root = {
    # 'billowing': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/billowing smoke',
    # 'white smoke': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/white smoke',
    # 'fire': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/fire',
    # 'tree': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/forest/',
    # 'fishingboat' : '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/fishingboat',
    # 'rowboat': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/rowboat',
    # 'buoy': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/buoy',
    'chimney': '/mnt/fillipo/yaowei/tower/annotation_data/crop_label/chimney',
    'electricity tower':'/mnt/fillipo/yaowei/tower/annotation_data/crop_label/electricity tower',
}

test_per = 0.2
img_size = 128
i = 0
for class_name, data_path in data_root.items():
    train_output_folder = os.path.join(output_root, 'train', class_name)
    test_output_folder = os.path.join(output_root, 'test', class_name)
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)
    images = glob.glob(os.path.join(data_path, '*.png'))
    image_nums = len(images)
    train_images = images[:int(image_nums*(1-test_per))]
    test_images = images[int(image_nums*(1-test_per)):]
    for index, image in enumerate(train_images):
        # image_name = image.split('/')[-1].split('.')[0].replace('_', '-')
        image_name = image.split('/')[-1].split('.')[0]
        # image_name = str(index) + f'_{class_name}'
        img = cv2.imread(image)
        
        # img = Image.open(image)
        # img.convert('RGB').save(os.path.join(train_output_folder, image_name)+'.png')
        out_image = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join(train_output_folder, image_name)+'.png', out_image)
    for index, image in enumerate(test_images):
        # image_name = image.split('/')[-1].split('.')[0].replace('_', '-')
        image_name = image.split('/')[-1].split('.')[0]
        # image_name = str(index) + f'_{class_name}'
        img = cv2.imread(image)
        # img = Image.open(image)
        # img.convert('RGB').save(os.path.join(test_output_folder, image_name)+'.png')
        
        out_image = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join(test_output_folder, image_name)+'.png', out_image)
        
    
    
    