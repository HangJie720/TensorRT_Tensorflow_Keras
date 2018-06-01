import os.path
from PIL import Image
import cv2
import os

def img_resave(img_file, path_save):
    img = Image.open(img_file)
    # new_image = img.resize((width,height),Image.BILINEAR)

    img.save(os.path.join(path_save,os.path.basename(img_file)))

# for i in range(2936):
#     img_resave('/container_dir/flower_photos','/container_dir/flower_photos/train')


root_path = "/container_dir/flower_photos/"

count = 735
pre = 0
for root, dir, files in os.walk(root_path):
    # print(len(files))
    for file in files:
        count -= 1
        if count != 0:
            srcImg = cv2.imread(root_path + str(file))
            cv2.imwrite(root_path + "val" + "/" + str(file), srcImg)
        else:
            break
    break
    # print(count)
    # print(pre)