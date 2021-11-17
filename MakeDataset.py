import cv2
import os
import matplotlib.pyplot as plt
img_path = './GF2_PMS1_E114.1_N30.1_20160207_L1A0001395956-MSS1'
img_list = os.listdir(img_path)
label_path = img_path+'_label'
label_list = os.listdir(label_path)
projection = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]#label:built_up, farmland, water
if not os.path.isdir('./dataset'):
    os.mkdir('./dataset')
for index, item in enumerate(label_list):

    print(label_path+'/'+item)
    label = cv2.imread(label_path+'/'+item, cv2.IMREAD_UNCHANGED)
    plt.imshow(label)
    plt.show()
    # label
    img = cv2.imread(label_path+'/'+img_list[index])
