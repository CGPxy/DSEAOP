import numpy as np
import SimpleITK as sitk
import os
import argparse
from tensorflow.keras.models import load_model
from pathlib import Path
# import tensorflow as tf
import cv2

def BCE():
    def dice(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return dice


def FillHole_RGB(data):
    # 读取图像为uint32,之所以选择uint32是因为下面转为0xbbggrr不溢出
    im_in_rgb = data #cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)# .astype(np.uint32)
    colors = [1,2]
    
    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),)+im_in_rgb.shape,np.uint8)
    
    # 初始化二值数组
    im_th = np.zeros(im_in_rgb.shape,np.uint8)
    for l in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if(im_in_rgb[j][k]==colors[l]):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0
        # 复制 im_in 图像
        im_floodfill = im_th.copy()
     
        # Mask 用于 floodFill，官方要求长宽+2.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        
        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if(im_floodfill[m][n]==0):
                    seedPoint=(m,n)
                    isbreak = True
                    break
            if(isbreak):
                break
                
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255);
         
        # 得到im_floodfill的逆im_floodfill_inv
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
         
        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv

        ##################
        ##### 二值化
        ret, thresh = cv2.threshold(im_out, 128, 255, cv2.THRESH_BINARY)
        # 寻找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        if num_labels>2:
            print(num_labels)
            # 计算平均面积
            areas = list()
            for i in range(num_labels):
                areas.append(stats[i][-1])
                # print("轮廓%d的面积:%d" % (i, stats[i][-1]))

            # area_avg = np.average(areas[1:-1])
            # print("轮廓平均面积:", area_avg)
            # print(min(areas))
            # 筛选超过平均面积的连通域
            image_filtered = np.zeros_like(im_out)
            for (i, label) in enumerate(np.unique(labels)):
                # 如果是背景，忽略
                if label == 0:
                    continue
                if stats[i][-1] > min(areas) :
                    image_filtered[labels == i] = 255
            im_result[l] = image_filtered
        else:
            im_result[l] = im_out

        ret1, thresh1 = cv2.threshold(im_result[l], 128, 255, cv2.THRESH_BINARY)
        kernel = np.uint8(np.zeros((2, 2)))
        for x in range(2):
            kernel[x, 1] = 1
            kernel[1, x] = 1

        # if l==0:
        #     # # 膨胀图像
        im_result[l] = cv2.dilate(thresh1, kernel)
        # else:
        #     # 腐蚀图像
        #     im_result[l] = cv2.erode(thresh1, kernel)

    # rgb结果图像
    im_fillhole = np.zeros((im_in_rgb.shape[0],im_in_rgb.shape[1]),np.uint8)
    
    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if(im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colors[k]
                    break

    return im_fillhole


def original_predict(inputpath, savepath):
    n_classes = 3

    model = load_model("./model/model.hdf5", custom_objects={'dice':BCE}) # , 'tf':tf
    model.summary()
    print("load model sucess")

    print("please upload input images")
    for filename in os.listdir(inputpath):
        image = sitk.ReadImage(inputpath + filename)
        image = sitk.GetArrayFromImage(image)
        image = image.astype("float") / 255.0
        image = np.transpose(image, (1,2,0))
        image = np.expand_dims(image, axis=0) 

        pred,pred1, pred2,pred3,pred4,pred5,pred6,pred7  = model.predict(image,verbose=2) # pred, pred1,pred2,pred3,pred4,pred5,pred6,pred7
        
        preimage = pred.reshape((256,256,n_classes)).argmax(axis=-1)

        preimage = np.uint8(preimage) ## uint8 .mha

        #print(preimage.GetSize())
        preimage = FillHole_RGB(data=preimage)

        preimage = sitk.GetImageFromArray(preimage)

        preimage = sitk.WriteImage(preimage, savepath+filename)

if __name__ == '__main__':
    
    inputpath = '/input/images/pelvic-2d-ultrasound/'
    if not os.path.exists(inputpath):
        os.makedirs(inputpath)

    savepath = '/output/images/symphysis-segmentation/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("mikdir inputpath and savepath sucess!")
    print(inputpath)
    print(savepath)

    original_predict(inputpath, savepath)
