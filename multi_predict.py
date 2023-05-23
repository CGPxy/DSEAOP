
import numpy as np
import SimpleITK as sitk
import os
import argparse
from tensorflow.keras.models import load_model


def BCE():
    def dice(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return dice

def original_predict():
    n_classes = 3

    model = load_model("./model/model.hdf5", custom_objects={'dice':BCE})
    model.summary()

    for filename in os.listdir("./input/images/pelvic-2d-ultrasound/"):
        image = sitk.ReadImage("./input/images/pelvic-2d-ultrasound/" + filename)
        image = sitk.GetArrayFromImage(image)
        image = image.astype("float") / 255.0
        image = np.transpose(image, (1,2,0))
        image = np.expand_dims(image, axis=0) 

        pred, pred1,pred2,pred3,pred4,pred5,pred6,pred7  = model.predict(image,verbose=2)
        
        preimage = pred.reshape((256,256,n_classes)).argmax(axis=-1)

        preimage = np.uint8(preimage) ## uint8 .mha

        preimage = sitk.GetImageFromArray(preimage)
        
        savepath = './output/images/symphysis-segmentation/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        preimage = sitk.WriteImage(preimage, savepath+filename)

if __name__ == '__main__':
    original_predict()