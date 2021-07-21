import matplotlib.pyplot as plt
import cv2
import os, sys
import numpy as np
import time

sys.path.append('..\\keras-retinanet\\')


from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.models import load_model



def save_drawed_boxes(boxes, scores, labels, threshold, img, output_path):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        
        if score < threshold:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(img, b, color=color)

    #     caption = "{} {:.3f}".format(labels_to_names[label], score)
    #     caption = ""
    #     draw_caption(draw, b, caption)

    cv2.imwrite(output_path, img)



def retina_predict(input_path, output_path, model, threshold):
    image = read_image_bgr(input_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    boxes /= scale
    
    save_drawed_boxes(boxes, scores, labels, threshold, draw, output_path)
    
    return boxes, scores, labels