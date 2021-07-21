import flask
import io
import string
import time
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image
import os,sys
import numpy as np
import cv2
import json
from flask import Flask, jsonify, request, render_template

sys.path.append('keras-retinanet/')


from models.opencv_utils import predict_cv, AP_metric
from models.retinanet_model import save_drawed_boxes, retina_predict
from keras_retinanet.models import load_model

model_input_path = "C:\\document_object_detection_task\\keras-retinanet\\model.h5"

model_output_path = "static/output_"

model = load_model(model_input_path, backbone_name='resnet50')

app = Flask(__name__)
@app.route("/")
def hello():
    return render_template('page.html')

@app.route("/", methods=['GET', 'POST'])
def test_model():
    if request.method == 'POST':
        file_name = request.form['file_name']
        
        boxes_cv = predict_cv(file_name, model_output_path + 'cv_' + file_name)
        boxes_model, scores, labels = retina_predict(input_path=file_name, 
                                               output_path=model_output_path + file_name,
                                                 model=model, threshold=0.06)
        boxes_model=boxes_model[0].astype(int)
        print("AP metric for boxes: {}".format(AP_metric(scores[0], 
                                thresholds=np.arange(start=0.035, stop=0.91, step = 0.035))))
        boxes = boxes_model if args.is_opencv_model_used == 'True' else boxes_cv
    return json.dumps(boxes.tolist(), separators=(',', ':'))

@app.route("/cv", methods=['GET', 'POST'])
def get_opencv_img():
    return "output82092117.png"

@app.route("/model", methods=['GET', 'POST'])
def get_model_img():
    return "output82092117.png"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection service')
    parser.add_argument('is_opencv_model_used', metavar='boolean', help="Use opencv model or note")
    args = parser.parse_args()
    app.run(port=8888)




