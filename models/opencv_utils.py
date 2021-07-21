import os,sys
import numpy as np
import cv2
import sklearn.metrics



def precision_recall_curve(pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
        y_true = ["positive"] * len(y_pred)

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label='positive')
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label='positive')
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls



def AP_metric_from_rec_prec(recalls, precisions):
    recalls.append(0)
    precisions.append(1)
    r = np.array(recalls)
    p = np.array(precisions)
    return np.sum((r[:-1] - r[1:]) * p[:-1])

def AP_metric(pred_scores, thresholds):
    precisions, recalls = precision_recall_curve(pred_scores, thresholds)
    return AP_metric_from_rec_prec(recalls, precisions)

def text_detect(img, ele_size=(8,2)): 
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_sobel = cv2.Sobel(img,cv2.CV_8U,1,0)#same as default,None,3,1,0,cv2.BORDER_DEFAULT)
    img_threshold = cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT,ele_size)
    img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
    res = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cv2.__version__.split(".")[0] == '3':
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res

    Rect = [cv2.boundingRect(i) for i in contours if i.shape[0]>100]
    RectP = [(int(i[0]-i[2]*0.08),int(i[1]-i[3]*0.08),int(i[0]+i[2]*1.1),int(i[1]+i[3]*1.1)) for i in Rect]
    return RectP



# def save_img(img, output_path):
#     cv2.imwrite(output_path, img)
#     return


def predict_cv(input_path, output_path):
    img = cv2.imread(input_path)
    rect = text_detect(img)
    for i in rect:
        cv2.rectangle(img,i[:2],i[2:],(0,0,255))
    cv2.imwrite(output_path, img)
    return np.array(rect)




