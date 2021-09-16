# # https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md
# import os
# import cv2
# import numpy as np
# import argparse
# import warnings
# import time
# from src.anti_spoof_predict import AntiSpoofPredict
# from src.generate_patches import CropImage
# from src.utility import parse_model_name
# warnings.filterwarnings('ignore')
#
# model_dir = 'resources/anti_spoof_models'
# # define a video capture object
# vid = cv2.VideoCapture(0)
#
# while (True):
#
#     ret, frame = vid.read()
#     # cv2.imshow('frame', frame)
#     image = frame
#     model_test = AntiSpoofPredict(0)
#     image_cropper = CropImage()
#     image_bbox = model_test.get_bbox(image)
#     prediction = np.zeros((1, 3))
#     # print(prediction)
#     test_speed = 0
#
#     for model_name in os.listdir(model_dir):
#         h_input, w_input, model_type, scale = parse_model_name(model_name)
#         param = {
#             "org_img": image,
#             "bbox": image_bbox,
#             "scale": scale,
#             "out_w": w_input,
#             "out_h": h_input,
#             "crop": True,
#         }
#
#         if scale is None:
#             param["crop"] = False
#         img = image_cropper.crop(**param)
#         start = time.time()
#         prediction += model_test.predict(img, os.path.join(model_dir, model_name))
#         test_speed += time.time() - start
#
#     label = np.argmax(prediction)
#     value = prediction[0][label] / 2
#     if label == 1:
#         print("Image '{}' is Real Face. Score: {:.2f}.".format(image, value))
#         result_text = "RealFace Score: {:.2f}".format(value)
#         color = (0, 255, 0)
#     else:
#         print("Image '{}' is Fake Face. Score: {:.2f}.".format(image, value))
#         result_text = "FakeFace Score: {:.2f}".format(value)
#         color = (0, 0, 255)
#     print("Prediction cost {:.2f} s".format(test_speed))
#     cv2.rectangle(
#         image,
#         (image_bbox[0], image_bbox[1]),
#         (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
#         color, 2)
#     cv2.putText(
#         image,
#         result_text,
#         (image_bbox[0], image_bbox[1] - 5),
#         cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)
#
#     cv2.imshow('frame', frame)
#
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
# vid.release()
# cv2.destroyAllWindows()
# https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md
import os
import cv2
import argparse
import warnings
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import joblib
import numpy as np
from PIL import Image
from face_recognition import preprocessing
from inference.util import draw_bb_on_img
from inference.constants import MODEL_PATH
import datetime




warnings.filterwarnings('ignore')
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

model_dir = 'resources/anti_spoof_models'
# define a video capture object
vid = cv2.VideoCapture(2)
def webcame():

    while (True):
        ret, frame = vid.read()
        # cv2.imshow('frame', frame)
        image = frame
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        # print(prediction)
        test_speed = 0
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time() - start

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        if label == 1:
            if value >= 0.90:
                # print("Image '{}' is Real Face. Score: {:.2f}.".format(image, value))
                print("Real Face. Score: {:.2f}.".format(value))

                result_text = "RealFace Score: {:.2f}".format(value)
                color = (0, 255, 0)
                return label
        else:
            # print("Image '{}' is Fake Face. Score: {:.2f}.".format(image, value))
            print("Fake Face. Score: {:.2f}.".format(value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))

        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)

        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    vid.release()
    cv2.destroyAllWindows()
# webcame()
def check():
    label = webcame()
    if label == 1:
        # cap = cv2.VideoCapture(0)
        face_recogniser = joblib.load(MODEL_PATH)
        preprocess = preprocessing.ExifOrientationNormalize()

        while True:
            # Capture frame-by-frame
            ret, frame = vid.read()
            frame = cv2.flip(frame, 1)

            img = Image.fromarray(frame)
            faces = face_recogniser(preprocess(img))
            if faces is not None:
                draw_bb_on_img(faces, img)
                return faces[0][0][0]

            # Display the resulting frame
            cv2.imshow('video', np.array(img))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the captureq
        vid.release()
        cv2.destroyAllWindows()
# check()
def attend():
    name = check()
    curr_date = datetime.date.today()
    ctime = time.strftime('%H:%M:%S')

    with open('attend.csv', 'r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            print(entry[0])
            nameList.append(entry[0])
        if name not in nameList:
            x = datetime.datetime.now()
            # print(x)
            f.writelines(f'\n{name},{x}')
        else:
            print("you are already in")

        print(nameList)

attend()