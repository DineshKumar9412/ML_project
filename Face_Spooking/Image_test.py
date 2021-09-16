import os
import joblib
import argparse
from PIL import Image
from inference.util import draw_bb_on_img
from inference.constants import MODEL_PATH
from face_recognition import preprocessing


def recognise_faces(img):
    faces = joblib.load(MODEL_PATH)(img)
    if faces:
        draw_bb_on_img(faces, img)
    return faces, img

preprocess = preprocessing.ExifOrientationNormalize()
img = Image.open('Moorthy.jpeg')
filename = img.filename
img = preprocess(img)
img = img.convert('RGB')

faces, img = recognise_faces(img)
print(faces[0][0][1] * 100)
if faces[0][0][1] * 100 >= 80:
    print(faces[0][0][0])
else:
    print("NO")
if not faces:
    print('No faces found in this image.')

# img.show()