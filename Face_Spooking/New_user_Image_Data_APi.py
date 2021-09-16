import os
import cv2
from torchvision import datasets, transforms
from facenet_pytorch.models.mtcnn import MTCNN
from face_recognition import preprocessing
from PIL import Image
import time
from flask import Flask, jsonify
from flask_restplus import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
app = Flask(__name__)
api = Api(app)

start_time = time.time()

# def main(directory, name, test):
a = input("Enter yor name : ")
directory = 'images/{}'.format(a)

if not os.path.exists(directory):
    os.mkdir(directory)
else:
    print("Already Folder exit")
    print("please ReEnter to Folder Name ")
    a = input("Enter yor name : ")

cap = cv2.VideoCapture(0)

i = 1
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow(a, frame)
    # if i % 5 == 0:
    cv2.imwrite("{}/{}{}.png".format(directory, a, int(i)), frame)
    i += 1
    if cv2.waitKey(1) == 13 or i == 105:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

print("{} folder successfully created ".format(a))

print("Normal image MTCNN process ")

def create_dirs(root_dir, classes):
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    for clazz in classes:
        path = root_dir + os.path.sep + clazz
        if not os.path.isdir(path):
            os.mkdir(path)
#Api
upload_parser = api.parser()

@api.route('/Aadharcard_ocr/')
@api.expect(upload_parser)
class UploadDemo(Resource):
    def post(self):
        try:
            trans = transforms.Compose([
                preprocessing.ExifOrientationNormalize(),
                transforms.Resize(1024)
            ])

            images = datasets.ImageFolder(root='images/')
            images.idx_to_class = {v: k for k, v in images.class_to_idx.items()}
            create_dirs('Images/', images.classes)

            mtcnn = MTCNN(prewhiten=False)

            for idx, (path, y) in enumerate(images.imgs):
                print("Aligning {} {}/{} ".format(path, idx + 1, len(images)), end='')
                aligned_path = 'Images/' + os.path.sep + images.idx_to_class[y] + os.path.sep + os.path.basename(path)
                if not os.path.exists(aligned_path):
                    img = mtcnn(img=trans(Image.open(path).convert('RGB')), save_path=aligned_path)
                    print("No face found" if img is None else '')
                else:
                    print('Already aligned')

        except Exception as e:
            print("Exception Occured", e)
            return jsonify({"Result": e})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8089)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
