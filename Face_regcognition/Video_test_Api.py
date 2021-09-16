import joblib
import cv2
import numpy as np
from PIL import Image
from face_recognition import preprocessing
from inference.util import draw_bb_on_img
from inference.constants import MODEL_PATH
from flask import Flask, jsonify
from flask_restplus import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()

@api.route('/Aadharcard_ocr/')
@api.expect(upload_parser)
class UploadDemo(Resource):
    def post(self):
        try:
            cap = cv2.VideoCapture(0)
            face_recogniser = joblib.load(MODEL_PATH)
            preprocess = preprocessing.ExifOrientationNormalize()

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                img = Image.fromarray(frame)
                faces = face_recogniser(preprocess(img))
                if faces is not None:
                    draw_bb_on_img(faces, img)

                # Display the resulting frame
                cv2.imshow('video', np.array(img))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # When everything done, release the captureq
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print("Exception Occured", e)
            return jsonify({"Result": e})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8089)












