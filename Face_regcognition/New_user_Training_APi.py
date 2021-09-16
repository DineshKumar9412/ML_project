# https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
import os
import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser
import time
from flask import Flask, jsonify
from flask_restplus import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
app = Flask(__name__)
api = Api(app)

start_time = time.time()
MODEL_DIR_PATH = 'model'

def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels

def load_data(features_extractor):
    dataset = datasets.ImageFolder('Images/')
    # print(dataset)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    # print(embeddings, labels, dataset.class_to_idx)
    return embeddings, labels, dataset.class_to_idx

def train(embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    clf = softmax
    clf.fit(embeddings, labels)

    return clf

upload_parser = api.parser()

@api.route('/Aadharcard_ocr/')
@api.expect(upload_parser)
class UploadDemo(Resource):
    def post(self):
        try:
            features_extractor = FaceFeaturesExtractor()

            embeddings, labels, class_to_idx = load_data(features_extractor)

            clf = train(embeddings, labels)

            idx_to_class = {v: k for k, v in class_to_idx.items()}

            target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
            print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

            if not os.path.isdir(MODEL_DIR_PATH):
                os.mkdir(MODEL_DIR_PATH)
            model_path = os.path.join('model', 'face_recogniser.pkl')
            joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)

        except Exception as e:
            print("Exception Occured", e)
            return jsonify({"Result": e})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8089)


print("time elapsed: {:.2f}s".format(time.time() - start_time))