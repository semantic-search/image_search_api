from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
from db_models.models.feature_model import Features
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from feature_extractor import FeatureExtractor
import uuid

fe = FeatureExtractor()
app = FastAPI()
global_init()
features = list()
img_ids = list()
doc_ids = list()


for feature_obj in Features.objects:
    features.append(pickle.loads(feature_obj.feature))
    img_ids.append(feature_obj.file)
    doc_ids.append(feature_obj.document)
features = np.array(features)

@app.post("/search/")
def search(file: UploadFile = File(...)):
    file_name = str(uuid.uuid4()) + file.filename
    with open(file_name, 'wb') as f:
        f.write(file.file.read())

    # Run search
    query = fe.extract(file_name)
    dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:30]  # Top 30 results
    scores = [(dists[id], ids[id]) for id in ids]
    print(scores)

