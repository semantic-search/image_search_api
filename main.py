from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
from db_models.models.feature_model import Features
from db_models.models.file_model import FilesModel
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from feature_extractor import FeatureExtractor
import uuid
import base64


fe = FeatureExtractor()
app = FastAPI()
global_init()
features = list()
feature_ids = list()
doc_ids = list()


for feature_obj in Features.objects:
    features.append(pickle.loads(feature_obj.feature))
    feature_ids.append(feature_obj)
    doc_ids.append(feature_obj.document.id)


@app.post("/search/")
def search(file: UploadFile = File(...), skip: int = 0):
    skip = skip * 10
    limit = skip + 10
    file_name = str(uuid.uuid4()) + file.filename
    with open(file_name, 'wb') as f:
        f.write(file.file.read())

    # Run search
    query = fe.extract(file_name)
    dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
    np_array = np.argsort(dists)
    ids = np_array[skip:limit]  # Top 30 results
    files_b_64 = list()
    scores = list()
    documents = list()
    for id in ids:
        scores.append(1 - float(dists[id]))
        files_b_64.append(base64.b64encode(feature_ids[id].file.read()))
        cache_obj = Cache.objects.get(id=doc_ids[id])
        documents.append(str(cache_obj.file_name))
    final = {
        "total_pages": len(np_array),
        "document": documents,
        "scores": scores,
        "files": files_b_64
    }
    return final

