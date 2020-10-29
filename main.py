from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
from db_models.models.feature_model import Features
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from feature_extractor import FeatureExtractor
import uuid
import base64
import os
import globals


fe = FeatureExtractor()
app = FastAPI()


origins = [
    globals.CORS_ORIGIN
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_init()
features = list()
feature_ids = list()
doc_ids = list()


print("############FETCHING IMAGE FEATURES FROM DB############")
for feature_obj in Features.objects:
    features.append(pickle.loads(feature_obj.feature))
    feature_ids.append(feature_obj)
    doc_ids.append(feature_obj.document.id)
print("############DONE############")


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
    grid_ids = list()
    document_ids = list()
    for id in ids:
        scores.append(1 - float(dists[id]))
        recog_file = feature_ids[id].file.read()
        b_64_val = base64.b64encode(recog_file)
        if b_64_val is None or not b_64_val:
            print("********************IN EMPTY**************************************")
            feature_ids[id].file.seek(0)
            recog_file = feature_ids[id].file.read()
            b_64_val = base64.b64encode(recog_file)
        files_b_64.append(b_64_val)
        grid_ids.append(str(feature_ids[id].id))
        cache_obj = Cache.objects.get(id=doc_ids[id])
        documents.append(str(cache_obj.file_name))
        document_ids.append(str(doc_ids[id]))
    final = {
        "total_pages": len(np_array),
        "document": documents,
        "document_ids": document_ids,
        "scores": scores,
        "files": files_b_64
    }
    os.remove(file_name)
    return final
