from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
print(os.listdir("/"))
model = SentenceTransformer('/jobTitleApp/jobTitleModel')

### Possible Titles, JDs
jd = pd.read_csv('/jobTitleApp/data.csv')
position_list = [{'tag':row['position']} for id,row in jd.iterrows()]
position_embeddings = model.encode(position_list)
position_embeddings_norm = np.divide(position_embeddings,np.linalg.norm(position_embeddings,axis=1)[:,np.newaxis])

desc_list = [{'doc':row['Job Description']} for id,row in jd.iterrows()]
desc_embeddings = model.encode(desc_list)
desc_embeddings_norm = np.divide(desc_embeddings,np.linalg.norm(desc_embeddings,axis=1)[:,np.newaxis])


@app.route('/test', methods=['POST'])
def test():
    return jsonify("success")

@app.route('/', methods=['POST'])
def encode():
    data = request.json
    jobDescription = data.get('jobDescription', '')
    batch_size = int(data.get('batch_size', 16))
    print(jobDescription)
    embeddings = model.encode([{"doc":jobDescription}], batch_size=batch_size)
    desc_embeddings_norm_ = np.divide(embeddings,np.linalg.norm(embeddings,axis=1)[:,np.newaxis])
    cosine_sim = np.dot(position_embeddings_norm,desc_embeddings_norm_[0])
    sorted_indexes = np.argsort(-cosine_sim)
    predTitle =  jd.loc[sorted_indexes[0:10],'position']
    return jsonify({"title":predTitle.to_json(),"status":'success'})


@app.route('/testJD', methods=['POST'])
def testJD():
    data = request.json
    i = int(data.get('i', 11))
    return jsonify(topn(i).to_json())


def topn(i,n=7):
  cosine_sim = np.dot(position_embeddings_norm,desc_embeddings_norm[i])
  sorted_indexes = np.argsort(-cosine_sim)
  print(desc_list[i]['doc'])
  return jd.loc[sorted_indexes[0:n],'position']


def similarn(i,n=7):
  cosine_sim = np.dot(position_embeddings_norm,position_embeddings_norm[i])
  sorted_indexes = np.argsort(-cosine_sim)
  return jd.loc[sorted_indexes[0:n],'position']


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

