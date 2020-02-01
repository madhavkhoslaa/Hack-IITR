from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import random

app = Flask(__name__)


@app.route('/pred_fertilizer', methods=['GET', 'POST'])
def pred_fertilizer():
    model= pickle.load(open("serialised_data/SVC_FERTILISER", 'rb'))
    Labels_crop = pickle.load(open("serialised_data/Labels_crop", 'rb'))
    Labels_Soil = pickle.load(open("serialised_data/Labels_Soil", 'rb'))
    Labels_fertilizer = pickle.load(open("serialised_data/Labels_fertilizer", 'rb'))
    data = request.json
    a = data['content']
    pred = Labels_fertilizer.inverse_transform(
        model.predict(np.array(a).reshape(1, 8)))
    return jsonify(pred[0])


@app.route('/stats', methods=['GET', 'POST'])
def stats():
    """Ideally this would get data from the sensors by reading their value
            but we're sending synthesised values just for execution"""
    data = dict()
    data["Moisture"] = random.uniform(0, 1)
    data["potassium"] = random.uniform(0, 1)
    data["Calcium"] = random.uniform(0, 1)
    data['Nitrogen'] = random.uniform(0, 1)
    data['Phosphorus'] = random.uniform(0, 1)
    return jsonify(data)


@app.route('/what2grow', methods=['GET', 'POST'])
def what2grow():
    data = request.json
    model2 = pickle.load(open("serialised_data/WHAT2GR0-MODEL", 'rb'))
    Label_Crops_type = pickle.load(open("serialised_data/Label_Crops_type", 'rb'))
    input = np.array(data['content']).reshape(1, 5)
    preds = Label_Crops_type.inverse_transform(model2.predict(input))[0]
    return jsonify(preds)

@app.route('/'):
def slash():
    data= request.json
    return jsonify(data)


if __name__ == '__main__':


    app.run(port=8081)
