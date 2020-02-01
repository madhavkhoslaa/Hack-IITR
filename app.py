from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import random

app = Flask(__name__)

def argparser_(strr, delim_='%'):
    return list(map(int, strr.split(delim_)))


@app.route('/pred_fertilizer/<string:arr>', methods=['GET'])
def pred_fertilizer(arr):
    model= pickle.load(open("serialised_data/SVC_FERTILISER", 'rb'))
    #Labels_crop = pickle.load(open("serialised_data/Labels_crop", 'rb'))
    #Labels_Soil = pickle.load(open("serialised_data/Labels_Soil", 'rb'))
    Labels_fertilizer = pickle.load(open("serialised_data/Labels_fertilizer", 'rb'))
    a = argparser_(arr, 'a')
    pred = Labels_fertilizer.inverse_transform(
            model.predict(np.array(a).reshape(1, 8)))
    return jsonify(pred[0])


@app.route('/stats', methods=['GET'])
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


@app.route('/what2grow/<string:arr>', methods=['GET'])
def what2grow(arr):
    model2 = pickle.load(open("serialised_data/WHAT2GR0-MODEL", 'rb'))
    Label_Crops_type = pickle.load(open("serialised_data/Label_Crops_type", 'rb'))
    data = argparser_(arr, 'a')
    input = np.array(data).reshape(1, 5)
    preds = Label_Crops_type.inverse_transform(model2.predict(input))[0]
    return jsonify(preds)

@app.route('/<string:arr>', methods=['GET'])
def slash(arr):
    return jsonify(arr)

if __name__ == '__main__':
    app.run(port=8081, debug=True)
