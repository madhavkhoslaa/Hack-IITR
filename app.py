from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/pred_fertilizer', methods=['GET'])
def pred_fertilizer():
    data = request.json
    a= data['content']
    pred= Labels_fertilizer.inverse_transform(model.predict(np.array(a).reshape(1,8)))
    return jsonify(pred[0])
if __name__ == '__main__':
    model = pickle.load(open("serialised_data/SVC_FERTILISER", 'rb'))
    Labels_crop = pickle.load(open("serialised_data/Labels_crop", 'rb'))
    Labels_fertilizer = pickle.load(open("serialised_data/Labels_fertilizer", 'rb'))
    Labels_Soil = pickle.load(open("serialised_data/Labels_Soil", 'rb'))
    app.run(port=8081)
