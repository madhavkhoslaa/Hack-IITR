from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import random
import psycopg2


app = Flask(__name__)

TOKEN_LIST=['DWUDBWQFIVEFESZJNFEFNZEFN']

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
    data["Moisture"] = random.uniform(0, 1)*100
    data["potassium"] = random.uniform(0, 1)*100
    data["Calcium"] = random.uniform(0, 1)*100
    data['Nitrogen'] = random.uniform(0, 1)*100
    data['Phosphorus'] = random.uniform(0, 1)*100
    data['potash'] = random.uniform(0, 1)*100
    return jsonify(data)


@app.route('/what2grow/<string:arr>', methods=['GET'])
def what2grow(arr):
    model2 = pickle.load(open("serialised_data/WHAT2GR0-MODEL", 'rb'))
    Label_Crops_type = pickle.load(open("serialised_data/Label_Crops_type", 'rb'))
    data = argparser_(arr, 'a')
    input = np.array(data).reshape(1, 5)
    preds = Label_Crops_type.inverse_transform(model2.predict(input))[0]
    return jsonify(preds)

@app.route('/add_to_database/<string:arr>', methods=["GET"])
def add_to_database(arr):
    data= argparser_(arr)
    if len(data) > 6 or < 6:
        return jsonify("Cannot fit in data 6 parameters expected (Moisture, potassium, Calcium, Nitrogen, Phosphorus, potash)")
    else:
        connection= psycopg2.connect(database= "da1p9ru6qnr7o1", user='wbwmyrgtobbutg', port="5432", password="5bdc8dbef1b91a1c52b4f74076c7076070b9a4b5e1c9f40775851c216fcb5e79", host="ec2-54-195-252-243.eu-west-1.compute.amazonaws.com")
        cursor = connection.cursor()
        postgres_insert_query = "INSERT INTO SENSOR_VALUES (Moisture,potassium , Calcium, Nitrogen, Phosphorus, potash) VALUES (%s, %s, %s, %s,%s, %s)".format(data)
        cursor.execute(postgres_insert_query)
        cursor.close()
        connection.close()
@app.route('/route_query/<string:query>/<string:token>', methods=["GET"])
def route_query(query, token):
    if token in TOKEN_LIST:
        connection= psycopg2.connect(database= "da1p9ru6qnr7o1", user='wbwmyrgtobbutg', port="5432", password="5bdc8dbef1b91a1c52b4f74076c7076070b9a4b5e1c9f40775851c216fcb5e79", host="ec2-54-195-252-243.eu-west-1.compute.amazonaws.com")
        cursor = connection.cursor()
        cursor.execute(query)
        return 400
    else:
        return jsonify("Token not correct")
@app.route("/", methods=['GET'])
def welcome():
    help_= dict()
    help_["Why and what ?"] = "This app is an aid to help farmers grow their crops and to curate a database for agriculture reasearchers from the sensors to study and apply machine learning on"
    help_["Sensors"]= "Sensors would be community owned by farmers and they can get to know about their individual fields"
    help_["Endpoints Available"]= ["/what2grow/<string:arr>", "/stats", '/pred_fertilizer/<string:arr>']
    help_["How to Rest call an array ?"]= "arr at the end of the url expects array elements with a delimiter as a, didnt use json because easier to parse this ways"
    help_["/what2grow/"]= "This inputs expected parameters from the farmer and tells him what to grow this season"
    help_["/pred_fertilizer/"]= "This uses expected data from the farmer about the soil and tells the fertilizer to use"
    help_["Example query"]= "https://kisan-app.herokuapp.com/1%2%3%4%5"
    help_["add_to_database/<string:arr>"]= "Command to add values to the database"
    help_["route_query/<string:query>/<string:token>"]= "Route queries to the open database for researchers who have access token"
    return jsonify(help_)

    

if __name__ == '__main__':
    app.run(port=8081, debug=True)