import numpy as np
from flask import Flask, request, jsonify
import pickle


load_model = pickle.load(open('iris_model.pkl', 'rb'))
server = Flask(__name__)

@server.route('/',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request=[[data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']]]
    predict_request=np.array(predict_request)
    # print(request)
    prediction = load_model.predict(predict_request)
    pred = prediction[0]
    result=""
    if int(pred) == 0:
        result = "Setosa"
    elif int(pred) == 1:
        result = "Versicolor"
    else:
        result = "Virginica"
    return jsonify(result)
