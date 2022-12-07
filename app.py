#%%

import os
from flask import Flask, render_template, request, send_file
from loadt4 import ejecutar
import json



#%%



app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

'''
@app.route('/get_data',methods=['GET'])
def summary():
    data = final_run()
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response
'''


#%%

@app.route('/taller_4',methods=['POST'])
def get_data():
    print("ENTER GET_DATA")
    
    
    json_request = request.json
    funcion=json_request["funcion"]
    json_=json_request["json_"]
    modelo=json_request["modelo"]

    try:
        funcion=json_request["funcion"]
    except:
        funcion=json_request["funcion"]
    try:
        json_=json_request["json_"]
    except:
        json_='Predecir'
    try:
        modelo=json_request["modelo"]
    except:
        modelo=None

    data=ejecutar(funcion, json_, modelo)
    response = app.response_class(
            response=json.dumps(data),
            status=200,
            content_type='application/json'
        )
    return response
        

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
