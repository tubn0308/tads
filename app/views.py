# -*- coding: utf-8 -*-
import logging
from flask import Flask, render_template, request
from process.some_process import SomeClass

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        some_instance = SomeClass(5)
        calc_reslt = some_instance.multiply(2)
        logging.info(calc_reslt)
        return render_template('index/index.html', calc_reslt=calc_reslt)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=3939)