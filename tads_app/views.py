# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, Response, redirect, flash, url_for, send_from_directory
# Flask の起動
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index/index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=3939)