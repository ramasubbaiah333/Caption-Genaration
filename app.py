import numpy as np
import os
import uuid
import requests
from whitenoise import WhiteNoise

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

import sample
import pickle
# import torch
app = Flask(__name__)

# model = pickle.load(open('model.pkl', 'rb'))
UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# YANDEX_API_KEY = 'YOUR API KEY HERE'
SECRET_KEY = os.urandom(24) #'YOUR SECRET KEY FOR FLASK HERE'

app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

@app.route('/')
def home():
    return render_template('index.html')

# check if file extension is right
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# force browser to hold no cache. Otherwise old result returns.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/shortenurl', methods=['GET', 'POST'])
def shortenurl():
    try:
        # remove files created more than 5 minute ago
        os.system("find static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    if request.method == 'POST':
        # check if the post request has the file part
        if 'content-file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        content_file = request.files['content-file']
        files = [content_file]
        # give unique name to each image
        content_name = str(uuid.uuid4()) + ".png"
        file_names = [content_name]
        for i, file in enumerate(files):
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect('/')
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
        args={
            'image' : "static/images/" + file_names[0],
            'encoder': 'models/encoder-1.pkl',
            'decoder': 'models/decoder-1.pkl',
            'vocab_path': 'vocab.pkl', 
            'embed_size': 512,
            'hidden_size': 512,
        }        
        # returns created caption
        caption = sample.main(args)
        params={
            'content': "static/images/" + file_names[0],
            'caption': caption,
            # 'tr_caption': tr_caption,
        }
        return render_template('success.html', **params)            
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)