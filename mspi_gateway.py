import argparse
import base64
import json
import os
import pickle
import git
import requests
import torch
from flask import Flask
from typing import Dict, Tuple, List
from flask import request

app = Flask(__name__)
urls = []
SLO = 0

@app.route("/")
def call_model():
    inputs = request.args.get('inputs')
    for url in urls:
        try:
            r = requests.get(url, params={'inputs': inputs}, timeout=SLO)
            if r.status_code != 200:
                return "time out!", 500
            # get strings from response
            inputs = r.content
            inputs = base64.encodebytes(inputs).decode()
        except requests.exceptions.Timeout:
            return "time out!", 500
    return "ok!"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='model gateway.')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--urls', nargs='+', required=True)
    parser.add_argument('--SLO', type=int, default=1)

    args = parser.parse_args()
    
    urls = args.urls
    SLO = args.SLO
    
    app.run(host='0.0.0.0', port=args.port, threaded=False, processes=16)