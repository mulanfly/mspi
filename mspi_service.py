import argparse
import base64
import json
import os
import pickle
import git
import torch
from flask import Flask
from typing import Dict, Tuple, List
from flask import request

from graph_partition.graph_structure import GraphsWrapper, InputsWrapper

app = Flask(__name__)
model = None
inputs = None
device_id = 0

@app.route("/")
def call_model():
    inputs = pickle.loads(base64.decodebytes(request.args.get('inputs').encode()))
    inputs = {k: v.to(f"cuda:{device_id}") if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    outputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in outputs.items()}
    return pickle.dumps(outputs)

def load_model(model_name: str, model_method: str, model_id: int):
    global model
    global inputs

    repo = git.Repo('.', search_parent_directories=True)
    repo.working_tree_dir

    with open(os.path.join(repo.working_tree_dir, 'configs/model_config.json'), 'r') as f:
        config = json.load(f)
        
    path = os.path.join(os.path.join(config["subgraph_root"], model_name), 'TIME_OAP' if model_method == 'WHOLE' else model_method)
    for f in os.listdir(path):
        path = os.path.join(path, f)
        break
    
    if model_method == 'WHOLE':
        ii = torch.load(os.path.join(path, f"{model_name}-inputs-{1}.pt"), map_location=f'cuda:{device_id}')
        mm = []
        for id in range(1, model_id):
            m = torch.load(os.path.join(path, f"{model_name}-submod-{id}.pt"), map_location=f'cuda:{device_id}')
            mm.append(m)
            
        model_wrapper = GraphsWrapper(mm)
        input_wrapper = InputsWrapper(ii)
        model = model_wrapper
        inputs = input_wrapper
    
    else:
        model = torch.load(os.path.join(path, f"{model_name}-submod-{model_id}.pt"), map_location=f'cuda:{device_id}')
        inputs = torch.load(os.path.join(path, f"{model_name}-inputs-{model_id}.pt"), map_location=f'cuda:{device_id}')
        
    
    # warmup
    with torch.inference_mode():
        # warmup 5 times
        for _ in range(5):
            model(**inputs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='model caller.')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default="llama2-7b")
    parser.add_argument('--method', type=str, default="TIME_OAP")

    args = parser.parse_args()
    
    device_id = args.device_id
            
    load_model(args.model_name, args.method, args.model_id)
    
    app.run(host='0.0.0.0', port=args.port, threaded=False, processes=1)