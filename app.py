import json
import os

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, jsonify, request
import joblib


app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = joblib.load('pretrained_vit_HAM10000_cpu.pkl')
model.eval()                                            



# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),    
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


# Get a prediction
def get_prediction(input_tensor):
    with torch.inference_mode():
        prediction = model(input_tensor)
    prediction_probs = torch.softmax(prediction, dim=1)
    predicted_label = torch.argmax(prediction_probs, dim=1).item()                      # Extract the int value from the PyTorch tensor
    return predicted_label

# Make the prediction human-readable
def render_prediction(prediction_idx):
    render_dic={5: 'Melanocytic nevi',
     4: 'Melanoma',
     2: 'Benign keratosis ',
     1: 'Basal cell carcinoma',
     0: 'Actinic keratoses',
     6: 'Vascular lesions',
     3: 'Dermatofibroma'}
    class_name=render_dic[prediction_idx]

    return prediction_idx, class_name


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0')