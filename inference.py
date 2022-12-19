import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net(num_classes=5):
    # load the pre-trained model - resnet50
    model = models.resnet50(pretrained=True)
    # freeze part of  model except the last linear layer 
    for param in model.parameters():
        param.requires_grad = False  
    # input size for last linear layer
    num_features = model.fc.in_features
    # update the last linear layer
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def model_fn(model_dir):
    model = net()
    model_path = os.path.join(model_dir, 'model.pth')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if (device == torch.device("cpu")) or (device=="cpu"):
                model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(
            torch.load(model_path))

    # with open(os.path.join(model_dir, "model.bin"), "rb") as f:
    #     model.load_state_dict(torch.load(f))
    logger.info('Successfully loaded the model')
    return model.to(device)

def input_fn(request_body, content_type='image/jpeg'):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint

    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    
    if content_type == 'image/jpeg': 
        return Image.open(io.BytesIO(request_body))
    
    elif content_type == 'application/json':
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    else:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    
    logger.info('In predict fn')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1))
    ])

    logger.info("transforming input")
    input_object=transform(input_object)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_object = input_object.to(device)
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

