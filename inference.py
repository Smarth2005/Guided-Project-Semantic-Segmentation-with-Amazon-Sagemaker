import torch
import os
import io
import json
from torchvision import models, transforms
from PIL import Image

# 1. Load the Model (Run once when endpoint starts)
def model_fn(model_dir):
    device = torch.device("cpu")
    print("Loading model...")
    
    # Must match the architecture used in training exactly
    model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=4)
    
    # Load the weights saved in model.pth
    path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(path, map_location=device))
    
    model.to(device).eval()
    print("Model loaded successfully.")
    return model

# 2. Process Input (Runs for every request)
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        # Convert bytes to PIL Image
        return Image.open(io.BytesIO(request_body)).convert("RGB")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# 3. Predict (Runs for every request)
def predict_fn(input_object, model):
    # Preprocess: Resize to match training size (256x256)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    input_tensor = preprocess(input_object).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        
    # Get the predicted class for each pixel (argmax)
    prediction = output.argmax(0).byte().cpu().numpy()
    
    return prediction.tolist() # Return as standard JSON list

# 4. Format Output
def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
