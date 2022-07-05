import torch
torch.manual_seed(0)
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms
from IPython.display import display
import PIL
import cv2
from PIL import ImageFilter
import os

from utils import ColumnsOrientationClassifier
from torchvision import transforms
from torchvision.transforms.functional import crop

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def preprocess_image(image):
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0).transpose(0,3,1,2)
    return torch.Tensor(image)

def deprocess_image(tensor):
    tensor = ((tensor * std) + mean)
    tensor = np.uint8(tensor * 255.0)
    return PIL.Image.fromarray(tensor)

def crop1200(image):
    return crop(image, 0, 0, 1200, 1200)

my_transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

my_transform = transforms.Compose([
        transforms.Resize(1200),
        transforms.Lambda(crop1200),
    ])

class ActivationsExtractor():
    
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output

def visualize(lr, steps, size, upscales, upscale_factor, model, activations, most_activated_filter_idx, out_pic_name):
    img = np.uint8(np.random.uniform(0, 1, (size, size, 3)))
    
    for _ in range(upscales):
        
        img_var = preprocess_image(img).clone().detach().cuda().requires_grad_(True)
        optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

        for _ in range(steps):
            optimizer.zero_grad()
            temp_output =  model(img_var)
            # print(temp_output)
            loss = -activations.features[0, most_activated_filter_idx].mean()
            loss.backward()
            optimizer.step()

        img = deprocess_image(img_var.data.cpu().numpy()[0].transpose(1,2,0))
        #display(img)
        img.save(out_pic_name)

        size = int(upscale_factor * size) 
        img = img.resize((size, size)) 
        img = img.filter(ImageFilter.BLUR)
        img = np.asarray(img) / 255.0

def process_pictures(folder_path) : 

    with open(os.path.join(folder_path, 'visual_features', 'res.txt'), 'a') as f:
        f.write('IM WEIGHTS')
        f.write('\n')

    classifier = ColumnsOrientationClassifier(on_gpu=True, checkpoint_path="./")
    model = classifier.net.resnet18

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            image_path = os.path.join(folder_path, file)
            with open(os.path.join(folder_path, 'visual_features', 'res.txt'), 'a') as f:
                f.write(image_path)
                f.write(' ')
            print("FILE NAME ", image_path)

            image_array = cv2.imread(image_path)
            image_cut = my_transform(PIL.Image.fromarray(np.uint8(image_array)))
            preprocessed_image = my_transform2(image_cut).unsqueeze(0).float().to(classifier.device) 

            layer = list(model.children())[-2] # last non-output layer
            activations = ActivationsExtractor(layer) # activations' hook
            outputs = model(preprocessed_image) # forward pass to collect activations
            columns_out, orientation_out = outputs[:, :2], outputs[:, 2:]
            _, columns_predicted = torch.max(columns_out, 1)
            _, orientation_predicted = torch.max(orientation_out, 1)
            columns, orientation = int(columns_predicted[0]), int(orientation_predicted[0])
            columns_predict = classifier.classes[columns]
            angle_predict = classifier.classes[2 + orientation]

            with open(os.path.join(folder_path, 'visual_features', 'res.txt'), 'a') as f:
                f.write(str(columns_predict))
                f.write(' ')
                f.write(str(angle_predict))

            average_activations_per_filters = [activations.features[0, i].mean().item() for i in range(activations.features.shape[1])]

            most_activated_filter_idx = np.argmax(average_activations_per_filters)

            with open(os.path.join(folder_path, 'visual_features', 'res.txt'), 'a') as f:
                f.write(' ')
                f.write(str(most_activated_filter_idx))
                f.write('\n')

            output_picture_name = 'visual_features/IM_WEIGHTS_' + file

            visualize(lr=0.075,
                      size=56,
                      steps=25,
                      upscales=15,
                      upscale_factor=1.25,
                      model=model, 
                      activations=activations,
                      most_activated_filter_idx=most_activated_filter_idx,
                      out_pic_name=os.path.join(folder_path, output_picture_name))

if __name__ == "__main__":
    process_pictures('/home/alexander/Documents/cnn_explainer/pictures_to_explain/im')
    process_pictures('/home/alexander/Documents/cnn_explainer/pictures_to_explain/np')