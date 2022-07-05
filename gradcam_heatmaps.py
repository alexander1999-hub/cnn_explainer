from heatmaps import process_pictures
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
torch.manual_seed(0)
import torchvision
import numpy as np
import torchvision.models as models
from torch.autograd import Function
from torchvision import transforms
from IPython.display import display
import PIL
import ast
import cv2
import os
from torchvision.transforms.functional import crop

from utils import preprocess_image, overlay_heatmap_on_image
from gradcam import GradCam
from utils import ColumnsOrientationClassifier

def process_pictures(folder_path) : 
    classifier = ColumnsOrientationClassifier(on_gpu=True, checkpoint_path="./")

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            image_path = os.path.join(folder_path, file)
            print("FILE NAME ", image_path)
            image_array = cv2.imread(image_path)
            output_class = classifier.predict(image_array)
            output_picture_name = 'gradcam_heatmaps/2rd_layer_np_' + file
            temp_pic = generate_gradcam_heatmap(classifier, image_array, output_class)
            cv2.imwrite(os.path.join(folder_path, output_picture_name), temp_pic)

def generate_cam_overlay_for_target_class(preprocessed_image, image_cut, gradcam, target_class=None):
        img = image_cut
        img = img[:, :, ::-1].copy() 
        img = np.float32(img) / 255

        cam_heatmap = gradcam(preprocessed_image, target_class)
        cam_overlay = overlay_heatmap_on_image(img, cam_heatmap)
        return cam_overlay

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

def generate_gradcam_heatmap(classifier, image, out_class):
    model = classifier.net.resnet18
    gradcam = GradCam(model=model, target_layer=model.layer2, target_layer_names=["1"])
    image_uncut = image
    image_cut = my_transform(PIL.Image.fromarray(np.uint8(image_uncut)))
    tensor_image = my_transform2(image_cut).unsqueeze(0).float().to(classifier.device)   
    image_cut = np.array(image_cut)
    first_overlay = generate_cam_overlay_for_target_class(tensor_image, image_cut, gradcam, out_class)
    
    return first_overlay   

if __name__ == "__main__":
    process_pictures('/home/alexander/Documents/cnn_explainer/pictures_to_explain/im')