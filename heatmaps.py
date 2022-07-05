from heapq import heapify
import torch
torch.manual_seed(0)
import numpy as np
import torchvision.transforms as transforms
import PIL
import cv2
import os, random

from utils import ColumnsOrientationClassifier
from torchvision.transforms.functional import crop

def generate_fine_heatmap(classifier, image, annotation, filename, step, block_size=100, loss_function = torch.nn.MSELoss(reduction='none')):
    model = classifier.net
    my_transform = transforms.Compose([
        transforms.Resize(1200),
        transforms.Lambda(crop1200),
    ])

    my_transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    noise = np.random.rand(block_size, block_size, 3) * 255

    image = my_transform(PIL.Image.fromarray(np.uint8(image)))
    image = np.array(image)

    height, width, channels = image.shape

    columns = width - block_size + 1
    rows = height - block_size + 1

    heatmap = np.zeros((rows // step + 1, columns // step + 1))

    noised = np.zeros((columns // step + 1, height, width, channels))

    for j, row in enumerate(range(0, rows, step)):
        print("ROW ", row, end='\r')
        noised[:] = image.copy()

        for i, column in enumerate(range(0, columns, step)):
            noised[i, row:row+block_size, column:column+block_size, :] = noise
        
        tensor_of_img = torch.split(torch.stack([my_transform2(item) for item in list(noised)]).to(classifier.device).float(), 6)

        prediction_list = []
        with torch.no_grad(): 
                for img in tensor_of_img : 
                    img_prediction = model(img)
                    prediction_list.append(img_prediction)
        prediction = torch.cat(prediction_list)

        loss_row = loss_function(prediction[:, 2:], annotation[:, 2:].repeat(rows // step + 1,1)).mean(1)
        heatmap[j] = loss_row.cpu()

    np.save(filename[:-4], heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = heatmap * 255
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)

    overlayed = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    combined = np.concatenate((image, heatmap, overlayed), axis=1)

    return combined


def crop1200(image):
    return crop(image, 0, 0, 1200, 1200)

def generate_area_importance_heatmap_with_occlusions(classifier, image, annotation, block_size=100, loss_function = torch.nn.MSELoss()):
    model = classifier.net
    my_transform = transforms.Compose([
        transforms.Resize(1200),
        transforms.Lambda(crop1200),
    ])

    my_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    image = my_transform(PIL.Image.fromarray(np.uint8(image)))
    image = np.array(image)

    height, width, channels = image.shape

    columns = width // block_size
    rows = height // block_size

    heatmap = np.zeros((columns, rows))
    
    for row in range(rows):
        print("ROW ", row, end = '\r')
        for column in range(columns):
            x = column * block_size
            y = row * block_size

            top = int(y)
            left = int(x)
            right = left + block_size
            bottom = top + block_size

            tmp_image = np.copy(image)

            noise = np.random.rand(block_size, block_size, 3) * 255
            tmp_image[int(top):int(bottom), int(left):int(right)] = noise
            tmp_image = PIL.Image.fromarray(tmp_image)
            tensor_image = my_transform2(tmp_image).unsqueeze(0).float().to(classifier.device) 

            with torch.no_grad(): 
                prediction = model(tensor_image)

            loss = round(float(loss_function(prediction[:, 2:], annotation[:, 2:])), 4)

            heatmap[row, column] = loss
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = heatmap * 255
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    overlayed = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    combined = np.concatenate((image, heatmap, overlayed), axis=1)

    return combined

def process_pictures(folder_path) : 
    classifier = ColumnsOrientationClassifier(on_gpu=True, checkpoint_path="./")

    file_list = os.listdir(folder_path)
    random.shuffle(file_list)
    for file in file_list:
        if file.endswith(".png"):
            image_path = os.path.join(folder_path, file)
            print("FILE NAME ", image_path)
            image_array = cv2.imread(image_path)
            output = classifier.predict(image_array)
            #print(output)
            output_picture_name = 'top_left_np_weights/final_heatmap_' + file
            # temp_pic = generate_area_importance_heatmap_with_occlusions(classifier, image_array, output)
            temp_pic = generate_fine_heatmap(classifier, image_array, output, os.path.join(folder_path, output_picture_name), step=25)
            cv2.imwrite(os.path.join(folder_path, output_picture_name), temp_pic)

if __name__ == "__main__":
    process_pictures('/home/alexander/Documents/cnn_explainer/pictures_to_explain/np')
    process_pictures('/home/alexander/Documents/cnn_explainer/pictures_to_explain/im')


