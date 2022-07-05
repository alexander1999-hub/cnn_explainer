import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
import PIL
import cv2
import os

def crop1200(image):
    return crop(image, 0, 0, 1200, 1200)

my_transform = transforms.Compose([
        transforms.Resize(1200),
        transforms.Lambda(crop1200),
    ])

def iterate_through_all_files_importance_heatmaps(folder_path): 
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            filename = file[:-4]
            im_weights_heatmap_filename = os.path.join(folder_path, 'top_left_im_weights', 'final_heatmap_' + filename+'.npy')
            np_weights_heatmap_filename = os.path.join(folder_path, 'top_left_np_weights', 'final_heatmap_' + filename+'.npy')

            im_weights_heatmap = np.load(im_weights_heatmap_filename)
            np_weights_heatmap = np.load(np_weights_heatmap_filename)

            im_weights_heatmap = np.uint8((im_weights_heatmap / im_weights_heatmap.max()) * 255)
            np_weights_heatmap = np.uint8((np_weights_heatmap / np_weights_heatmap.max()) * 255)

            orig_image_name = os.path.join(folder_path, file)
            orig_image = cv2.imread(orig_image_name)
            orig_image = my_transform(PIL.Image.fromarray(np.uint8(orig_image)))
            orig_image = np.array(orig_image)

            im_weights_heatmap = cv2.applyColorMap(im_weights_heatmap, cv2.COLORMAP_JET)
            im_weights_heatmap = cv2.resize(im_weights_heatmap, (1200, 1200), interpolation=cv2.INTER_NEAREST)

            im_overlayed = cv2.addWeighted(orig_image, 0.5, im_weights_heatmap, 0.5, 0)
            im_combined = np.concatenate((orig_image, im_weights_heatmap, im_overlayed), axis=1)

            np_weights_heatmap = cv2.applyColorMap(np_weights_heatmap, cv2.COLORMAP_JET)
            np_weights_heatmap = cv2.resize(np_weights_heatmap, (1200, 1200), interpolation=cv2.INTER_NEAREST)

            np_overlayed = cv2.addWeighted(orig_image, 0.5, np_weights_heatmap, 0.5, 0)
            np_combined = np.concatenate((orig_image, np_weights_heatmap, np_overlayed), axis=1)

            combined = np.concatenate((im_combined, np_combined), axis=0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (10, 2350)
            org_im = (10, 60)
            org_np = (10, 1260)
            
            fontScale = 2
            color = (0, 0, 0)
            thickness = 2
            
            combined = cv2.putText(combined, 'IMUTILS WEIGHTS', org_im, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            combined = cv2.putText(combined, 'NUMPY WEIGHTS', org_np, font, 
                            fontScale, color, thickness, cv2.LINE_AA) 
            combined = cv2.putText(combined, folder_path[-2:].upper()+' PICTURE', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)

            cv2.imwrite(os.path.join(folder_path, 'combined_heatmaps', 'heatmap' + file), combined)

def iterate_through_all_files_gradcam_heatmaps(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            # im_weights_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', 'im_' + file)
            # np_weights_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', 'np_' + file)

            im_weights_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', '2nd_layer_im_' + file)
            np_weights_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', '2nd_layer_np_' + file)

            im_weights_heatmap = cv2.imread(im_weights_heatmap_filename)
            np_weights_heatmap = cv2.imread(np_weights_heatmap_filename)
            print(im_weights_heatmap.shape, np_weights_heatmap.shape)
            combined = np.concatenate((im_weights_heatmap, np_weights_heatmap), axis=1)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # org = (1200, 1150)
            # org_im = (600, 50)
            # org_np = (1800, 50)
            
            # fontScale = 2
            # color = (0, 0, 0)
            # thickness = 2
            
            # combined = cv2.putText(combined, 'IMUTILS WEIGHTS', org_im, font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)
            # combined = cv2.putText(combined, 'NUMPY WEIGHTS', org_np, font, 
            #                 fontScale, color, thickness, cv2.LINE_AA) 
            # combined = cv2.putText(combined, folder_path[-2:].upper()+' PICTURE', org, font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)

            cv2.imwrite(os.path.join(folder_path, 'gradcam_heatmaps', '2nd_layer_combined_' + file), combined)

def combine_gradcam_heatmaps(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".png"):

            first_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', '1st_layer_combined_' + file)
            second_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', '2nd_layer_combined_' + file)
            third_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', '3rd_layer_combined_' + file)
            fourth_heatmap_filename = os.path.join(folder_path, 'gradcam_heatmaps', 'combined_' + file)

            first_heatmap = cv2.imread(first_heatmap_filename)
            second_heatmap = cv2.imread(second_heatmap_filename)
            third_heatmap = cv2.imread(third_heatmap_filename)
            fourth_heatmap = cv2.imread(fourth_heatmap_filename)

            print(file)

            combined = np.concatenate((fourth_heatmap, third_heatmap, second_heatmap, first_heatmap), axis=0)

            cv2.imwrite(os.path.join(folder_path, 'gradcam_heatmaps', 'all_layers_combined_' + file), combined)

if __name__ == "__main__":
    combine_gradcam_heatmaps('/home/alexander/Documents/cnn_explainer/pictures_to_explain/np')