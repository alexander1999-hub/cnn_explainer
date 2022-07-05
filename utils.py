import torch
import torch.nn.functional as F
import cv2
torch.manual_seed(0)
from typing import Tuple, Optional
import torchvision
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from IPython.display import display
from torch import nn
from os import path
import warnings
import PIL

from torchvision.transforms.functional import crop

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def crop1200(image):
    return crop(image, 0, 0, 1200, 1200)

def preprocess_image(image):
    image = transforms.functional.to_tensor(image)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def overlay_heatmap_on_image(img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    overlay = heatmap + np.float32(img)
    overlay = overlay / np.max(overlay)
    return np.uint8(255*overlay)

class ClassificationModelTorch(nn.Module):
    """
    Class detects Resnet18 Model
    """
    def __init__(self, model_path: Optional[str], num_classes: int = 6) -> None:
        """
        first 2 classes mean columns number
        last 4 classes mean orientation
        """
        super(ClassificationModelTorch, self).__init__()

        self.resnet18 = models.resnet18(pretrained=model_path is None)

        self.resnet18.fc = nn.Linear(512, num_classes)  # change output class number

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet18(x)
        return out

class ColumnsOrientationClassifier(object):
    """
    Class Classifier for work with Orientation Network. This class set device,
    preprocessing (transform) input data, weights of model
    """
   
    _nets = {}

    def __init__(self, on_gpu: bool, checkpoint_path: Optional[str]) -> None:
        self._set_device(on_gpu)
        self._set_transform_image()
        self.checkpoint_path = path.abspath(checkpoint_path)
        self.classes = [1, 2, 0, 90, 180, 270]

    @property
    def net(self) -> ClassificationModelTorch:
        # lazy loading and net sharing, comrade
        if self.checkpoint_path not in self._nets:
            net = ClassificationModelTorch(path.join(self.checkpoint_path, "orient_class_resnet18_bigger_bs1_old.pth"))
            self._load_weights(net)
            net.to(self.device)
            self._nets[self.checkpoint_path] = net
        return self._nets[self.checkpoint_path]

    def _set_device(self, on_gpu: bool) -> None:
        """
        Set device configuration
        """
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.location = lambda storage, loc: storage.cuda()
        else:
            self.device = torch.device("cpu")
            self.location = 'cpu'

    def _load_weights(self, net: ClassificationModelTorch) -> None:
        path_checkpoint = path.join(self.checkpoint_path, "orient_class_resnet18_bigger_bs1_CROP_TOP_LEFT_OLD_DATA.pth")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.load_state_dict(torch.load(path_checkpoint, map_location=self.location))
            print("Weights were loaded from ", path_checkpoint)

    def save_weights(self, path_checkpoint: str) -> None:
        torch.save(self.net.state_dict(), path_checkpoint)

    def _set_transform_image(self) -> None:
        """
        Set configuration preprocessing for input image
        """
        self.transform = transforms.Compose([
            transforms.Resize(1200),
            transforms.Lambda(crop1200),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Predict class orientation of input image
        """
        with torch.no_grad():

            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(np.uint8(image)).convert('RGB')

            tensor_image = self.transform(pil_image).unsqueeze(0).float().to(self.device)

            outputs = self.net(tensor_image)

            # first 2 classes mean columns number
            # last 4 classes mean orientation
            columns_out, orientation_out = outputs[:, :2], outputs[:, 2:]

            _, columns_predicted = torch.max(columns_out, 1)
            _, orientation_predicted = torch.max(orientation_out, 1)

        columns, orientation = int(columns_predicted[0]), int(orientation_predicted[0])
        columns_predict = self.classes[columns]
        angle_predict = self.classes[2 + orientation]
        print("Correct prediction: ", columns_predict, angle_predict)
        # return outputs  # for area importance heatmaps
        # return orientation + 2 # for gradcam heatmaps
        return columns_predict, angle_predict # for feature visualization
