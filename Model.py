import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2 as cv2
from os import listdir

FILE = "model_9_2021_04_27__16_10_46.pth"
#FILE = "model_12_2021_04_28__09_34_36.pth"
device = torch.device("cpu")

height = 224
width = 224

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.required_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 13)

model.load_state_dict(torch.load(FILE, map_location=device))
model.eval()


def transforms_image(image_bytes):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transformer = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    )

    #image = cv2.imread(image_bytes, 1)
    image = Image.open(image_bytes)
    return transformer(image).unsqueeze(0)

def transforms_array(image_array):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transformer = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    )

    #image = cv2.imread(image_bytes, 1)
    image = Image.fromarray(image_array)
    return transformer(image).unsqueeze(0)

def get_prediction(image_tensor):
    images = image_tensor
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    return predictions

def accuracy_test():
    images_path = r"C:\pdf\pp\8_Semestr\Semestr VIII\ICR\Projekt\Dataset\data\\"
    class_name = ["bb", "bk", "bn", "bp", "bq", "br", "empty", "wb",
    "wk", "wn", "wp", "wq", "wr"]
    errors_matrix = np.zeros((len(class_name), len(class_name)))
    for cn in range(len(class_name)):
        class_path = images_path + class_name[cn] + "\\"
        f = listdir(class_path)
        for image in f:            
            image_path = class_path + image
            crop_img = cv2.imread(image_path, 1)
            tensor = transforms_array(crop_img)
            prediction = get_prediction(tensor)
            errors_matrix[cn][prediction] += 1
    print(errors_matrix)


#tensor = transforms_image("data_image1423.jpeg")
#prediction = get_prediction(tensor)
#print(prediction)
#accuracy_test()