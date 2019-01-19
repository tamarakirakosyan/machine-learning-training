import torch
from Models import Simple_FC
from Load import load_data_test
from PIL import Image
from torchvision import transforms
import numpy as np

class Flower_classifier():
    def __init__(self, model_path):
        self._load_model(model_path)
        self._prepare_transform()
        self.i2class = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

    def _load_model(self, model_path):
        model = Simple_FC() 
        state_dict = torch.load(model_path) 
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def _prepare_transform(self):
        self.transformer = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def predict(self, path_img):
        img = Image.open(path_img)
        img = self.transformer(img)
        img = img.view(1, 64*64*3)
        print(img.shape)
        out = self.model(img)#batchsize, channels, width, height
        prediction = torch.argmax(out, 1).item()

        return self.i2class[prediction]


    def predict_proba(self, path_img):
        img = Image.open(path_img)
        img = self.transformer(img)
        img = img.view(1, 64*64*3)
        print(img.shape)
        out = self.model(img) #batchsize, channels, width, height
        prob = torch.nn.functional.softmax(out, dim = 1)
        prob_list = prob.data[0].tolist()
        prob_dict = dict(zip(self.i2class, prob_list))

        return prob_dict

