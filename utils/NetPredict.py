from torchvision import transforms as T
import models
from PIL import Image
import numpy as np
import torch


class NetPredict:
    def __init__(self, arch, cls_num, model_path, use_gpu):
        self.use_gpu = use_gpu
        self.transform_test = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = models.init_model(name=arch, num_classes=cls_num)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        if self.use_gpu:
            self.model = self.model.cuda()

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform_test(img)
        img = img.unsqueeze(0)
        if self.use_gpu:
            img = img.cuda()
        output = self.model(img)
        if self.use_gpu:
            np_outputs = output[0].data.cpu().data.numpy()
        else:
            np_outputs = output[0].detach().numpy()
        np_outputs_sorted_idx = np.argsort(-np_outputs, axis=1)
        rank5 = np_outputs_sorted_idx[:, :5]
        return rank5.flatten()
