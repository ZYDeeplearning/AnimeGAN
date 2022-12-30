from numpy.lib.arraysetops import isin
import torchvision.models as models
import torch.nn as nn
import torch
from PIL import Image
import numpy as np

vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).float()
#VGG19
class VGG19(nn.Module):
    def __init__(self,batch_norm=False,num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                    'M']
        self.batch_norm=batch_norm
        self.num_clases = num_classes
        self.features=self.make_layers(self.cfg,self.batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    def forward(self,x):
        module_list = list(self.features.modules())
        for l in module_list[1:27]:  # conv4_4
            x = l(x)
        return x
    def normalize_vgg(self, image):
        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = Image.open('/Users/mac/Downloads/AnimeGAN/1 (63).jpg')
    image = image.resize((256, 256))
    np_img = np.array(image).astype('float32')
    np_img = ((np_img/255)-0.5)/0.5
    img = torch.from_numpy(np_img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    vgg = VGG19()
    vgg.load_state_dict(torch.load('/Users/mac/Downloads/animeGAN.py/vgg19.pth'))
    print(vgg)
    feat = vgg(img)
    print(feat.shape)