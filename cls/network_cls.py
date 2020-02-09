import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import math

class net(nn.Module):
    def __init__(self, num_cls):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(net, self).__init__()
        model = models.resnet152(pretrained=True)

        #Stage-1, Freeze all the layers
        for i, param in model.named_parameters():
            param.requires_grad = False
        
        # Edit the primary net1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_cls)

        #Stage-2, Freeze all the layers till "layer 3"
        ct = []
        for name, child in model.named_children():
            if "layer3" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)
        
        self.resnet = model
        self.linear = nn.Linear(model.fc.in_features, num_cls)
        self.bn = nn.BatchNorm1d(num_cls, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)
        return features

