import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input,target)


class CNNClassifier(torch.nn.Module):

  #Set up block
  class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
      super().__init__()
      self.net = torch.nn.Sequential(
        #Strided Convolution
        torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
        #Non-lin activation
        torch.nn.ReLU(),
        #non-strided convolution
        torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
        #non-lin activation
        torch.nn.ReLU()
      )
    
    def forward(self, x):
        return self.net(x)
      
  def __init__(self, layers=[32,64,128], n_input_channels=3):
    super().__init__()
    #Initial convolution, larger kernel with padding and stride
    #Use maxpooling here to reduce dimensions/down-sample
    L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    c = 32
    #Add block for specified channels
    for l in layers:
        L.append(self.Block(c, l, stride=2))
        c = l
    #final network setup
    self.network = torch.nn.Sequential(*L)
    #classifier (6 outputs for 6 classification objects)
    self.classifier = torch.nn.Linear(c, 6)
    
  def forward(self, x):
    # Compute the features
    z = self.network(x)
    # Global average pooling along spatial dimensions
    z = z.mean(dim=[2,3])
    # Classify
    return self.classifier(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
