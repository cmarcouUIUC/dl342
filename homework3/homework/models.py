import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input,target)


class CNNClassifier(torch.nn.Module):

  #Set up block
  class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, norm, residual, stride=1,):
      super().__init__()
      self.residual=residual
      self.net = torch.nn.Sequential(
        #Strided Convolution
        torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
        #Non-lin activation
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(n_output) if norm == True else torch.nn.Identity(),
        #non-strided convolution
        torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
        #non-lin activation
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(n_output) if norm == True else torch.nn.Identity()
      )
      self.downsample = None
      if n_input != n_output or stride!=1:
        self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, kernel_size=1),
        torch.nn.BatchNorm2d(n_output) if norm == True else torch.nn.Identity())
    
    def forward(self, x):
      identity=x
      if self.downsample is not None:
        identity = self.downsample(x)
      return self.net(x) + identity if self.residual == True else self.net(x)
      
  def __init__(self, norm=False, residual=False, layers=[32,64,128],  n_input_channels=3):
    super().__init__()
    #Initial convolution, larger kernel with padding and stride
    #Use maxpooling here to reduce dimensions/down-sample
    L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32) if norm == True else torch.nn.Identity(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    c = 32
    #Add block for specified channels
    for l in layers:
        L.append(self.Block(c, l, norm, residual, stride=2))
        c = l
    #final network setup
    self.network = torch.nn.Sequential(*L)
    #classifier (6 outputs for 6 classification objects)
    self.fcl = torch.nn.Linear(c,4096)
    self.classifier = torch.nn.Linear(4096, 6)
    
  def forward(self, x):
    # Compute the features
    z = self.network(x)
    # Global average pooling along spatial dimensions
    z = z.mean(dim=[2,3])
    z = self.fcl(z)
    # Classify
    return self.classifier(z)


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
