import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input,target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear((64*64*3),6)


    def forward(self, x):

        return self.linear(x.view(x.size(0),-1))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        layers.append(linear1=torch.nn.Linear((64*64*3),6))
        layers.append(activation=torch.nn.ReLU())
        layers.append(torch.nn.Linear(6, 6))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
       return self.linear(x.view(x.size(0),-1))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
