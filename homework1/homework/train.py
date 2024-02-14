from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch


def train(args):
    n_epochs=5
    batch_size=64

    train_data=load_data('data/train')
    valid_data=load_data('data/valid')

    model = model_factory[args.model]()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss=ClassificationLoss()

    for epoch in range(n_epochs):
          permutation = torch.randperm(train_data.size(0))

          train_accuracy = []
          for it in range(0,len(permutation)-batch_size+1, batch_size):
            batch_samples = permutation[it:it+batch_size]
            batch_data= train_data[batch_samples][0]
            batch_label= train_data[batch_samples[1]]

            o = model(batch_data)
            loss_val = loss(0, batch_label.float())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
