from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNNClassifier().to(device)


    #load data
    train_data=load_data('data/train')
    valid_data=load_data('data/valid')

    #loss
    loss = ClassificationLoss()

    #initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)

    global_step=0

    for epoch in range(args.n_epochs):

      train_acc = []
      for i,data in enumerate(train_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        o = model(inputs)
        #print(labels.shape)
        #print(o.shape)
        loss_val = loss(o, labels)
        train_accuracy=accuracy(o, labels).cpu().detach().numpy()
        train_acc.append(train_accuracy)
        train_logger.add_scalar('loss', loss_val, global_step)
        loss_val.backward()
        optimizer.step()
        global_step+=1
      train_acc = torch.mean(torch.cat(train_acc))
      train_logger.add_scalar('accuracy', train_acc, global_step)

      valid_acc = []
      for i,data in enumerate(valid_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        valid_o = model(inputs)
        valid_accuracy=accuracy(valid_o, labels).cpu().detach().numpy()
        valid_acc.append(valid_accuracy)
      valid_acc = torch.mean(torch.cat(valid_acc))
      valid_logger.add_scalar('accuracy', valid_acc, global_step)

      




    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--n_epochs', default=1)

    args = parser.parse_args()
    train(args)
