from .models import ClassificationLoss, model_factory, save_model, load_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import tempfile
import numpy as np

logger = tb.SummaryWriter('model_logs')


def train(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_epochs=int(args.epochs)
    batch_size=int(args.batch_size)

    #load data
    train_data=load_data('data/train')
    valid_data=load_data('data/valid')

    #create the model
    if args.load==True:
      model = load_model(model_factory[args.model])().to(device)
    else:
      model = model_factory[args.model]().to(device)

    #Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum=args.momentum, weight_decay=1e-4)

    #Create the loss
    loss=ClassificationLoss()

    #Start Training
    global_step=0
    best_loss = 0
    epochs_no_improve = 0
    for epoch in range(n_epochs):
      
      train_accuracy = []
      train_loss = []
      for i, data in enumerate(train_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        o = model(inputs)
        loss_val = loss(o, labels)
        train_loss.append(loss_val.cpu().detach().numpy())
        train_accuracy.append(accuracy(o, labels).cpu().detach().numpy())
        loss_val.backward()
        optimizer.step()

      valid_accuracy = []
      valid_loss = []
      for i, data in enumerate(valid_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        valid_o = model(inputs)
        loss_val = loss(valid_o, labels)
        valid_loss.append(loss_val.cpu().detach().numpy())
        valid_accuracy.append(accuracy(valid_o, labels).cpu().detach().numpy())
      
      logger.add_scalar('model/train/acc', np.mean(train_accuracy), global_step)
      logger.add_scalar('model/train/loss', np.mean(train_loss), global_step)
      logger.add_scalar('model/valid/acc', np.mean(valid_accuracy), global_step)
      logger.add_scalar('model/valid/loss', np.mean(valid_loss), global_step)
      
      global_step+=1

      if epoch <=20:
        if epoch ==1:
          save_model(model)
        elif np.mean(valid_loss) <= best_loss:
          best_loss=np.mean(valid_loss)
          save_model(model)
      else:
        if np.mean(valid_loss) >= best_loss:
            epochs_no_improve+=1
        else:
            epochs_no_improve=0
            best_loss=np.mean(valid_loss)
            save_model(model)

      if epochs_no_improve==10:
          break
      
      prior_val_loss=np.mean(valid_loss)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-mod', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-b', '--batch_size', default=128)
    parser.add_argument('-e', '--epochs', default=1)
    parser.add_argument('-lr', '--learning_rate', default=.01)
    parser.add_argument('-mom', '--momentum', default=.9)
    parser.add_argument('-lo','--load',default=False)

    

    args = parser.parse_args()
    train(args)
