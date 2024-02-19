from .models import ClassificationLoss, model_factory, save_model, load_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import tempfile
import numpy as np



def train(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_epochs=int(args.epochs)
    batch_size=int(args.batch_size)

    log_dir = tempfile.mkdtemp()

    train_logger = tb.SummaryWriter(log_dir+'/'+args.model+'/train', flush_secs=1)
    valid_logger = tb.SummaryWriter(log_dir+'/'+args.model+'/valid', flush_secs=1)

    #load data
    train_data=load_data('data/train')
    valid_data=load_data('data/valid')

    #split into data and labels
    #for i, data in enumerate(train_data):
    #    train_inputs, train_labels = data

    #for i, data in enumerate(valid_data):
    #    valid_inputs, valid_labels = data

    #Put data on device
    #train_inputs=train_inputs.to(device)
    #train_labels=train_labels.to(device)
    #valid_inputs=valid_inputs.to(device)
    #valid_labels=valid_labels.to(device)

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
    epoch_avg_acc = []
    epoch_avg_loss = []
    for epoch in range(n_epochs):
      
      train_accuracy = []
      train_loss = []
      for i, data in enumerate(train_data):
        inputs, labels = data
        optimizer.zero_grad()
        o = model(inputs)
        loss_val = loss(o, labels)
        train_loss.append(loss_val)
        train_accuracy.append(accuracy(o, labels))
        loss_val.backward()
        optimizer.step()

        
      epoch_avg_acc.append(np.mean(train_accuracy))
      #epoch_avg_loss.append(np.mean(train_loss))

    #Save Model
    save_model(model)
    print(epoch_avg_acc)
    #print(len(train_accuracy))


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
