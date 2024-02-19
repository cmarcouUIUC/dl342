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
    for i, data in enumerate(train_data):
        train_inputs, train_labels = data

    for i, data in enumerate(valid_data):
        valid_inputs, valid_labels = data

    #Put data on device
    train_inputs=train_inputs.to(device)
    train_labels=train_labels.to(device)
    valid_inputs=valid_inputs.to(device)
    valid_labels=valid_labels.to(device)

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
    train_accuracy = []

    for epoch in range(n_epochs):
      permutation = torch.randperm(train_inputs.size(0))

      #Iterate
      for it in range(0,len(permutation)-batch_size+1, batch_size):
        batch_samples = permutation[it:it+batch_size]
        batch_data= train_inputs[batch_samples]
        batch_label= train_labels[batch_samples]

        #Compute Loss
        o = model(batch_data)
        loss_val = loss(o, batch_label.int())
        
        train_logger.add_scalar('train/loss', loss_val, global_step=global_step)
        train_accuracy.append(accuracy(o, batch_label.int()))
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        #Increase global step
        global_step += 1

      #Evaluate Model
      valid_pred = model(valid_inputs)
      valid_accuracy = accuracy(valid_pred, valid_labels.int())
      
      #train_logger.add_scalar('train/accuracy', np.mean(train_accuracy), global_step=global_step)
      valid_logger.add_scalar('valid/accuracy', valid_accuracy, global_step=global_step)
    
    #Save Model
    save_model(model)
    print(valid_accuracy)
    print(train_accuracy)


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
