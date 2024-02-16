from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import tempfile
import numpy as np



def train(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_epochs=1000
    batch_size=128

    log_dir = tempfile.mkdtemp()

    train_logger = tb.SummaryWriter(log_dir+'/'+args.model+'/train', flush_secs=1)
    valid_logger = tb.SummaryWriter(log_dir+'/'+args.model+'/valid', flush_secs=1)

    train_data=load_data('data/train')
    valid_data=load_data('data/valid')

    for i, data in enumerate(train_data):
        train_data, train_labels = data

    train_data=train_data.to(device)
    train_labels=train_labels.to(device)

    for i, data in enumerate(valid_data):
        valid_data, valid_labels = data

    valid_data=valid_data.to(device)
    valid_labels=valid_labels.to(device)

    model = model_factory[args.model]().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss=ClassificationLoss()

    global_step=0
    for epoch in range(n_epochs):
          permutation = torch.randperm(train_data.size(0))

          train_accuracy = []
          for it in range(0,len(permutation)-batch_size+1, batch_size):
            batch_samples = permutation[it:it+batch_size]
            batch_data= train_data[batch_samples]
            batch_label= train_labels[batch_samples]

            o = model(batch_data)
            loss_val = loss(o, batch_label)
            
            train_logger.add_scalar('train/loss', loss_val, global_step=global_step)
    
            train_accuracy.extend(accuracy(o, batch_label))
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

          train_pred = model(train_data)
          valid_pred = model(valid_data)
          valid_accuracy = accuracy(valid_pred, valid_labels)
          #train_logger.add_scalar('train/accuracy', train_accuracy, global_step=global_step)
          valid_logger.add_scalar('valid/accuracy', valid_accuracy, global_step=global_step)

    save_model(model)
    print(valid_accuracy)
    print(train_accuracy)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')

    # Put custom arguments here

    args = parser.parse_args()
    train(args)
