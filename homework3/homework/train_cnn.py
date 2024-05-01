from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np

def train(args):
    from os import path

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNNClassifier(norm=args.norm, residual=args.residual_connections).to(device)

    if args.seed is not None:
      torch.manual_seed(args.seed)
      np.random.seed(args.seed)
      




    #load data
    train_data=load_data('data/train', resize=args.resize, random_rotate=args.random_rotate, random_crop=args.random_crop, random_horizontal_flip=args.random_horizontal_flip, color_jitter=args.color_jitter, normalize=args.normalize_input,  is_resnet=args.is_resnet)
    valid_data=load_data('data/valid', resize=args.resize, random_rotate=args.random_rotate, random_crop=args.random_crop, random_horizontal_flip=args.random_horizontal_flip, color_jitter=args.color_jitter, normalize=args.normalize_input,  is_resnet=args.is_resnet)

    #loss
    loss = ClassificationLoss()

    #initialize optimizer
    if args.optim == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)
    elif args.oprtim == 'ADAM':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)


    scheduler = None
    if args.lr_schedule is not None:
      if args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
      elif args.lr_schedule =='plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
      else: scheduler = None

    global_step=0
    best_loss = 1000000
    epochs_no_improve=0

    for epoch in range(args.n_epochs):
      
      #train loop
      accuracies=[]
      for i,data in enumerate(train_data):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        o = model(inputs)
        loss_val = loss(o, labels)

        #track accuracy and log loss
        accuracies.append(accuracy(o,labels).detach().cpu().numpy())
        train_logger.add_scalar('accuracy', accuracy(o, labels),global_step)
        #train_acc.append(accuracy(o, labels).cpu().detach().numpy())
        train_logger.add_scalar('loss', loss_val, global_step)

        loss_val.backward()
        optimizer.step()
        global_step+=1
      #scheduler.step()
      if scheduler is not None: scheduler.step(np.mean(accuracies))
      #log accuracy

      #check on valid accuracy
      valid_acc = []
      valid_loss = []
      for i,data in enumerate(valid_data):
        model.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        valid_o = model(inputs)
        valid_l = loss(valid_o, labels)

        valid_acc.append(accuracy(valid_o, labels).cpu().detach().numpy())
        valid_loss.append(valid_l.cpu().detach().numpy())
      #log validation accuracy
      valid_logger.add_scalar('accuracy', np.mean(valid_acc), global_step)
      valid_logger.add_scalar('loss', np.mean(valid_loss), global_step)

      if epoch <= args.early_stop:
        if epoch == 1:
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
      
      #prior_val_loss=np.mean(valid_loss)




    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--norm', default=False)
    parser.add_argument('--is_resnet',default=False)
    parser.add_argument('--resize',default=None)
    parser.add_argument('--normalize_input',default=False)
    parser.add_argument('--random_horizontal_flip',default=False)
    parser.add_argument('--random_crop', default=False)
    parser.add_argument('--color_jitter', default=False)
    parser.add_argument('--residual_connections',default=False)
    parser.add_argument('--random_rotate',default=False)    
    parser.add_argument('--lr_schedule',default=None)
    parser.add_argument('--optim',default='SGD')
    parser.add_argument('--seed',default=None, type=int)







    args = parser.parse_args()
    train(args)