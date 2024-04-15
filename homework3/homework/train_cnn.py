from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
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
      
      #train loop
      for i,data in enumerate(train_data):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        o = model(inputs)
        loss_val = loss(o, labels)

        #track accuracy and log loss
        train_logger.add_scalar('accuracy', accuracy(o, labels),global_step)
        #train_acc.append(accuracy(o, labels).cpu().detach().numpy())
        train_logger.add_scalar('loss', loss_val, global_step)

        loss_val.backward()
        optimizer.step()
        global_step+=1
      
      #log accuracy

      #check on valid accuracy
      valid_acc = []
      valid_loss = []
      for i,data in enumerate(valid_data):
        model.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        valid_o = model(inputs)
        valid_l = loss(o, labels)

        valid_acc.append(accuracy(valid_o, labels).cpu().detach().numpy())
        valid_loss.append(valid_l.cpu().detach().numpy())
      #log validation accuracy
      valid_logger.add_scalar('accuracy', np.mean(valid_acc), global_step)
      valid_logger.add_scalar('accuracy', np.mean(val_loss), global_step)

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
      
      prior_val_loss=np.mean(valid_loss)




    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=10)


    args = parser.parse_args()
    train(args)