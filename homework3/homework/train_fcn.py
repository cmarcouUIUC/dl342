import torch
import numpy as np

from .models import FCN, save_model,  ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = FCN().to(device)

    if args.seed is not None:
      torch.manual_seed(args.seed)
      np.random.seed(args.seed)


    #load data
    train_data=load_dense_data('dense_data/train')
    valid_data=load_dense_data('dense_data/valid')

    #loss
    loss = ClassificationLoss()

    #initialize optimizer
    if args.optim == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)
    elif args.optim == 'ADAM':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)


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
      c=ConfusionMatrix()
      for i,data in enumerate(train_data):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).long()

        optimizer.zero_grad()
        o = model(inputs)
        loss_val = loss(o, labels)
        c.add(preds=o.argmax(1),labels=labels)

        #track accuracy, iou, and log loss
        accuracies.append(accuracy(o,labels).detach().cpu().numpy())
        #train_acc.append(accuracy(o, labels).cpu().detach().numpy())
        train_logger.add_scalar('accuracy', c.global_accuracy, global_step)
        train_logger.add_scalar('loss', loss_val, global_step)
        train_logger.add_scalar('IOU', c.iou, global_step)


        loss_val.backward()
        optimizer.step()
        global_step+=1

      #scheduler.step()
      if scheduler is not None: scheduler.step(np.mean(accuracies))
      #log accuracy

      #check on valid accuracy
      valid_acc = []
      valid_loss = []
      c2=ConfusionMatrix()
      for i,data in enumerate(valid_data):
        model.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        valid_o = model(inputs)
        valid_l = loss(valid_o, labels)
        c2.add(preds=valid_o.argmax(1),labels=valid_l)
        valid_loss.append(valid_l.cpu().detach().numpy())
      #log validation accuracy
      valid_logger.add_scalar('accuracy', c2.global_accuracy, global_step)
      valid_logger.add_scalar('loss', np.mean(valid_loss), global_step)
      valid_logger.add_scalar('IOU', c2.iou, global_step)


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


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

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
