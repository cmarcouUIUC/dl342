from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    n_epochs=100
    batch_size=64

    train_logger = tb.SummaryWriter(log_dir+'/'+args.model+'/train', flush_secs=1)
    valid_logger = tb.SummaryWriter(log_dir++'/'+args.model+'/valid', flush_secs=1)

    train_data=load_data('data/train')
    valid_data=load_data('data/valid')
    for i, data in enumerate(train_data):
        train_data, train_labels = data
    for i, data in enumerate(valid_data):
        valid_data, valid_labels = data

    model = model_factory[args.model]()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss=ClassificationLoss()

    for epoch in range(n_epochs):
          permutation = torch.randperm(train_data.size(0))

          train_accuracy = []
          for it in range(0,len(permutation)-batch_size+1, batch_size):
            batch_samples = permutation[it:it+batch_size]
            batch_data= train_data[batch_samples]
            batch_label= train_labels[batch_samples]

            o = model(batch_data)
            loss_val = loss(0, batch_label.float())
            
            train_accuracy.extend(((o > 0).long() == batch_label).cpu().detach().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1
          valid_pred = net2(valid_data) > 0

          valid_accuracy = float((valid_pred.long() == valid_labels).float().mean())
          train_logger.add_scalar('train/accuracy', np.mean(train_accuracy), global_step=global_step)
          valid_logger.add_scalar('valid/accuracy', valid_accuracy, global_step=global_step)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')

    # Put custom arguments here

    args = parser.parse_args()
    train(args)
