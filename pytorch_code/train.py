import glob
import os
import timeit

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, I3D_model, T3D_model, STP_model

# Use GPU if available else revert to CPU
from pytorch_code.network import P3D_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model_name, dataset, save_dir, lr, num_epochs, save_epoch, useTest, n_test_interval, resume_epoch,
                save_name):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
            :param save_name:
            :param resume_epoch:
            :param lr:
            :param model_name:
            :param n_test_interval:
            :param useTest:
            :param num_epochs:
            :param save_epoch:
            :param save_dir:
            :param dataset:
    """

    train_ds = VideoDataset(dataset=dataset, split='train', clip_len=16, model_name=model_name)
    num_classes = train_ds.num_classes()

    if model_name == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif model_name == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(3, 4, 6, 3))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif model_name == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(3, 4, 6, 3))
        train_params = model.parameters()
    elif model_name == 'P3D':
        model = P3D_model.P3D63(num_classes=num_classes)
        train_params = model.parameters()
    elif model_name == 'I3D':
        model = I3D_model.InceptionI3d(num_classes=num_classes, in_channels=3)
        train_params = model.parameters()
    elif model_name == 'T3D':
        model = T3D_model.inception_v1(num_classes=num_classes)
        train_params = model.parameters()
    elif model_name == 'STP':
        model = STP_model.STP(num_classes=num_classes, in_channels=3)
        train_params = model.parameters()
    else:
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard cross-entropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(model_name))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', save_name + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', save_name + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = "./logs"
    writer = SummaryWriter(logdir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16, model_name=model_name),
                                batch_size=8, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16, model_name=model_name),
                                 batch_size=8, num_workers=4)

    train_val_loaders = {'train': train_dataloader, 'val': val_dataloader}
    train_val_sizes = {x: len(train_val_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(train_val_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    if not model_name == 'STP':
                        outputs = model(inputs)
                        # outputs = nn.parallel.data_parallel(model, inputs, range(2))
                    else:
                        outputs, index = model(inputs)
                        # outputs, index = nn.parallel.data_parallel(model, inputs, range(2))
                else:
                    with torch.no_grad():
                        if not model_name == 'STP':
                            outputs = model(inputs)
                        else:
                            outputs, index = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]

                if model_name == 'I3D':
                    labels = labels.reshape(labels.shape[0], 1)
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)

                if model_name == 'STP':
                    sp_loss = -torch.log(
                        torch.sum(index[:, :, 0:int(index.size(2) / 2), :, :]) / int(index.size(2))) + torch.log(
                        1 - torch.div(torch.sum(index[:, :, int(index.size(2) / 2) + 1:, :, :]), int(index.size(2)),
                                      rounding_mode='trunc'))
                    loss = loss + sp_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # scheduler.step() is to be called once every epoch during training
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / train_val_sizes[phase]
            epoch_acc = running_corrects / train_val_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % n_test_interval == (n_test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    if not model_name == 'STP':
                        outputs = model(inputs)
                    else:
                        outputs, index = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                if model_name == 'I3D':
                    labels = labels.reshape(labels.shape[0], 1)
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, num_epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


def main():
    num_epochs = 24  # Number of epochs for training
    resume_epoch = 22  # Default is 0, change if you want to resume
    use_test = True  # See evolution of the test set when training
    n_test_interval = 10  # Run on test set every n_test_interval epochs
    snapshot = 2  # Store a model every snapshot epochs
    lr = 1e-2  # Learning rate
    model_name = 'STP'  # Options: C3D or R2Plus1D or R3D or P3D or T3D
    dataset = 'hmdb51'  # Options: hmdb51 or ucf101

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    if resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

    save_name = model_name + '-' + dataset

    train_model(model_name=model_name, dataset=dataset, save_dir=save_dir, lr=lr, num_epochs=num_epochs,
                save_epoch=snapshot, useTest=use_test, n_test_interval=n_test_interval, resume_epoch=resume_epoch,
                save_name=save_name)


if __name__ == "__main__":
    main()
