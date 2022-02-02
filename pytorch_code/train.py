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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if you want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 20  # Store a model every snapshot epochs
lr = 1e-3  # Learning rate
modelName = 'STP'  # Options: C3D or R2Plus1D or R3D or P3D or T3D
dataset = 'ucf50'  # Options: hmdb51 or ucf101
num_classes = 3
# if dataset == 'hmdb51':
#     num_classes=51
# elif dataset == 'ucf101':
#     num_classes = 4
# else:
#     print('We only implemented hmdb and ucf datasets.')
#     raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

saveName = modelName + '-' + dataset


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
            :param test_interval:
            :param useTest:
            :param num_epochs:
            :param save_epoch:
            :param num_classes:
            :param save_dir:
            :param dataset:
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(3, 4, 6, 3))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(3, 4, 6, 3))
        train_params = model.parameters()
    elif modelName == 'P3D':
        model = p3d_model.P3D63(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'I3D':
        model = I3D_model.InceptionI3d(num_classes=num_classes, in_channels=3)
        train_params = model.parameters()
    elif modelName == 'T3D':
        model = T3D_model.inception_v1(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'STP':
        model = STP_model.STP(num_classes=num_classes, in_channels=3)
        train_params = model.parameters()
    else:
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = "./logs"
    writer = SummaryWriter(logdir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16, model_name=modelName),
                                  batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16, model_name=modelName),
                                batch_size=8, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16, model_name=modelName),
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
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(train_val_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    if not modelName == 'STP':
                        import torch.nn.parallel
                        outputs = nn.parallel.data_parallel(model, inputs, range(2))
                    else:
                        outputs, index = nn.parallel.data_parallel(model, inputs, range(2))
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                if modelName == 'I3D':
                    labels = labels.reshape(labels.shape[0], 1)
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)

                if modelName == 'STP':
                    sp_loss = -torch.log(
                        torch.sum(index[:, :, 0:int(index.size(2) / 2), :, :]) / int(index.size(2))) + torch.log(
                        1 - torch.sum(index[:, :, int(index.size(2) / 2) + 1:, :, :]) // int(index.size(2)))
                    loss = loss + sp_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_val_sizes[phase]
            epoch_acc = running_corrects.double() / train_val_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join('models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                if modelName == 'I3D':
                    labels = labels.reshape(labels.shape[0], 1)
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
