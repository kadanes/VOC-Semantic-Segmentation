import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from criterion.CrossEntropy import getCrossEntropyLoss
from model.Naive import Naive
import numpy as np

def train(model, voc2012, model_name, optimizer=None, start_epoch=0, criterionType="ce", weighted=False, ignore=False, num_epochs=5, batch_size=64, learning_rate=1e-3, weight_decay=1e-5):

    train_labels = voc2012.train_labels
    eval_frequency = 10

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if criterionType == "ce":
        criterion = getCrossEntropyLoss(train_labels, weighted, ignore)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        torch.cuda.manual_seed(0)
        model.cuda()
    else:
        torch.manual_seed(0)

    model_name = model_name + "_" + criterionType
    if criterionType == "ce":
        if weighted:
            model_name += "_weighted"
        if ignore:
            model_name += "_ignore"

    log = open("./model/" + model_name + ".log", "w+")

    model_arch = str(model)
    criterion_details = str(criterion)

    log.write(model_arch + "\n")
    log.write("Criterion: " + criterion_details + "\n")  # Try to also list passed parameters
    log.write("Optimizer: " + str(optimizer) + "\n")  # Try to also list passed parameters
    log.write("Learning Rate: " + str(learning_rate) + "\n")
    log.write("Weight Decay: " + str(weight_decay) + "\n")
    log.write("Batch Size: " + str(batch_size) + "\n")
    log.write("Epochs: " + str(num_epochs) + "\n")
    log.write("-----------------------\n")
    log.close()

    model.train()

    for epoch in range(start_epoch, num_epochs):
        for start in range(0, len(voc2012.train_images), batch_size):

            end = min(start + batch_size, len(voc2012.train_images))
            batch_train_images = voc2012.train_images[start:end]
            batch_train_labels = voc2012.train_labels[start:end]

            batch_train_images, batch_train_labels = torch.FloatTensor(batch_train_images).permute(0, 3, 1, 2), torch.LongTensor(batch_train_labels)

            if cuda_avail:
                batch_train_images, batch_train_labels = batch_train_images.cuda(), batch_train_labels.cuda()

            optimizer.zero_grad()
            segments = model(batch_train_images)
            loss = criterion(segments, batch_train_labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 30 == 0:
            # torch.save(model, "./model/checkpoint/" + model_name + "_e" + str(epoch+1) + '.pt')
            save_model(model, model_name, optimizer, epoch, save_epoch=True)

        log = open("./model/" + model_name + ".log", "a+")
        loss_status = 'Epoch:{}, Loss:{:.4f} '.format(epoch+1, float(loss))

        if (epoch + 1) % eval_frequency == 0:
            loss_status += evaluate(model, voc2012, criterion)

        log.write(loss_status + "\n")
        log.close()
        print(loss_status)

    # torch.save(model, "./model/" + model_name + '.pt')
    save_model(model, model_name, optimizer, num_epochs)

# X: N, C, H, W
def predict(model, X):
    X = torch.FloatTensor(X).permute(0, 3, 1, 2)
    if torch.cuda.is_available():
      X = X.cuda()

    pred = model(X)
    return pred

def evaluate(model, voc2012, criterion, split="Val", batch_size=64, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    # if split == 'val':
    #     loader = val_loader
    # elif split == 'test':
    #     loader = test_loader
    n_iter = 4
    itr = 1

    for start in range(0, len(voc2012.val_images), batch_size):

        end = min(start + batch_size, len(voc2012.train_images))
        batch_eval_images = voc2012.val_images[start:end]
        batch_eval_labels = torch.LongTensor(voc2012.val_labels[start:end])

        if torch.cuda.is_available():
            batch_eval_labels = batch_eval_labels.cuda()

        output = predict(model, batch_eval_images)

        loss += criterion(output, batch_eval_labels).data

        if torch.cuda.is_available():
            # print(i, " unique: ", np.unique(preds[i].cpu().detach()))
            pred = output.cpu().detach().numpy().argmax(0)
            batch_eval_labels = batch_eval_labels.cpu().detach().numpy()
            # print(i, "after argmax unique: ", np.unique(pred))
        else:
            # print(i, " unique: ", np.unique(preds[i].detach()))
            pred = output.detach().numpy().argmax(0)
            batch_eval_labels = batch_eval_labels.detach().numpy()
            # print(i, "after argmax unique: ", np.unique(pred))

        correct += np.sum(pred == batch_eval_labels)
        n_examples += batch_eval_images.shape[0]

        if itr == n_iter:
            break
        itr += 1

    loss /= n_examples
    acc = 100. * correct / n_examples

    eval_log = '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(split, loss, correct, n_examples, acc)

    # if verbose:print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         split, loss, correct, n_examples, acc))

    return eval_log


def save_model(model, model_name, optimizer, epoch, save_epoch=False):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    if not save_epoch:
        torch.save(checkpoint, "./model/" + model_name + '.pt')
    else:
        torch.save(checkpoint, "./model/checkpoint/" + model_name + "_e" + str(epoch+1) + '.pt')


def load_model(name):
    naive = Naive()
    optimizer = torch.optim.Adam(naive.parameters(), lr=1e-3, weight_decay=1e-5)

    if torch.cuda.is_available():
        checkpoint = torch.load("./model/" + name)
        start_epoch = checkpoint['epoch']
        naive.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        checkpoint = torch.load("./model/" + name, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        naive.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return naive, optimizer, start_epoch


