import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(model, voc2012, criterion, num_epochs=5, batch_size=64, learning_rate=1e-3, weight_decay=1e-5):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        torch.cuda.manual_seed(0)
        model.cuda()
    else:
        torch.manual_seed(0)


    model.train()

    for epoch in range(num_epochs):
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

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))

    # torch.save(model, "./model/naive" + '.pt')


def predict(model, image, label):
    pass
