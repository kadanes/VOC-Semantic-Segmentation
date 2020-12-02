import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch

from train import predict, load_model

color_dict = range(100, 1000, 25)
color_dict = [*color_dict][:21]
num_class = 21
# print("Color dict: ", color_dict)

# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/datasets/voc.py

palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
           [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
           [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

def labelVisualize(img):
    img_out = np.zeros(img.shape + (3,))

    for i in range(num_class):
        img_out[img == i] = palette[i]
    # print(type(img_out[0][0][0]))    
    return np.int_(img_out)

def visualizePrediction(model, images, labels, heatmap=False):

    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')


    try: 
        if (len(images.shape) == 3 and len(labels.shape) == 2):
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)

        image_count = len(images)


        # fig = plt.figure(figsize=(20, 20))

        preds = predict(model, images)

        ind = 1
        for i in range(image_count):
            plt.subplots(1, 3, figsize=(15, 15))  # specifying the overall grid size

            plt.subplot(1, 3, ind)
            # plt.imshow(images[i])

            # fig.add_subplot(image_count, 3, ind)
            plt.imshow(images[i])

            ind += 1

            # print("Label: ", labels[i])
            # print("Label Unique: ", np.unique(labels[i]))

            plt.subplot(1, 3, ind)
            # fig.add_subplot(image_count, 3, ind)

            if heatmap:
                plt.imshow(labels[i], cmap='hot')
            else:
                plt.imshow(labelVisualize(labels[i]))

            ind += 1

            if torch.cuda.is_available():
                # print(i, " unique: ", np.unique(preds[i].cpu().detach()))
                pred = preds[i].cpu().detach().numpy().argmax(0)
                # print(i, "after argmax unique: ", np.unique(pred))
            else:
                # print(i, " unique: ", np.unique(preds[i].detach()))
                pred = preds[i].detach().numpy().argmax(0)
                # print(i, "after argmax unique: ", np.unique(pred))

            # print("Pred: ", pred)
            # print("Label Unique Pred: ", np.unique(pred))
            plt.subplot(1, 3, ind)
            # fig.add_subplot(image_count, 3, ind)

            # plt.imshow(pred, cmap='hot')
            if heatmap:
                plt.imshow(pred, cmap='hot')
            else:
                plt.imshow(labelVisualize(pred))
            ind = 1

        plt.show()
    except Exception as e:
        print(e)

def visualizeModels(model_list, images,labels):
    # https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    
    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

    try: 
        if (len(images.shape) == 3 and len(labels.shape) == 2):
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)

        image_count = len(images)
        preds = []

        nrow = image_count
        ncol = len(model_list) + 2
        gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0,  top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

        fig, ax = plt.subplots(figsize=(ncol+1, nrow))  # specifying the overall grid size

        for model in model_list:
            preds.append(predict(model, images))
        preds = np.array(preds)
        ind = 1

        for i in range(image_count):

            # plt.subplot(image_count, len(model_list)+2, ind)
            # plt.imshow(images[i])
            # ind += 1

            ax = plt.subplot(gs[i,0])
            ax.imshow(images[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # plt.subplot(image_count, len(model_list)+2, ind)
            # plt.imshow(labelVisualize(labels[i]))
            # ind += 1

            ax = plt.subplot(gs[i,1])
            ax.imshow(labelVisualize(labels[i]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            for j,model in enumerate(model_list):
                if torch.cuda.is_available():
                    pred = preds[j][i].cpu().detach().numpy().argmax(0)
                else:
                    pred = preds[j][i].detach().numpy().argmax(0)
                    
                # plt.subplot(image_count, len(model_list)+2, ind)
                # plt.imshow(labelVisualize(pred))
                # ind += 1
                ax = plt.subplot(gs[i,j+2])
                ax.imshow(labelVisualize(pred))
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            # ind = 1
        # fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig("model_comparision_[" + str(time.time()) + "].png")
        plt.show()
    except Exception as e:
        print(e)


def compare_model_performance(name, voc2012, ind = range(0, 5)):
    
    try: 
        print("Name:", name)

        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            torch.cuda.manual_seed(0)
        else:
            torch.manual_seed(0)
        model,_ ,_ = load_model(name)
 
        visualizePrediction(model, voc2012.train_images[ind], voc2012.train_labels[ind])
        visualizePrediction(model, voc2012.val_images[ind], voc2012.val_labels[ind])
    
    except Exception as e:
        print(e)