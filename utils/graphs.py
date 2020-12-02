import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy():
    x=np.arange(6)

    fig=plt.figure()
    fig.show()
    ax=fig.add_subplot(111)
    
    x = range(0,8)
    acc_ce = [68, 71, 80, 80, 81, 80, 83, 84]
    acc_dice = [55, 55, 55, 55, 55, 55, 55, 55]
    acc_focal  = [57, 57, 57, 57, 57, 57, 57, 57]
    acc_sm = [58, 58, 58, 58, 58, 58, 58, 58]

    ax.plot(x,acc_ce,c='b',marker="^",ls="-", label='CrossEntropy',fillstyle='none')
    ax.plot(x,acc_focal,c='g',marker=(8,2,0),ls="-", label='Focal')
    ax.plot(x,acc_dice,c='k', marker="X", ls='-',label='Dice')
    ax.plot(x,acc_sm,c='r',marker="v",ls='-',label='Las softmax')


    models = ['Naive','Skip','VGG \nFCN','VGG \nFCN 8','VGG \nFCN BN','Resnet \nDecoder','Resnet \nFCN8', "Resnet \nFCN8 (Aug)"]
    ax.set_xticks(range(0,8))
    ax.set_xticklabels(models)

    plt.title("Accuracy for different loss functions")
    plt.legend(loc=4, title='Loss functions')
    plt.savefig("accuracy.png")

if __name__ == "__main__":
    plot_accuracy()