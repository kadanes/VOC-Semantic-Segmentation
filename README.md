# VOC-Semantic-Segmentation

In this project we experiment with different models for the task of semantic segmentaion on the VOC2012 datatset. We try a nive autoencorder architecture and move up from there to use a pretrained backbone. We also experiment with differnt loss functions. 

To see our different models please go to the `models` directory. In it there are all the models that we have trained along with the logs. This directory also has the model code.

If you want to train the models, then please look at the `train_parth.ipynb`. 

If you want to look at our visualizations , then please see the `results.ipynb` notebook.


A breif description of the `train` function:

`
def train(model_name, optimizer=None, start_epoch=0, criterionType="ce", weighted=False, ignore=False, augumented=False, num_epochs=5, batch_size=64, learning_rate=1e-3, weight_decay=1e-5):
`

- `mode_name` : Name of the model you want to train. Allowed values are: `["naive", "skip", "fcn", "fcn8", "fcn_2", "fcn_resnet_bn", "fcn_resnet_bn_skp"]`

- `optimizer`: Currently unused 

- `start_epoch`: Currently unused

- `criterionType`: The loss function to use. The supported options are: `['ce']`

- `weighted`: For cross entropy loss if weights of `1 - label_frequency` should be used 

- `ignore`: For cross entropy if label 0 should not be considered in loss calculation 

- `augmented`: If the training data should be augmented with center crop and scale with 50% probability 

- `num_epochs`: Number of epochs 

- `batch_size`: Batch size

- `learning_rate`: Learning rate

- `weight_decay`: Weight decay 


If you don't have the training data downloaded, then the train function will first download the data, unzip it, convert the data to h5 file for validation and train (with augmentation for train if true) and then delete the downloaded zip file. You will need about 10 GB of free space for this inital load. Afterwards the train method will directly load the h5 files that take significantly lesser space. 

We have taken help of [REFunction/VOC2012-Segmentation](https://github.com/REFunction/VOC2012-Segmentation) for the data loader code and modified it to support augmentation. 


# Training Performance 


![](https://github.com/parthv21/VOC-Semantic-Segmentation/blob/master/model_comparision_train.png)

# Valadation Performance 

![](https://github.com/parthv21/VOC-Semantic-Segmentation/blob/master/model_comparision_val.png)

> Model List:  `"/checkpoint/fcn_resnet_bn_skp_ce_augumented_e90.pt", "/checkpoint/fcn_resnet_bn_skp_ce_e90.pt", "/checkpoint/fcn_resnet_bn_ce_e120.pt", "/fcn_2_ce.pt", "/fcn8_ce.pt", "/fcn_ce.pt", "/skip_ce.pt", "/naive_ce.pt"`

