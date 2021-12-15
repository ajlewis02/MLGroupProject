# TODO
* This is just a rough draft and needs to be revised at some point!
* Most of the intro is lifted from the proposal, and it may need to be reworked
* Add your models! the penguins are lonely :(
* Might need to come up with a better name
# Dynamic Pruning
Our class project, Dynamic Pruning, is about designing and testing a new pruning method designed to both speed up training and improve model performance.
## Introduction
Overparameterized neural networks are computationally expensive to train and use. One way to solve this problem is to prune a trained neural net by deleting specific weights. Pruning in this way decreases the parameter count of the neural net, which decreases the required storage space and the computational resources required to make predictions. However, this method requires that the neural net be trained while fully connected [4]. 

A recent paper, “Lottery Ticket Hypothesis,” [1] introduced a methodology to find the “winning ticket” in a fully-connected network, which is a subnetwork with ~90% fewer parameters and similar or in some cases higher accuracy than the original network. The methodology used by [1] is an iterative pruning technique where after training the network p% of the weights with a value below a threshold are removed, after the pruning process the weights are reset to their initial value and the process is again repeated. These winning tickets are faster to train than their fully-connected counterparts, but because the process of creating the winning tickets requires training the neural net many times it does not actually result in a faster overall training time.

We propose a pruning method where instead of resetting the weights to their initial value after pruning we carry on the training with the trained parameter values. We hypothesize that this method will tend to take lesser epochs to converge and reduce overall training time while creating a smaller network.
## Implementation
For this project, we are using the Pytorch library to create, train, prune, and evaluate models. We also use the matplotlib library to automatically generate graphs.
## Testing
In order to test our pruning method, we have implemented our pruning method on models designed for four different datasets. We also implemented other pruning methods on these same models, so as to compare our method to existing pruning methods. The datasets we used are:
* The [Penguin Dataset](https://allisonhorst.github.io/palmerpenguins/), a dataset containing body measurements for over 300 different penguins labeled by species
# Penguins
The Penguin Dataset is a dataset containing body measurements of penguins and is labeled by species. The problem to be solved with a neural net is to classify a penguin's species given:
* The island from which the penguin was found
* The bill length of the penguin, in mm
* The bill depth of the penguin, in mm
* The penguin's fipper length in mm
* The penguin's mass in g
* The penguin's sex
* The year in which the penguin was recorded

The penguin's species could be one of:
* Adelie
* Gentoo
* Chinstrap
## Data Preprocessing
In order to facilitate training, the data was lightly preprocessed. The labels and classification data dimensions (island, sex, and year) were converted into one-hot vectors, and the other data dimensions were normalized by a constant value to ensure they were in a 0-1 range. Finally, we ignored any entries which had a value of NA for any field.

After preprocessing, the data was of the format:
```
0,1,0|0,1,0,1.022,0.825,1.125,1.05,1,0,0,0,1
1,0,0|0,0,1,0.7120000000000001,0.875,0.955,0.635,0,1,0,0,1
1,0,0|0,0,1,0.8220000000000001,0.905,1.025,0.86,1,0,0,1,0
```
This data was saved in the [Processed Penguins](./penguins_processed.txt) file. The first 200 entries in this file was the training data, the next 100 were the validation set, and the remaining 33 were the test set.
## Model Design
The model used for this dataset was a simple feed-forward neural net with two layers. The first layer was a Linear layer with 12 inputs and 12 outputs, and ReLU activation. The second layer was a Linear layer with 12 inputs and 3 outputs, and Tanh activation. We chose cross-entropy loss as this model's loss function, and Adam as this model's optimizer. The model was defined in Pytorch as:
```python
class PenguinModel(nn.Module):
    def __init__(self):
        super(PenguinModel, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.stack1 = nn.Linear(12, 12)
        self.stack2 = nn.Linear(12, 3)

    def forward(self, x):
        return self.tanh(self.stack2(self.relu(self.stack1(x))))
```
## Trials
We put this model through four different trials in order to test our pruning method. Each time the model was trained, it was trained for 4,000 epochs with a training rate of 0.0001.
### Unpruned Model
First, we trained and tested this model without any pruning at all. On the test data, the model performed with a loss of 0.4699.
![Unpruned Model Training Performance](./penguin_graphs/penguins_unpruned.png)

(Note that there were 4 batches per epoch, and performance during training was recorded per epoch - 4 units on the x-axis of these graphs corresponds to 1 epoch of training.)
### Pruned Model
Next, we tested the model after pruning the 30% lowest weights from the model. After pruning, the model performed on the test data with a loss of 0.4812.

This trial is defined by training a model normally and then pruning the fully-trained model. Because the training part is untouched, we reused the model trained from the first trial.
### Lottery Ticket
Third, we tested the performance of the model using the Lottery Ticket method, by resetting the weights and biases of the model to its original initialization without resetting the pruning. The model performed with a loss of 0.4672 on the test data.
![Lottery Ticket Model Training Performance](./penguin_graphs/penguins_lottery.png)
### Dynamic Pruning Model
Finally, we reset the model in order to test our pruning method. For this model, we pruned the 17% lowest weights from the model 1/3rd and 2/3rds of the way through training, for a total of just over 30% of the weights removed. The model performed with a loss of 0.4289 on the test data.

![Dynamic Pruning Model Training Performance](./penguin_graphs/penguins_novel.png)
## Running The Code
In order to run all of these tests at once, simply run [penguin_all.py](./penguin_all.py). It requires Pytorch and MatplotLib, and assumes that the computer running it has a GPU. Note that the program will create a file, ``penguin_checkpoint.pt`` within its working directory.

If your device does not have a GPU, or if you don't want to use your GPU, change line 8,
```python
device = "cuda"
```
to:
```python
device = "cpu"
```
## Penguin Conclusions
Using the unpruned model performance as a baseline, we can note improved performance from both the lottery ticket method and our dynamic pruning method. The model produced by normal pruning seemed to perform slightly worse than the baseline.

Due to time and resource constraints, the models could not be trained until perfect convergence. However, the unpruned and lottery ticket models seem to have very nearly converged by the end of the 4,000 epochs. However, the model trained with dynamic pruning seems to have not converged within 4,000 epochs, which suggests that dynamic pruning improves a model's capability to learn.
# Fashion MNIST

## Model Metrics Provided in Colab Pages

### Unpruned Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GOZwJFO1r4FI8VvrhQCYka0oQaW7tait?usp=sharing)

### Pruned Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U9uL8PVFfAk6sELFoTGyGu8GbZVfQcnD?usp=sharing)

### Lottery Ticket
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DEN0E8ZHVD_D3MezIch6noNJ5wddqFLc?usp=sharing)

### Dynamic Pruning Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cL2z82Brv3aPKEPvJFbGldJIRYnEzrDg?usp=sharing)

### Fashion MNIST Conclusions
For larger models, pruning was more of a detriment to model train time than helping short term. Increasing training/pruning time for more accurate model would help, but speed of model train time would suffer as a result.

The model was not trained to convergance in order to see how well the result performed under time constraint. In the future, training to convergance may provide insight on how well the pruning methods would fair under regular conditions.


# Flowers
## Data Preprocessing
## Model Design
## Trials
### Unpruned Model
### Pruned Model
### Lottery Ticket
### Dynamic Pruning Model
## Running The Code
## Flowers Conclusions
# CIFAR-10
## Data Preprocessing
## Model Design
## Trials
### Unpruned Model
### Pruned Model
### Lottery Ticket
### Dynamic Pruning Model
## Running The Code
## CIFAR-10 Conclusions
# Conclusion
# References
[1] Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. arXiv: Learning.

[2] Frankle, J., Dziugaite, G.K., Roy, D.M., & Carbin, M. (2021). Pruning Neural Networks at Initialization: Why are We Missing the Mark? ArXiv, abs/2009.08576.

[3] Tanaka, H., Kunin, D., Yamins, D.L., & Ganguli, S. (2020). Pruning neural networks without any data by iteratively conserving synaptic flow. ArXiv, abs/2006.05467.

[4] Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. (2020). What is the State of Neural Network Pruning? ArXiv, abs/2003.03033.
