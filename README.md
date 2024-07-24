# Vision-Networks-and-Fast-Training

In this assignment, we will take a closer look at some famous vision architectures.
Since most of these architectures are very large, it requires high-end hardware to train from scratch.
To leverage the limited availability of hardware, also *Transfer Learning* can be used. 
By using stored weights of a large network a new network can be trained cheaply on new datasets.

### Exercise 1: Hardware Acceleration 

In order to allow our computations to be accelerated,
the utility functions `evaluate` and `update` require some minor adjustments.

-------------------------------------------------------------------------------------------------------------
### Exercise 2: VGG for CIFAR-10 

Most vision architectures have been trained on the ImageNet dataset, which is hard to come by:
it is very large (a few 100GB) and requires registration to get access to the images.
[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
are similar datasets that are much easier to obtain
and they are one of the standard datasets in `torchvision.datasets`.
In this exercise the goal is to modify a vision network that was trained on ImageNet
to make predictions on CIFAR-10 so that we can reuse large parts of the weights.

-------------------------------------------------------------------------------------------------------------
### Exercise 3: Existing Features 

Training a network like VGG (or any of the other networks in this assignment)
can take a few hours when training on a GPU.
Therefore it is often useful to be able to load pre-trained weights into the network.
Also, saving a model that has been trained for hours can often save a lot of time.
In pytorch this is possible through what is called 
[`state_dict`s](https://pytorch.org/tutorials/beginner/saving_loading_models.html).
Saving the parameters of a pytorch module can be done with `torch.save(module.state_dict(), path)`,
whereas loading saved parameters is done with `module.load_state_dict(torch.load(path))`.

-------------------------------------------------------------------------------------------------------------
### Exercise 4: Training (part of) the Network 

Obviously, a classifier for CIFAR 10 will be different from a classifier for Imagenet.
With the initialisation above, the `CifarVGG` has a ready-to-go feature extractor,
but the classifier part still has to be trained.

-------------------------------------------------------------------------------------------------------------
### Exercise 5: Pre-Residual Networks 

The original and most commonly used residual networks actually do not implement skip connections as in the formula above.
Upon closer inspection (e.g. `torchvision.models.resnet`), it becomes clear that the most famous skip-connection looks more like

$$\boldsymbol{a} = \phi(\boldsymbol{x} + f(\boldsymbol{x})),$$

where $\phi$ is some non-linear activation function.
This non-linearity typically interferes with the signal propagation of the network.
As a result, gradients might still vanish despite the skip-connection.

Pre-Residual Networks aim to counter this problem by moving skip-connections to the level of pre-activations, such that

$$\boldsymbol{a} = \boldsymbol{x} + f(\phi(\boldsymbol{x})).$$

This way, clean signal propagation can be guaranteed and learning should become easier.
