#import "@template/setting:0.1.0": *

#show: doc => assignment(
		title: [ Machine Learning \ Home Assignment 6 ],
		doc
)

= Logistic regression in PyTorch

```python
class LogisticRegressionPytorch(nn.Module):
    def __init__(self, d, m):
      """
      d: input dimensionality
      m: number of classes (output dimensionality)
      """
      super(LogisticRegressionPytorch, self).__init__()
      self.fc = nn.Linear(d, m)                     # added
      self.activation_fn = nn.Sigmoid()             # added

    def forward(self, x):
      """
      x: input data
      """
      return self.activation_fn(self.fc(x))         # added

# definition of the loss function
criterion = nn.MultiMarginLoss()                    # added

# definition of the training
for epoch in range(no_epochs):  # Loop over the dataset multiple times
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward + backward + optimize
    outputs = logreg_pytorch(X_train_T)             # added
    loss = criterion(outputs, y_train_T)            # added
    loss.backward()                                 # added
    optimizer.step()

# getting the biases
bs_torch = logreg_pytorch.fc.bias.detach().numpy()  # added
```

While I don't get the same accuracy on the training and test data, I think the
implementation is correct, because the losses aren't too different and the
weights are initialized to random values.
 
= Convolutional Neural Networks

== Sobel filter

```python
# define convolutional layer
conv = nn.Conv2d(1, 2, W, padding = 0, bias=False)

# define the weights
simmetry = np.array([1, 0, -1])
g_x = torch.from_numpy(np.array([simmetry, 2 * simmetry, simmetry]))
params = torch.from_numpy(np.array([g_x, g_x.T])).reshape((2, 1, W, W))
conv.weight = torch.nn.Parameter(params.float(), requires_grad=False)

# apply the convolution
c = conv(x)

# combine the two channels to the final feature map
x2 = torch.sqrt(torch.sum(torch.square(c), dim=1))
```

So I defined the convolutional layer with 2 output channels. Considering the
input image @input, applying the convolutional layer results in two feature
maps, one for each channel, respectively @first-channel and @second-channel.
Finally, I combined the two channels to obtain the final feature map @combined,
and thus the Sobel filter.

#figure(
    image("img/input.png", width: 200pt),
    caption: "Input image",
) <input>

#figure(
    image("img/1.png", width: 200pt),
    caption: "First channel of the feature map"
) <first-channel>

#figure(
    image("img/2.png", width: 200pt),
    caption: "Second channel of the feature map"
) <second-channel>

#figure(
    image("img/3.png", width: 200pt),
    caption: "Combined feature map"
) <combined>

== Convolutional neural networks

```python
class Net(nn.Module):
  def __init__(self, img_size=28):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 5)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(64, 64, 5)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(1024, 43)

  def forward(self, x):
    x = self.pool1(F.elu(self.conv1(x)))
    x = self.pool2(F.elu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.elu(self.fc1(x))
    return x
```

== Augmentation

#ask()[Which transformations are applied to the input images during training?]
The `Data` class in `GTSRBTrafficSigns.py` applies the following transformations
to augment the dataset:
+ *Resize*: which makes all images the same size, 32x32 pixels.
+ *RandomAffine*: applies slight random rotation to the images.
+ *RandomCrop*: extracts random sections of the images.
+ *CenterCrop*: focuses on the center of the images.
+ *ColorJitter*: adjusts the brightness and contrast of the images.
+ *RandomHorizontalFlip*: flips the images horizontally.

#ask()[Why is a transformation conditioned on the label?]
Since the `RandomHorizontalFlip` transformation is conditioned on the label,
I suppose that there are some traffic signs that are symmetric, and flipping
them horizontally doesn't change their meaning, therefore you can augment the
dataset by adding the flipped images of such signs.\
I note that I don't understand how the labels are given to the signs, so I'm a
bit confused about providing a actual example.

#ask()[Please add at least one additional (not completely nonsensical)
transformation.]

For example you could add the `RandomPerspective` transformation, which changes
the perspective of the images, simulating the effect of viewing the signs from
different angles. Follows the code to add this transformation:

```python
  def __getitem__(self, index):
      image, label = self.dataset_train.__getitem__(index)
      image = transforms.Resize((self.img_width,self.img_height))(image)
      
      # These transformations just serve as examples
      if self.train:
        image = transforms.RandomAffine((-5,5))(image)
        image = transforms.RandomCrop((self.img_width_crop, self.img_height_crop))(image)
        image = transforms.ColorJitter(0.8, contrast = 0.4)(image)
        image = transforms.RandomPerpective(distortion_scale=0.2, p = 0.5)(image)
        if label in [11, 12, 13, 17, 18, 26, 30, 35]:
          image = transforms.RandomHorizontalFlip(p=0.5)(image)
      else:
        image = transforms.CenterCrop((self.img_width_crop, self.img_height_crop))(image)

      image = transforms.ToTensor()(image)

      return image, label
```

I think this transformation may be a reasonable augmentation, because weather or
people can rotate the traffic signs, therefore the perspective of the sign 
changes and we hope the drivers are still able distinguish the sign.  
Moreover, driving different vehicles or being in different positions on the road
can affect the perspective of the signs.

