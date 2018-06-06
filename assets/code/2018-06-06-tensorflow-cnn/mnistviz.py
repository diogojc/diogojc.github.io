import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train = mnist.train
test = mnist.test
validation = mnist.validation

nexamples = 9
examples = train.next_batch(nexamples)[0].reshape((nexamples, 28, 28))
plt.gray()
fig, axes = plt.subplots(3, 3)
axes[0, 0].imshow(examples[0])
axes[0, 1].imshow(examples[1])
axes[0, 2].imshow(examples[2])
axes[1, 0].imshow(examples[3])
axes[1, 1].imshow(examples[4])
axes[1, 2].imshow(examples[5])
axes[2, 0].imshow(examples[6])
axes[2, 1].imshow(examples[7])
axes[2, 2].imshow(examples[8])
plt.show()

fig, axes = plt.subplots(3, 3)
axes[0, 0].plot(examples[0].reshape((28*28)))
axes[0, 1].plot(examples[1].reshape((28*28)))
axes[0, 2].plot(examples[2].reshape((28*28)))
axes[1, 0].plot(examples[3].reshape((28*28)))
axes[1, 1].plot(examples[4].reshape((28*28)))
axes[1, 2].plot(examples[5].reshape((28*28)))
axes[2, 0].plot(examples[6].reshape((28*28)))
axes[2, 1].plot(examples[7].reshape((28*28)))
axes[2, 2].plot(examples[8].reshape((28*28)))
plt.show()
