import numpy as np
from data.setup_cifar import CIFAR
from data.setup_mnist import MNIST


def select_dataset(dataset_name, size):

	print("The selected dataset is: " + dataset_name + "\n")

	if dataset_name == 'cifar':
		data = CIFAR()
	else:
		if dataset_name == 'mnist':
			data = MNIST()
		else:
			print("\nNo dataset named " + dataset_name + " founded \n")
			exit()	

	x_test = data.test_data
	y_test = np.argmax(data.test_labels, axis=1)
	x_train = data.train_data
	y_train = np.argmax(data.train_labels, axis=1)
	x_test = x_test[0:size]
	y_test = y_test[0:size]
	#x_test = x_test.astype('float32')

	return x_test, y_test, x_train, y_train