# import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.figure(1)
	plt.title('Loss Curve')
	plt.plot(range(0, len(train_losses)), train_losses,label='Training Loss')
	plt.plot(range(0, len(valid_losses)), valid_losses,label='Validation Loss')
	# plt.ylim(ymax=.8) 
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	plt.figure(2)
	plt.title('Accuracy Curve')
	plt.plot(range(0, len(train_accuracies)), train_accuracies,label='Training Accuracy')
	plt.plot(range(0, len(valid_accuracies)), valid_accuracies,label='Validation Accuracy')	
	# plt.ylim(ymax=.8) 
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

	pass


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# print(results)
	plt.figure()
	cmap=plt.cm.Blues
	cm = confusion_matrix(list(zip(*results))[0], list(zip(*results))[1])
	np.set_printoptions(precision=2)

	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title('Normalized Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	fmt = '.2f' 
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True')
	plt.xlabel('Predicted')
	plt.tight_layout()


	plt.show()

	pass
