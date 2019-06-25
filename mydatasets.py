import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset


def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

	if model_type == 'MLP':
		data = pd.read_csv(path)
		data['y'] = data['y']-1

		labels = data['y'].values
		data = data.iloc[:,0:-1].values
		
		print('Data Shape:', np.shape(data))
		print('Label Shape', np.shape(labels)) 

		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)

		dataset = TensorDataset(torch.Tensor(data).type(torch.FloatTensor), torch.Tensor(labels).type(torch.LongTensor))
	elif model_type == 'CNN':
		data = pd.read_csv(path)
		data['y'] = data['y']-1

		labels = data['y'].values
		data = data.iloc[:,0:-1].values
		
		print('Data Shape:', np.shape(data))
		print('Label Shape', np.shape(labels)) 

		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		
		dataset = TensorDataset(torch.Tensor(data).type(torch.FloatTensor).unsqueeze(1), torch.Tensor(labels).type(torch.LongTensor))

		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		# dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		data = pd.read_csv(path)
		data['y'] = data['y']-1

		labels = data['y'].values
		data = data.iloc[:,0:-1].values
		
		print('Data Shape:', np.shape(data))
		print('Label Shape', np.shape(labels)) 

		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		
		dataset = TensorDataset(torch.Tensor(data).type(torch.FloatTensor).unsqueeze(2), torch.Tensor(labels).type(torch.LongTensor))


		# data = torch.zeros((2, 2))
		# target = torch.zeros(2)
		# dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		# self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.

		# loc_seq = []
		# for patient_list in seqs:
		#     visits = len(patient_list)
		#     row_arr = []
		#     col_arr = []
		#     data = []
		#     for i,row in enumerate(patient_list):
		#         for j,col in enumerate(row):
		#             data.append(1)
		#             row_arr.append()

		#     loc_seq.append(csr_matrix((data, (row_arr, col_arr)), shape=(visits, num_features)))
			
		loc_seq = []
		for patient_list in seqs:
			indptr=[0]
			data = []
			indices = []
			for i,row in enumerate(patient_list):
				indices.extend(row)
				data.extend([1]*len(row))
				indptr.append(len(indices))
			loc_seq.append(sparse.csr_matrix((data, indices, indptr),shape=(len(patient_list), num_features), dtype = np.int_))

		# for matrix in loc_seq:
		#     print(matrix.toarray())
		self.seqs = loc_seq #[np.matrix(i) for i in seqs]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	# raise Exception(batch[0])
	batch.sort(reverse=True, key= lambda x : x[0].shape[0])
	max_length = batch[0][0].shape[0]
	num_features = batch[0][0].shape[1]

	seqs_tensor = torch.zeros([len(batch), max_length, num_features], dtype=torch.float)
	lengths_tensor = torch.zeros([len(batch)], dtype=torch.long)
	labels_tensor = torch.zeros([len(batch)], dtype=torch.long)

	# for minibatch in batch:

	for i in range(len(batch)):
		num_rows = batch[i][0].shape[0]
		lengths_tensor[i] = num_rows
		labels_tensor[i] = batch[i][1]

		coo = batch[i][0].tocoo()	
		seqs_tensor[i,0:num_rows] = torch.from_numpy(coo.toarray())

		# values = coo.data
		# indices = np.vstack((coo.row, coo.col))

		# i = torch.LongTensor(torch.from_numpy(indices))
		# v = torch.LongTensor(torch.from_numpy(values))
		# shape = coo.shape

		# seqs_tensor[i,0:num_rows]= torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


	# seqs_tensor = torch.FloatTensor()
	# lengths_tensor = torch.LongTensor()
	# labels_tensor = torch.LongTensor()

	return (seqs_tensor, lengths_tensor), labels_tensor
