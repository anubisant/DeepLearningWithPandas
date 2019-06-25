import os
import pickle
import pandas as pd


PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	if icd9_str.startswith('E'):
		converted = icd9_str[0:4]
	else:
		converted = icd9_str[0:3]
	return converted


def build_codemap():
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"]).dropna()
	df_digits = df_icd9['ICD9_CODE'].apply(convert_icd9)
	codemap = {}
	i=0
	for unique_icd_digits in df_digits.unique():
		codemap[unique_icd_digits] = i
		i += 1
	# codemap = {123: 0, 456: 1}
	# print(i)
	return codemap


def create_dataset(path, codemap):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: 1. Load data from the three csv files
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv")) #? .dropna()
	df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
	df_diag = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv")).dropna()

	# TODO: 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'convert_icd9' you implemented and 'codemap'.
	df_diag['ICD9_CODE'] = df_diag['ICD9_CODE'].apply(convert_icd9)
	df_diag['ICD9_CODE'] = df_diag['ICD9_CODE'].map(codemap)
	df_diag = df_diag.dropna() 
	# print(df_diag[df_diag.isnull().any(axis=1)])

	# TODO: 3. Group the diagnosis codes for the same visit.
	merged_df = df_diag[['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE']].merge(df_admissions[['SUBJECT_ID','HADM_ID','ADMITTIME']], on=['SUBJECT_ID','HADM_ID']).sort_values(by=['SEQ_NUM'])

	grouped_diag = merged_df.groupby(['SUBJECT_ID','ADMITTIME'])['ICD9_CODE'].apply(list).reset_index().sort_values(by=['ADMITTIME'])
	

	# TODO: 4. Group the visits for the same patient.
	patient_diag_grouped = grouped_diag.groupby(['SUBJECT_ID'])['ICD9_CODE'].apply(list).reset_index()


	# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# TODO: Visits for each patient must be sorted in chronological order.
	seq_data = patient_diag_grouped['ICD9_CODE'].dropna().values.tolist()

	# TODO: 6. Make patient-id List and label List also.
	# TODO: The order of patients in the three List output must be consistent.
	patient_ids = list(patient_diag_grouped['SUBJECT_ID'].dropna().unique())
	labels = df_mortality.sort_values(by=['SUBJECT_ID'])['MORTALITY'].dropna().values.tolist()

	# patient_ids = [0, 1, 2]
	# labels = [1, 0, 1]
	# seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
	print("Len Patients: ", len(patient_ids), "Len labels: ", len(labels), "Len seq_data: ", len(seq_data))
	# print(patient_diag_grouped[patient_diag_grouped.isnull().any(axis=1)])
	# print(df_admissions[df_admissions.isnull().any(axis=1)])
	# print(df_mortality[df_mortality.isnull().any(axis=1)])
	# print(seq_data)
	# print(seq_data[seq_data.isnull().any(axis=1)])
	return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	codemap = build_codemap()
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
