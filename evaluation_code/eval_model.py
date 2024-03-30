from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib 
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import code
import os, sys
import tensorflow as tf
import keras
import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile

import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import KFold
from scipy import interp

from itertools import cycle
#import seaborn as sns
from scipy import stats 

import argparse
import datetime

TIME_GAP = 5

filename = ""
analysis_df = None
is_given_unknownset = False
is_given_confidence_thres = False
given_confidence_thres = float(0)
model_verbose = False
given_epoch = 0

learning_time_list = []
learning_end_time_list = []
learning_start_time_list = []

def show_confusion_matrix(validations, predictions):
	matrix = metrics.confusion_matrix(validations, predictions)
	print (matrix)
	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

	#plt.figure(figsize=(18, 12))
	sns.heatmap(matrix,
				cmap="coolwarm",
				linecolor='white',
				linewidths=1,
				#xticklabels=LABELS,
				#yticklabels=LABELS,
				annot=False,
				fmt="d")
	plt.title("Confusion Matrix")
	plt.ylabel("True Label")
	plt.xlabel("Predicted Label")
	plt.show()
	fname_fig = filename + "_confusion.pdf"
	#plt.savefig('res.pdf')
	plt.savefig(fname_fig)

	print (matrix)


def get_links(fname):
	with open(fname, 'r') as f:
		links = f.read().splitlines()
	links = list(map(lambda x: x.split(','), links))
	targets = list(map(lambda x: x[0], links))
	return links, targets

def get_file_list(file_path):
	# argument check
	print("Read logs from the given dir: " + str(file_path))

	# extract csv files in 
	dirname = file_path.split('/')
	full_dirname = ""
	for name in dirname:
		if name != "" and name != ".":
			full_dirname = "./{}/".format(name)
			break
	print (full_dirname)
	
	if os.path.isdir(full_dirname):
		file_list = os.listdir(full_dirname)
		file_list.sort()
		#print (file_list)
	else:
		print("given dir is not directory namea!: " + str(dirname))
		sys.exit()

	return full_dirname, file_list

def extract_file_info(given_file_name):
	splited = given_file_name.split('_')
	#print (splited)

	time_gap  = int(splited[-2][0:-3])
	label = splited[0]
	if len(splited) == 4:
		label = label + "_" + splited[1]

	#print (time_gap, label)
	return time_gap, label

def convert_to_int(x):
	try:
		return np.int(x)
	except:
		#return np.nan
		return np.int(0)

def extract_log(df, n_time, n_features, max_value):
	logs = list()

	print ("given n_time: ", n_time)

	for index, rows in df.iterrows():
		vlist = rows[2:].tolist()
				
		if len(vlist) != n_time:
			print ("Too short!!!\n\n\n")
			sys.exit()
		
		logs.append(vlist)

	if (max_value == -1):
		# print (logs)
		# print (max(logs))
		# print (max((max(logs))))
		max_value = float(max(max(logs)))
		print ("max value: ", max_value)

	reshaped_x = np.asarray(logs, dtype=np.float64).reshape(-1, n_time, n_features)
	reshaped_x = reshaped_x.astype('float64')
	np.nan_to_num(reshaped_x, copy=False)
	reshaped_x = np.divide(reshaped_x, max_value, dtype='float64')

	if not model_verbose:
		print ("shape: ", reshaped_x.shape)
		print (reshaped_x.shape[0], " samples")  
			
	return reshaped_x, max_value
	

################################
# Read Train Data
################################
def augmenting_train_data(trainX, trainY, df_train, max_timestep):
	# new_trainX = np.asarray(trainX)
	# new_trainY = np.asarray(trainY)

	# copy the original data
	new_trainX = trainX.copy()
	new_trainY = trainY.copy()
	new_df_train = df_train.copy()

	min_timestep = 40
	augmentation_gap = 40

	for logged_time in range (min_timestep, max_timestep, augmentation_gap):
		print (logged_time)
		for l_idx in range(len(trainX)):
			temp_trainx = trainX[l_idx].copy()
			aug_label = trainY[l_idx] + "_" + str(logged_time)
			# padding
			for t_idx in range(logged_time, max_timestep):
					temp_trainx[t_idx] = 0

			# add to train set
			new_trainX = np.append(new_trainX, temp_trainx.reshape(1,max_timestep,1), axis=0)						
			new_trainY.append(aug_label)

			# add to dataframe
			new_pd = pd.DataFrame({'Label':[aug_label]})
			new_df_train = new_df_train.append(new_pd)
			
	#code.interact(local=locals())
	
	return new_trainX, new_trainY, new_df_train

################################
# Read Train Data
################################
def read_train_data(file_path, video_list_fname, max_timestep):
	print ("\n==================================================\nRead Train Data")
	print ("===================================================\n")

	video_list, targets = get_links(video_list_fname)
	print (targets)
	full_dirname, file_list = get_file_list(file_path)
	#print (file_list)

	df = pd.DataFrame()
	trainY = list()
	trainY_name = list()
	
	for file_name in file_list:
		full_filename = os.path.join(full_dirname + file_name)
		#print("Currnet file: " + file_name + " / " + full_filename)

		time_gap, label = extract_file_info(file_name)
		if time_gap != TIME_GAP:
			continue
		
		if full_filename.endswith(".xlsx") and label in targets:
			print("processing " + full_filename + "....")
			
			# read xlsx file 
			# second sheet would be filtered_Traffic sheet
			read_df = pd.read_excel(full_filename, sheet_name="filered_traffic")

			#print (read_df.shape)
			
			# merge df & ADD LABEL
			df = pd.concat([df, read_df])

			# create label (first column contains log name)
			print (label)
			label_list = [label] * len(read_df.index)
			trainY.extend(label_list)
			trainY_name.append(read_df['Time'].values.tolist())

	# insert label
	df.insert(0, "Label", trainY, True)

	# shuffle
	df = df.sample(frac=1).reset_index(drop=True)
	trainY = df['Label'].values.tolist()
	
	# strip & normalize 
	# df.dropna(axis = 1, how = 'any', inplace=True)
	df.dropna(axis = 1, how = 'all', inplace=True)
	df.fillna(0)

	n_features = 1   # hardcoded
	time_steps = df.shape[1] - 2  # discount label, filename column
	print (df.shape)
	print ("time_steps: ",time_steps)
	print ("max_timestep: ", max_timestep)

	if time_steps > max_timestep:
		droplist = list(range(time_steps - max_timestep))
		droplist = list(map(lambda x:x+max_timestep+2, droplist))
		#print (droplist)
		df = df.drop(df.columns[droplist], axis = 1)
		time_steps = max_timestep

	# Make trainX (extract logs & reshaping)
	trainX, max_value  = extract_log(df, time_steps, n_features, -1)
	#print (trainX)
	
	# # Write to the excel file
	# tot_input_data_fname = file_path[:-1] + ".xlsx"
	# print (tot_input_data_fname)
	# writer = ExcelWriter(tot_input_data_fname)
	# df.to_excel(writer,'Sheet1',index=False)
	# writer.save()
	
	# one-hot encoding for preparing trainY
	label_encoder = preprocessing.LabelEncoder()
	# trainY = label_encoder.fit_transform(np.asarray(trainY))

	print ("(train data) DF shape: ", df.shape)    

	return trainX, trainY, targets, label_encoder, max_value, df

################################
# Read Test Data
################################
def read_test_data(file_path, video_list_fname, label_encoder, max_value, max_timestep):
	print ("\n==================================================\nRead Test Data")
	print ("===================================================\n")

	video_list, targets = get_links(video_list_fname)
	full_dirname, file_list = get_file_list(file_path)
	
	df = pd.DataFrame()
	testY = list()
	testY_name = list()
	
	for file_name in file_list:
		full_filename = os.path.join(full_dirname + file_name)
		#print("Currnet file: " + file_name + " / " + full_filename)

		time_gap, label = extract_file_info(file_name)
		if time_gap != TIME_GAP:
			continue
		
		if full_filename.endswith(".xlsx") and label in targets:
			print("processing " + full_filename + "....")
			
			# read xlsx file 
			# second sheet would be filtered_Traffic sheet
			read_df = pd.read_excel(full_filename, sheet_name="filered_traffic")
			
			# merge df & ADD LABEL
			df = pd.concat([df, read_df])

			# create label (first column contains log name)	
			print (label)
			label_list = [label] * len(read_df.index)
			testY.extend(label_list)
			testY_name.append(read_df['Time'].values.tolist())

	# insert label
	df.insert(0, "Label", testY, True)

	# shuffle
	df = df.sample(frac=1).reset_index(drop=True)
	testY = df['Label'].values.tolist()
	
	# strip & normalize 
	#df.dropna(axis = 1, how = 'any', inplace=True)
	df.dropna(axis = 1, how = 'all', inplace=True)
	df.fillna(0)


	n_features = 1   # hardcoded
	time_steps = df.shape[1] - 2  # discount label, filename column
	print ("time_steps: {} // max_timestep: {}".format(time_steps, max_timestep))

	if time_steps > max_timestep:
		droplist = list(range(time_steps - max_timestep))
		droplist = list(map(lambda x:x+max_timestep+2, droplist))
		#print (droplist)
		df = df.drop(df.columns[droplist], axis = 1)
		time_steps = max_timestep

	# Make testX (extract logs & reshaping)
	testX, max_value  = extract_log(df, time_steps, n_features, max_value)
		
	#Write to the excel file
	# tot_input_data_fname = file_path[:-1] + "_testinput.xlsx"
	# print (tot_input_data_fname)
	# writer = ExcelWriter(tot_input_data_fname)
	# df.to_excel(writer,'Sheet1',index=False)
	# writer.save()


	#code.interact(local=locals())


	print ("TEST DF shape: ", df.shape)    
	return testX, testY, df

def convert_to_one_hot_encoding(trainY, testY, label_encoder, has_unknown, test_n):
	original_testY = None

	# Change label to 'Unknown' before fitting label_encoder
	if has_unknown == "Unknown":
		# copy the original testY
		original_testY = list()

		# change label to unknown
		for idx, label_item in enumerate(testY):
			original_testY.append(testY[idx])
			testY[idx] = "Unknown"

	elif has_unknown == "Combined":
		# copy the original testY
		original_testY = list()

		#code.interact(local=locals())	

		# change label to unknown
		for idx, label_item in enumerate(testY):
			if (idx >= test_n):
				#print (idx, label_item)
				original_testY.append(testY[idx])
				testY[idx] = "Unknown"
			else:
				original_testY.append(testY[idx])			

		#print (testY)


	# Fitting label encoder 
	if (testY is not None):
		label_encoder.fit(np.asarray(trainY + testY))
	else:
		label_encoder.fit(np.asarray(trainY))
		
	print (list(label_encoder.classes_))

	# transform to numerical value 
	trainY = label_encoder.transform(np.asarray(trainY))

	# [Debug] label count information
	unique, counts = np.unique(trainY, return_counts=True)
	print ("(Train) Dataset information: (Label: Count)")
	print (dict(zip(unique, counts)))

	# convert to OHE
	trainY = np_utils.to_categorical(trainY, len(list(label_encoder.classes_)))
	# print (trainY)

	if (testY):
		# transform
		testY = label_encoder.transform(np.asarray(testY))	

		# label count information
		# with np.printoptions(threshold=np.inf):
		# 	print (testY)
		# 	print (label_encoder.inverse_transform(testY))

		unique_test, counts_test = np.unique(testY, return_counts=True)
		print ("(TEST) Dataset information: (Label: Count)")
		print (dict(zip(unique_test, counts_test)))

		# conver to OHE
		testY = np_utils.to_categorical(testY, len(list(label_encoder.classes_)))
	
	return trainY, testY, original_testY, label_encoder

def auc_computation(n_output, y_pred_test, max_y_pred_test, max_y_test, y_test):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	lw = 2

	# print (y_test[:, 0])
	# print (y_pred_test[:, 0])
	# print (len(y_test))

	for i in range(n_output):

		# res_y_test = list(range(len(y_test)))
		# for j in range(len(y_test)):
		# 	print (max_y_test[j])
		# 	print (max_y_pred_test[j])
		# 	if max_y_test[j] == max_y_pred_test[j]:
		# 		res_y_test[j] = 1
		# 	else:
		# 		res_y_test[j] = 0
			
		# new_y_test = np.asarray(res_y_test)
		# print (new_y_test)

		# # new_y_test[j] = 1
		# # new_y_test[j] = 0
		# print ("========== ", i)
		# print (new_y_test)
		# print (y_pred_test[:, i])
		# fpr[i], tpr[i], _ = roc_curve(new_y_test , y_pred_test[:, i])
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i] , y_pred_test[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
		
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_test.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_output)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_output):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])


	# Finally average it and compute AUC
	mean_tpr /= n_output

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# print(fpr[0])
	# print(tpr[0])

	print("Micro auc: ", roc_auc["micro"])
	print("macro auc: ", roc_auc["macro"])

	return (roc_auc["micro"], roc_auc["macro"])


	# Plot all ROC curves
	# plt.figure(1)
	# plt.plot(fpr["micro"], tpr["micro"],
	# 				label='micro-average ROC curve (area = {0:0.2f})'
	# 							''.format(roc_auc["micro"]),
	# 				color='deeppink', linestyle=':', linewidth=4)

	# plt.plot(fpr["macro"], tpr["macro"],
	# 				label='macro-average ROC curve (area = {0:0.2f})'
	# 							''.format(roc_auc["macro"]),
	# 				color='navy', linestyle=':', linewidth=4)

	# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	# for i, color in zip(range(4), colors):
	# 		plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	# 						label='ROC curve of class {0} (area = {1:0.2f})'
	# 						''.format(i, roc_auc[i]))

	# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Some extension of Receiver operating characteristic to multi-class')
	# plt.legend(loc="lower right")
	# plt.savefig("roc.pdf")

# for unknown set analysis
def print_current_confidence(y_pred_test, max_y_pred_test, label_encoder):
	confidence_y_pred_test = np.max(y_pred_test, axis=1)

	max_confidence = np.max(confidence_y_pred_test)
	min_confidence = np.min(confidence_y_pred_test)

	print (list(label_encoder.classes_))

	# for np_idx in range(y_pred_test.shape[0]):
	# 	if (confidence_y_pred_test[np_idx] > 0):
	# 		#code.interact(local=locals())
	# 		print ("Original Class: {} | Determined Class: {} ({}) | Confidence: {}".format(analysis_df.iloc[np_idx, 1], max_y_pred_test[np_idx], label_encoder.inverse_transform([max_y_pred_test[np_idx]]) , confidence_y_pred_test[np_idx]))

	print ("[Confidence Max: {} | Min: {}".format(max_confidence, min_confidence))		

def print_classificiation_report(n_output, y_pred_test, max_y_pred_test, max_y_test, label_encoder):
	confidence_y_pred_test = np.max(y_pred_test, axis=1)
	max_y_pred_test_temp = copy.deepcopy(max_y_pred_test)
	print ("Given confidence thres: {}".format(given_confidence_thres))

	if is_given_unknownset:
		max_idx = label_encoder.transform(["Unknown"])
	else:
		max_idx= n_output
	
	for np_idx in range(y_pred_test.shape[0]):
		if (confidence_y_pred_test[np_idx] <= given_confidence_thres):
			max_y_pred_test_temp[np_idx] = max_idx

	print(classification_report(max_y_test, max_y_pred_test_temp))

def confidence_evaluation(n_output, y_pred_test, max_y_pred_test, max_y_test, label_encoder):
	#for debugging
	#print (y_pred_test)
	confidence_y_pred_test = np.max(y_pred_test, axis=1)

	if is_given_unknownset:
		max_idx = label_encoder.transform(["Unknown"])
	else:
		max_idx= n_output

	print ("unknown Class IDX: {}".format(max_idx))

	accuracy_arr = list()
	precision_arr = list()
	recall_arr = list()
	confidence_thres_arr = list()
	
	for confidence_thres_np in np.arange(0, 1.001, 0.001):
		confidence_thres = float(confidence_thres_np)
		for np_idx in range(y_pred_test.shape[0]):
			if (confidence_y_pred_test[np_idx] <= confidence_thres):
				max_y_pred_test[np_idx] = max_idx

		# print("\n--- Classification report for test data (threshold: %f---\n" % confidence_thres)
		# print(classification_report(max_y_test, max_y_pred_test))
		# print(confidence_y_pred_test)
		# print("accuracy_score: {}".format( accuracy_score(max_y_test, max_y_pred_test)))
		# print("precision_score: {}".format( precision_score(max_y_test, max_y_pred_test, average='weighted')))
		# print("recall_score: {}".format( recall_score(max_y_test, max_y_pred_test, average='weighted')))

		
		accuracy_arr.append(accuracy_score(max_y_test, max_y_pred_test))
		precision_arr.append(precision_score(max_y_test, max_y_pred_test, average='weighted', zero_division=1))
		recall_arr.append(recall_score(max_y_test, max_y_pred_test, average='weighted', zero_division=1))
		confidence_thres_arr.append(confidence_thres)

		# accuracy_arr.append(accuracy_score(max_y_test, max_y_pred_test))
		# precision_arr.append(precision_score(max_y_test, max_y_pred_test, average='macro', zero_division=1))
		# recall_arr.append(recall_score(max_y_test, max_y_pred_test, average='macro', zero_division=1))
		# confidence_thres_arr.append(confidence_thres)

	print ("confidence_y_pred_test")
	print (confidence_y_pred_test)
	print ("========================")
	#print (max_y_pred_test)
	#print (max_y_test)
	#print(classification_report(max_y_test, max_y_pred_test))
	print_precision_recall(accuracy_arr,precision_arr, recall_arr, 	confidence_thres_arr)


def one_d_cnn_model(trainX, trainY, testX, testY, 
										given_model, 
										do_kfold_validation, do_confidence_evaluation, do_failure_analysis, label_encoder):
	n_time_steps, n_features = trainX.shape[1], trainX.shape[2]
	n_output = trainY.shape[1]
	print ("TrainX,Y infos")
	print ("\tInput: n_time_steps: %d, n_features: %d" % (n_time_steps, n_features))
	print ("\tOutput: n_output: ", n_output)


	# code.interact(local=locals())
	

	##################################
	# Consturct Model
	##################################
	
	model = Sequential();

	#model 5 (Beauty and burst model + long window size 10sec)
	if given_model == 5:
		print ("Given model is ", given_model)
		if (TIME_GAP > 99):
			window_size = int ( 10000 / TIME_GAP )
			second_window_size = int (window_size/2)
		else:
			window_size = int (10 / TIME_GAP)
			second_window_size = int (window_size/2)

		print ("Window size is ", window_size)

		model.add(Conv1D(50, window_size, activation='relu', input_shape=(n_time_steps, n_features)))
		model.add(Conv1D(50, second_window_size, activation='relu'))
		model.add(Dropout(0.5))
		model.add(MaxPooling1D(5))
		model.add(Dropout(0.7))
		model.add(Flatten())
		model.add(Dense(100, activation='relu'))
		model.add(Dense(n_output, activation='softmax'))

	#model 555 (Beauty and burst model + long window size 10sec)
	elif given_model == 555:
		print ("Given model is ", given_model)
		if (TIME_GAP > 99):
			window_size = int ( 10000 / TIME_GAP / 2)
			second_window_size = int (window_size/2)
			third_window_size = int (second_window_size)
		else:
			window_size = int (10 / TIME_GAP)
			second_window_size = int (window_size/2)

		print ("Window size is ", window_size)

		model.add(Conv1D(50, window_size, activation='relu', input_shape=(n_time_steps, n_features)))
		model.add(Dropout(0.3))
		model.add(MaxPooling1D(3))	
		model.add(Conv1D(100, second_window_size, activation='relu'))
		model.add(Dropout(0.3))
		model.add(MaxPooling1D(3))
		model.add(Conv1D(100, third_window_size, activation='relu'))
		model.add(Dropout(0.3))
		model.add(MaxPooling1D(3))
		model.add(Flatten())
		model.add(Dense(400, activation='relu'))
		model.add(Dense(n_output, activation='softmax'))


	#model 60 new
	elif given_model == 60:
		print ("Given model is ", given_model)
		if (TIME_GAP > 99):
			window_size = int ( 10000 / TIME_GAP)
			second_window_size = int (window_size/2)
			third_window_size = int (second_window_size)
		else:
			window_size = int (10 / TIME_GAP)
			second_window_size = int (window_size/2)

		print ("Window size is ", window_size)

		model.add(Conv1D(150, window_size, activation='relu', input_shape=(n_time_steps, n_features)))
		model.add(Dropout(0.4))
		model.add(MaxPooling1D(3))	
		model.add(Conv1D(50, second_window_size, activation='relu'))
		model.add(Dropout(0.3))
		model.add(MaxPooling1D(3))
		model.add(Conv1D(50, third_window_size, activation='relu'))
		model.add(Dropout(0.3))
		model.add(MaxPooling1D(3))
		model.add(Flatten())
		model.add(Dense(500, activation='relu'))
		model.add(Dense(n_output, activation='softmax'))

	# netflix model
	elif given_model == 55:
		print ("Given model is ", given_model)
		if (TIME_GAP > 99):
			window_size = int ( 60000 / TIME_GAP )
			second_window_size = int (window_size/2)
		else:
			window_size = int (10 / TIME_GAP)
			second_window_size = int (window_size/2)

		print ("Window size is ", window_size)

		model.add(Conv1D(50, window_size, activation='relu', input_shape=(n_time_steps, n_features)))
		model.add(Conv1D(50, second_window_size, activation='relu'))
		model.add(Dropout(0.5))
		model.add(MaxPooling1D(5))
		model.add(Dropout(0.7))
		model.add(Flatten())
		model.add(Dense(100, activation='relu'))
		model.add(Dense(n_output, activation='softmax'))	

	else:
		print ("Wrong!")
		sys.exit()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

	if not model_verbose:
		print(model.summary())

	##################################
	# Fit model
	##################################
	print ("\nFit the model\n")

	# hardcoded for now
	##BATCH_SIZE = 64
	BATCH_SIZE = 32
	# BATCH_SIZE = 20
	EPOCHS = 800
	#EPOCHS = 1
	VERBOSE = 1
	
	#m_epochs = EPOCHS
	m_epochs = given_epoch
	m_batch_size = BATCH_SIZE

	print ("Epoch: ", m_epochs)
	print ("Batch size: ", m_batch_size)

	if do_kfold_validation:
		# turn off the verbose option during k-fold validation
		learning_start_time = datetime.datetime.now()
		history = model.fit(trainX,
							trainY,                    
							batch_size=m_batch_size,
							epochs=m_epochs,
							#validation_split=0.2,
							#callbacks=callbacks_list,
							verbose=1)
		learning_end_time = datetime.datetime.now()
	else:
		learning_start_time = datetime.datetime.now()
		history = model.fit(trainX,
							trainY,                    
							batch_size=m_batch_size,
							epochs=m_epochs,
							verbose=1)
		learning_end_time = datetime.datetime.now()

	#model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose) 

	fit_time =  (learning_end_time - learning_start_time)
	print (fit_time)
	learning_time_list.append(fit_time)
	learning_start_time_list.append(learning_start_time)
	learning_end_time_list.append(learning_end_time)

	##################################
	# save model
	##################################
	if save_model_name is not "No name":
		model.save(save_model_name)


	##################################
	# evaluate model
	##################################
	print ("testX shape; ", testX.shape)
	print ("testY shape: ", testY.shape)

	# evaluate  
	score = model.evaluate(testX, testY, verbose=1)
	print (model.metrics_names)

	print("\nAccuracy on test data: %0.5f" % score[1])
	print("\nLoss on test data: %0.5f" % score[0])

	# Take the class with the highest probability from the test predictions	
	y_pred_test = model.predict(testX)
	max_y_pred_test = np.argmax(y_pred_test, axis=1)
	max_y_test = np.argmax(testY, axis=1)

	# Check only if the unknown testset is given 
	if is_given_unknownset:
		print ("\n ---- y_pred_test result (unknown set)\n")
		print_current_confidence(y_pred_test, max_y_pred_test, label_encoder)
	else:
		print_current_confidence(y_pred_test, max_y_pred_test, label_encoder)

	if (is_given_confidence_thres):
		print("\n--- Classification report for test data with Given Confidence Thres---\n")
		print_classificiation_report(n_output, y_pred_test, max_y_pred_test, max_y_test, label_encoder)

	else:
		print("\n--- Classification report for test data ---\n")
		print(classification_report(max_y_test, max_y_pred_test))

	# # Do failure analysis 
	# failure_analysis(n_output, max_y_pred_test, max_y_test)

	# extract metrics 
	roc_micro, roc_macro = auc_computation(n_output, y_pred_test, max_y_pred_test, max_y_test, testY)
	precision_s = precision_score(max_y_test, max_y_pred_test , average="macro")
	recall_s = recall_score(max_y_test, max_y_pred_test , average="macro")
	f1_s = f1_score(max_y_test, max_y_pred_test , average="macro")
	print("precision_score: ",  precision_s)
	print("recall_score: ", recall_s)
	print("f1_score: ", f1_s) 

	if do_failure_analysis:
		failure_analysis(n_output, max_y_pred_test, max_y_test)


	if do_kfold_validation:
		# during K-fold validation, skip to save confusion matrix and classification report
		acc_s = score[1]
		loss_s = score[0]
		return (acc_s, loss_s, precision_s, recall_s, f1_s, roc_micro, roc_macro)

	else:	
		if not model_verbose:
			show_confusion_matrix(max_y_test, max_y_pred_test)

	##################################
	# evaluate model (Confidence)
	##################################
	if do_confidence_evaluation:
		confidence_evaluation(n_output, y_pred_test, max_y_pred_test, max_y_test, label_encoder)

	
	#code.interact(local=locals())

def failure_analysis(n_output, max_y_pred_test, max_y_test):

		## for debugging
		print ("\n\nmax_y_test")
		print (max_y_test)
		print ("\n\nmax_y_pred_test")
		print (max_y_pred_test)

		print ("\n\nn_ouput")
		print (n_output)

		print ("\n\nlen(max_y_pred_test)")
		print (len(max_y_pred_test))
		print (len(max_y_test))

		wrong_list = list()
		for idx in np.arange(0,len(max_y_pred_test),1):
			if max_y_pred_test[idx] != max_y_test[idx]:
				wrong_list.append(idx)

		print ("\n\nwrong_list")
		print (wrong_list)

		for idx in np.arange(0,len(wrong_list), 1):
			print (analysis_df.iloc[wrong_list[idx], 1])

	
def print_precision_recall(accuracy_arr, precision_arr, recall_arr, 	confidence_thres_arr):
	#plt.rcParams['axes.grid'] = True
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams['lines.markersize'] = 5
	#plt.rcParams["font.size"] = 8
	#plt.rc('font', family='serif', serif='Times')
	plt.rc('legend', fontsize=7)
	plt.rc('xtick', labelsize=8)
	plt.rc('ytick', labelsize=8)
	plt.rc('axes', labelsize=8)
	width = 3.487
	height = width / 1.618
	fig, ax = plt.subplots()
	fig.subplots_adjust(left=.15, bottom=.2, right=.99, top=.97)

	print ("confidence_thres =============")
	print (confidence_thres_arr)
	print ("\nAccuracy =============")
	print (accuracy_arr)
	print ("\nPrecision =============")
	print (precision_arr)
	print ("\nRecall =============")
	print (recall_arr)

	print ("max accuracy: {} // confidence_thres: {}".format(max(accuracy_arr), confidence_thres_arr[accuracy_arr.index(max(accuracy_arr))]))
	
	ax.set_xlabel("Confidence threshold")
	ax.set_ylabel("Average precision / recall value")
	plt.plot(confidence_thres_arr, precision_arr, 'r--', confidence_thres_arr, recall_arr, 'b:')
	ax.legend(['Precision', 'Recall'], loc='upper right')

	fig.set_size_inches(width, height)
	fname_fig = filename + "_precision_recall.pdf"
	fig.savefig(fname_fig)

def get_dataset_info(Y, label_encoder):
	print ("Dataset information")
	print (label_encoder.get_params())
	print (list(label_encoder.classes_))

def print_running_env():
	print ('\n================================')
	print ('tRunning Environmnet')
	print ('================================')
	print('\tPython version : ', sys.version)
	print('\tTensorFlow version : ', tf.__version__)
	print('\tKeras version : ', keras.__version__)
	print ('\n================================\n')


if __name__ == "__main__":
	#global analysis_df
	#global model_verbose

	###############################################################################
	# argument handling
	###############################################################################
	argparser = argparse.ArgumentParser(epilog="-d/--dir is required")
	argparser.add_argument("-d", "--dir", 
						   help="directory name", 
						   default="no_name", 
						   required=True)
	argparser.add_argument("-l" ,"--list",
						   help="video list",
						   default="no name",
						   required=True)
	argparser.add_argument("-g", "--granularity",
						   help="Time granularity",
						   default="5")
	argparser.add_argument("-t", "--testdir",
						   help="test set directory name",
						   default="no_name",
						   required=False)
	argparser.add_argument("-m", "--maxstep",
						   help="Max timestamp",
						   default=int(10000),
						   required=False)
	argparser.add_argument("-u", "--unknown",
						   help="Unknown video list",
						   default="no name",
						   required=False)
	argparser.add_argument("--aug",
						   help="Do augmentation",
						   action='store_true')

	argparser.add_argument("--save",
						   help="Save the model",
						   default="No name", 
							 required=False)

	argparser.add_argument("--load",
						   help="load the model",
						   default="No name", 
							 required=False)

	argparser.add_argument("-a", "--analysis", help="Failure analysis", action='store_true')
	argparser.add_argument("-c", "--cnnmodel", help="Model", default=int(1), required=False)
	argparser.add_argument("-k", "--kfold", help="Do K-fold validation", action='store_true')
	argparser.add_argument("-f", "--confidence", help="Do Confidence evaluation", action='store_true')
	argparser.add_argument("-v", "--verbose", help="Verbose", action='store_true')	
	argparser.add_argument("-e", "--epoch", help="Epoch number", default=int(500), required=False)
	argparser.add_argument("--confidence_thres", help="confidence threshold", default=float(0.0), required=False)
	argparser.add_argument("--combined_unknown", help="Evaluation with unknown + known dataset", action='store_true')
	args = argparser.parse_args()

	given_train_dir = args.dir
	video_list_name = args.list
	TIME_GAP = int(args.granularity)
	given_test_dir = args.testdir
	given_max_timestep = int(args.maxstep)
	given_model = int(args.cnnmodel)
	given_epoch = int(args.epoch)
	unknown_video_list_name = args.unknown
	is_combined_evaluation = args.combined_unknown
	
	if (args.confidence_thres):
		is_given_confidence_thres = True
		given_confidence_thres = float(args.confidence_thres)

	save_model_name = args.save
	load_model_name = args.load

	# augmentation
	do_augmentation = args.aug

	# Evalaution options
	do_kfold_validation = args.kfold
	do_confidence_evaluation = args.confidence
	do_failure_analysis = args.analysis

	model_verbose = args.verbose

	filename = given_train_dir[0:-1]

	#print_running_env()

	###############################################################################
	# Prepare the train / test datasets
	###############################################################################
	np.random.seed()

	trainX, trainY, targets, label_encoder, max_value, df_train  = read_train_data(given_train_dir,
												video_list_name, given_max_timestep)
	#get_dataset_info(trainY, label_encoder)
	
	if (do_augmentation):
		trainX, trainY, df_train = augmenting_train_data(trainX, trainY, df_train, given_max_timestep)
		print(df_train)

	if given_test_dir == "no_name": 
		sample_n = trainX.shape[0]
		train_n = int(sample_n * 0.8)
		test_n = sample_n - train_n

		if do_kfold_validation:
			print ("\n\n\nDo K-fold validation")

			(trainY, testY, original_testY, label_encoder) = convert_to_one_hot_encoding(trainY, None, label_encoder, None, 0)

			#kfold_validation = KFold(n_splits=5, shuffle=False)
			kfold_validation = KFold(n_splits=5, shuffle=True)
			accuracy_score_list = []
			loss_score_list = []
			precision_score_list = []
			recall_score_list = []
			f1_score_list =[]
			roc_micro_list = []
			roc_macro_list = []

			accuracy_score_sum = float(0)
			loss_score_sum = float(0)
			precision_score_sum = float(0)
			recall_score_sum = float(0)
			f1_score_sum = float(0)
			roc_micro_sum = float(0)
			roc_macro_sum = float(0)


			for train_idxs, validation_idxs in kfold_validation.split(trainX, trainY):
				print ("\nTrain / validation idx")
				print (train_idxs)
				print (validation_idxs)

				analysis_df = df_train.iloc[validation_idxs]

				(k_acc_s, k_loss_s, k_precision_s, k_recall_s, k_f1_s, k_roc_micro, k_roc_macro) = one_d_cnn_model(trainX[train_idxs], trainY[train_idxs], 
																		trainX[validation_idxs], trainY[validation_idxs], 
																		given_model, do_kfold_validation, do_confidence_evaluation, do_failure_analysis, label_encoder)

				# store the score information
				k_accuracy = k_acc_s
				accuracy_score_list.append(k_accuracy)
				accuracy_score_sum += float(k_accuracy)

				loss_score_list.append(k_loss_s)
				loss_score_sum += float(k_loss_s)

				precision_score_list.append(k_precision_s)
				precision_score_sum += float(k_precision_s)

				recall_score_list.append(k_recall_s)
				recall_score_sum += float(k_recall_s)

				f1_score_list.append(k_f1_s)
				f1_score_sum += float(k_f1_s)
				
				roc_micro_list.append(k_roc_micro)
				roc_micro_sum += float(k_roc_micro)

				roc_macro_list.append(k_roc_macro)
				roc_macro_sum += float(k_roc_macro)				

			# get Averaged value 
			accuracy_score_sum = float(accuracy_score_sum / 5)
			loss_score_sum = loss_score_sum / 5
			precision_score_sum = precision_score_sum / 5
			recall_score_sum = recall_score_sum / 5
			f1_score_sum = float(f1_score_sum / 5)
			roc_micro_sum = roc_micro_sum / 5
			roc_macro_sum = roc_macro_sum / 5

			# print final results
			print('\nK-fold cross validation Accuracy: {}'.format(accuracy_score_list))
			print('\nAvg of K-fold cross validation Accuracy: {}\n'.format(accuracy_score_sum))

			print('f1 score: {:0.4f}  /  list: {}'.format(f1_score_sum, f1_score_list))
			print('roc mirco: {:0.4f} /  list: {}'.format(roc_micro_sum, roc_micro_list))
			print('roc macro: {:0.4f} /  list: {}'.format(roc_macro_sum, roc_macro_list))
			print('precision: {:0.4f} /  list: {}'.format(precision_score_sum, precision_score_list))
			print('recall: {:0.4f} /  list:{}'.format(recall_score_sum, recall_score_list))

			for i in range(0, len(learning_time_list)):
				print (learning_time_list[i])

			for i in range(0, len(learning_start_time_list)):
				print (learning_start_time_list[i])

			for i in range(0, len(learning_end_time_list)):
				print (learning_end_time_list[i])

			sys.exit()
			# Finish after K-fold evaluation

		else:
			print ("\n\n\nValidate with randomly selected 20percent of data")

	
		(trainY, testY, original_testY, label_encoder) = convert_to_one_hot_encoding(trainY, None, label_encoder, None, 0)

		testX = trainX[train_n:]
		testY = trainY[train_n:]
		trainX = trainX[0:train_n]
		trainY = trainY[0:train_n]
		
		analysis_df = df_train[train_n:]

	else:
		if (unknown_video_list_name == "no name"):
			testX, testY, df_test = read_test_data(given_test_dir, video_list_name, label_encoder, max_value, trainX.shape[1])
			# convert Y to one hot encoding
			(trainY, testY, original_testY, label_encoder) = convert_to_one_hot_encoding(trainY, testY, label_encoder, None, 0)

		else:
			# Testing with unknown set
			is_given_unknownset = True
			testX, testY, df_test = read_test_data(given_test_dir, unknown_video_list_name, label_encoder, max_value, trainX.shape[1])
			
			print ("Read TEST done!!\n\n\n")
			# convert Y to one hot encoding
			testY[0] = "Unknown"
			
			# mix 
			if is_combined_evaluation:
				print ("\n\n\n Validate with randomly selected 20percent of Train data + Given Unknown data")
				
				sample_n = trainX.shape[0]
				train_n = int(sample_n * 0.8)
				test_n = sample_n - train_n

				# use 20 % for the test
				known_testX = trainX[train_n:]
				known_testY = trainY[train_n:]
				known_df = df_train[train_n:]
				
				trainX = trainX[0:train_n]
				trainY = trainY[0:train_n]

				testX = testX[0:test_n]
				testY = testY[0:test_n]

				# merge unknown dataset and known dataset into new test dataset
				new_testY = known_testY + testY  # list type
				new_testX = np.concatenate((known_testX, testX))  # np array type 
				testX = new_testX
				
				(trainY, testY, original_testY, label_encoder) = convert_to_one_hot_encoding(trainY, new_testY, label_encoder, "Combined", test_n)
				
				df_test = pd.concat([known_df, df_test])

			else:
				(trainY, testY, original_testY, label_encoder) = convert_to_one_hot_encoding(trainY, testY, label_encoder, "Unknown", 0)
		
		# shape check
		
		train_n = trainX.shape[0]
		test_n = testX.shape[0]        
		print ("after dataset info train_n:{} test_n:{}".format(train_n, test_n))
		
		analysis_df = df_test

		if testX.shape[1] != trainX.shape[1]:
			print ("test and train shape mismatch")
			print ("trainX shape; ", trainX.shape)
			print ("trainY shape: ", trainY.shape)
	
			print ("testX shape; ", testX.shape)
			print ("testY shape: ", testY.shape)

			sys.exit()
			
		
	print ("Main: Test/Train input shape information")
	print ("trainX shape; ", trainX.shape)
	print ("trainY shape: ", trainY.shape)
	
	print ("testX shape; ", testX.shape)
	print ("testY shape: ", testY.shape)

	

	one_d_cnn_model(trainX, trainY, testX, testY, 
									given_model, do_kfold_validation, do_confidence_evaluation, do_failure_analysis, label_encoder)


	## debugging
	#print (trainX[2])
	#print (df.iloc[2, 1])


	print (list(label_encoder.classes_))
	print ("Done\n\n")

	sys.exit()




