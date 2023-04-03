import os
import sys
import keras
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf
import pandas as pd 
import numpy as np
from data.setup_cifar import CIFAR
import modules.output_util as ou
from sewar import *
import numpy as np 
from utils import my_utils as ut
import sklearn
from sklearn.metrics import *
from matplotlib import pyplot as plt
import pandas as pd
import modules.output_util as ou
import modules.dataset_selector as ds
import modules.model_selector as ms
import modules.detectors_selector as des
from sewar import *

ut.limit_gpu_usage()


dataset_name = "cifar"
model_name = "resnet50"

path = "./out_dir/rensnet50_compose.csv"
df = pd.read_csv(path)

data = CIFAR()
x_test = data.test_data
y_test = np.argmax(data.test_labels, axis=1)
x_train = data.train_data
y_train = np.argmax(data.train_labels, axis=1)


#LOAD CLASSIFIER AND DETECTORS_______________________________________________

classifier, model = ms.select_model(dataset_name, model_name)

magnet_detector, squeezer = des.select_detectors(dataset_name, model, model_name)
magnet_detector.train(x_train, y_train)
squeezer.train(x_train, y_train)

y_test_pred, y_pred_score = magnet_detector.test(x_test)
mag_fpr = ut.calculate_accuracy_bool(y_test_pred)
print("MagNet FPR calculated on the CIFAR-10 test set", mag_fpr)

y_test_pred, y_test_pred_score = squeezer.test(x_test)
squ_fpr = ut.calculate_accuracy_bool(y_test_pred)
print("Squeezer FPR calculated on the CIFAR-10 test set", squ_fpr)

rows = []

print(df)

for index, row in df.iterrows():
		
	if(row['success_rate'] != 1.0 or "miss" not in row['file_name']):

		x_adv = np.load("./out_dir/resnet50/" + row['csv_name'] + "/" + row['file_name'] + '_sample.npy')
		adv_idx = np.load("./out_dir/resnet50/" + row['csv_name'] + "/" + row['file_name'] + '_index.npy')

		x_test_succ = x_test[adv_idx]

		y_pred_mag, y_pred_score_adv = magnet_detector.test(x_adv)
		print(y_pred_mag.shape)
		print(y_pred_mag)
		y_pred_squ, y_att_pred_score = squeezer.test(x_adv)
		print(y_pred_squ.shape)
		print(y_pred_squ)
		
		for i in range(len(x_adv)):

			metrics = []

			new_row = pd.Series(pd.Series([row['attack'], row['file_name'], row['csv_name'], row['dataset_name'], row['model_name'],  row['adv_size'], row['success_rate']],
			index=['attack', 'file_name', 'csv_name', 'dataset_name', 'model_name', 'adv_size', 'success_rate']))

			print(new_row)

			x_test_lin = np.ndarray.flatten(x_test_succ[i])
			x_adv_lin = np.ndarray.flatten(x_adv[i])

			metrics.append((np.linalg.norm(x_test_lin - x_adv_lin), "L2"))
			metrics.append((np.linalg.norm(x_test_lin - x_adv_lin, ord=1), "L1"))
			metrics.append((np.linalg.norm(x_test_lin - x_adv_lin, ord=np.inf), "Linf"))
			metrics.append((np.linalg.norm(x_test_lin - x_adv_lin, ord=0), "L0"))
			metrics.append((mse(x_test_succ[i], x_adv[i]), "mse"))
			metrics.append((uqi(x_test_succ[i], x_adv[i]), "uqi"))
			metrics.append((ergas(x_test_succ[i], x_adv[i]), "ergas"))
			metrics.append((sam(x_test_succ[i], x_adv[i]), "sam"))
			metrics.append((scc(x_test_succ[i], x_adv[i]), "scc"))
			metrics.append((vifp(x_test_succ[i], x_adv[i]), "vifp"))
			metrics.append((rase(x_test_succ[i], x_adv[i]), "rase"))
			metrics.append((psnrb(x_test_succ[i], x_adv[i]), "psnrb")) 

			for j in metrics:
				add_row = pd.Series([j[0]], index=[j[1]])
				new_row = new_row.append(add_row)

			add_row = pd.Series([ut.to_binary(y_pred_mag[i])], index=['magnet_det'])
			new_row = new_row.append(add_row)

			add_row = pd.Series([ut.to_binary(y_pred_squ[i])], index=['squeezer_det'])
			new_row = new_row.append(add_row)

			is_succ = 1
			if "miss" in row['file_name']:
				is_succ = 0 
			
			add_row = pd.Series([is_succ], index=['is_succ'])
			new_row = new_row.append(add_row)

			add_row = pd.Series([adv_idx[i]], index=['cifar_idx'])
			new_row = new_row.append(add_row)		

			print(new_row)
			rows.append(new_row)

	df_out = ou.create_df_out_explode(rows)
	print(df_out)
	df_out.to_csv("./out_dir/" + 'resnet50_success_explode.csv', index=False)