import sys
import os
import time  
import keras
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd 
from sklearn.metrics import *
import modules.output_util as ou
from utils import my_utils as ut
import modules.model_selector as ms
import modules.attack_selector as sa
import modules.dataset_selector as ds
import modules.metrics_evaluator as me
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def adv_generation(args):

	ut.limit_gpu_usage()
	
	dataset_name = args[0]
	model_name = args[1]
	csv_name = args[2]
	size = int(args[3])

	#LOAD DATASET:_______________________________________________________

	x_test, y_test, x_train, y_train = ds.select_dataset(dataset_name, size)

	#LOAD CLASSIFIER_____________________________________________________

	classifier, model = ms.select_model(dataset_name, model_name)

	#TEST ACCURACY ON NORMAL EXAMPLE_____________________________________

	y_pred = np.argmax(classifier.predict(x_test), axis=1)
	x_test_acc = accuracy_score(y_pred, y_test)
	print("Model accuracy on normal samples: " + str(x_test_acc) + "\n")

	#LOAD GENERATION PARAMETERS VIA CSV__________________________________

	path = "./gen_params/" + csv_name + ".csv"
	df = pd.read_csv(path)

	print("\n________________Attack generation list__________________ \n")
	print(df)
	print("__________________________________________________________ \n")

	#GENERATION LOOP_____________________________________________________

	rows = []

	for index, row in df.iterrows():

		#GENERATE ATTACKS
		attack, name = sa.select_attack(classifier, row)
		start_time = time.time()
		try:
			x_test_adv = attack.generate(x = x_test)
		except:
			print("over")

		end_time = time.time() - start_time 
		x_miss_adv, x_succ_adv, succ_idx, miss_idx, correct_shape = sa.select_adv_samples(classifier, y_test, x_test_adv, y_pred)
		
		#COLLECTING METRICS
		metrics_succ, metrics_miss = me.evaluate_metrics(x_miss_adv, x_succ_adv, x_test, miss_idx, succ_idx)
		out_row = ou.create_output_row(row, name, csv_name, size, correct_shape, succ_idx, miss_idx, end_time, metrics_succ, metrics_miss, dataset_name, model_name)
		rows.append(out_row)
		ou.save_x_adv(name, model_name, csv_name, x_succ_adv, x_miss_adv, succ_idx, miss_idx)

		#SAVING THE DATAFRAME
		df_out = ou.create_df_out(rows)
		path = "./out_dir/adv_attacks/" + model_name + "/" + csv_name 
		df_out.to_csv(path + "/" + csv_name + model_name + '_output.csv', index=False)

	print(df_out)

def main():
	ut.limit_gpu_usage()
	args = sys.argv[1:]
	adv_generation(args)

if __name__=="__main__":
	
	
	
	main()
