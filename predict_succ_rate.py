import sys
import random 
import sklearn
import numpy as np
import pandas as pd
from utils import my_utils as ut
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE
from scipy.spatial import distance
from sklearn.metrics import r2_score
from scipy.spatial import distance



def create_df_split(csv_name, label, rnd_state=10):

	path = "./out_dir/" + csv_name + ".csv"

	df = pd.read_csv(path)
	df = drop_labels(df, label)

	#SPLIT DF BY CIFAR ORIGINAL INDEX 
	#cifar_idx = df.cifar_idx.unique()
	df_train, df_test = train_test_split(df, test_size =.20, shuffle=True, random_state=541) 
	print(df_test)
	print(df_train)

	#df_train = df[df['cifar_idx'].isin(train_idx)]
	#df_test = df[df['cifar_idx'].isin(test_idx) ]

	df_train = df_train.replace(np.inf,0).reset_index(drop=True)
	df_test = df_test.replace(np.inf,0).reset_index(drop=True)
	df_train = df_train.replace(np.nan,0).reset_index(drop=True)
	df_test = df_test.replace(np.nan,0).reset_index(drop=True)

	df_test.to_csv('out_dir' + '/'  + 'test' + '.csv', index=False)
	df_train.to_csv('out_dir' + '/'  + 'train' + '.csv', index=False)

	print("\nTraining Set shape :", df_train.shape)
	print("Test Set shape :", df_test.shape)

	return df_train, df_test

def drop_labels(df, label):

	df = df.drop('magnet_det', axis=1)
	df = df.drop('squeezer_det', axis=1)
	df = df.drop('is_succ', axis=1)
	df = df.drop('adv_size', axis=1)
	
	return df

def prepare_label(df_train, df_test, label):

	y_train = pd.DataFrame(df_train[label], dtype='category')
	x_train = df_train.drop(label, axis=1)

	print(y_train)
	print(x_train)

	y_test = pd.DataFrame(df_test[label], dtype='category')
	x_test = df_test.drop(label, axis=1)

	print(y_test)
	print(x_test)

	x_train = x_train.drop('cifar_idx', axis=1)
	x_train = x_train.drop('file_name', axis=1)
	y_train = y_train.values.ravel()
	y_test = y_test.values.ravel()

	return x_train, y_train, x_test, y_test

def run_classifier(x_train, y_train, x_test, y_test, features, id_exp, label, param=0):

	x_train = x_train[features]

	clf = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
	clf.fit(x_train, y_train)

	attacks_names = x_test.file_name.unique()
	x_test = x_test.reset_index() 

	df_packet = pd.DataFrame(columns=['id', 'label', 'features', 'file_name', 'csv_name','attack', 'dataset_name', 'model_name', 'ACC', 'COS'])
	df_attack = pd.DataFrame(columns=['id', 'label', 'features', 'file_name', 'csv_name', 'attack', 'dataset_name', 'model_name', 'ACC', 'COS'])
	df_all = pd.DataFrame(columns=['file_name', 'ACC', 'COS'])

	"""
	#CALCULATE ACC FOR EACH ATTACK IMAGE PACKET
	for name in attacks_names:
		x_test_atk = x_test[x_test['file_name'] == name]
		y_index = x_test.index[x_test['file_name'] == name]
		y_test_atk = y_test[y_index]
		csv_name = x_test_atk['csv_name'].iloc[0]
		attack = x_test_atk['attack'].iloc[0]
		dataset_name = x_test_atk['dataset_name'].iloc[0]
		model_name = x_test_atk['model_name'].iloc[0]	
		x_test_atk = x_test_atk[features]	
		
		y_pred = clf.predict(x_test_atk)
		print(y_pred)
		print(y_test_atk)
		acc = np.sqrt(MSE(y_test_atk, y_pred))

		new_row = pd.DataFrame({'id': id_exp, 'label': label, 'features': ' '.join(features), 'csv_name': csv_name, 
			'attack': attack, 'file_name': name, 'dataset_name': dataset_name,
			'model_name': model_name, 'ACC': acc}, index=[1])
		df_packet = pd.concat([df_packet, new_row], ignore_index=True)

	attacks_type = x_test.attack.unique()

	#CALCULATE ACC FOR EACH ATTACK TYPE
	for att in attacks_type:
		x_test_atk = x_test[x_test['attack'] == att]
		y_index = x_test.index[x_test['attack'] == att]
		y_test_atk = y_test[y_index]
		csv_name = x_test_atk['csv_name'].iloc[0]
		attack = x_test_atk['attack'].iloc[0]
		dataset_name = x_test_atk['dataset_name'].iloc[0]
		model_name = x_test_atk['model_name'].iloc[0]
		x_test_atk = x_test_atk[features]
	
		y_pred = clf.predict(x_test_atk)
		acc = np.sqrt(MSE(y_test_atk, y_pred))

		new_row = pd.DataFrame({'id': id_exp, 'label': label, 'features': ' '.join(features), 'csv_name': csv_name, 
			'attack': attack, 'file_name': name, 'dataset_name': dataset_name,
			'model_name': model_name, 'ACC': acc}, index=[1])
		df_attack = pd.concat([df_attack, new_row], ignore_index=True)
"""
	#CALCULATE ACC FOR ALL IMAGES
	x_test_atk = x_test#[~x_test['file_name'].str.contains('miss')]
	y_index = x_test.index#[~x_test['file_name'].str.contains('miss')]
	y_test_atk = y_test[y_index]
	x_test_atk = x_test_atk[features]

	y_pred = clf.predict(x_test_atk)
	acc = np.sqrt(MSE(y_test_atk, y_pred))
	#r2 =  r2_score(y_test_atk, y_pred)
	cos = distance.cosine(y_test_atk, y_pred)

	print(y_test_atk)
	print(y_pred)
	#cos = distance.cosine(y_test_atk, y_pred)

	new_row = pd.DataFrame({'id': id_exp, 'label': label, 'features': ' '.join(features), 'csv_name': 'all', 
			'attack': 'all', 'file_name': 'all', 'dataset_name': "cifar",
			'model_name': "conv12", 'ACC': acc, 'COS': cos}, index=[1])
	df_all = pd.concat([df_all, new_row], ignore_index=True)

	return df_packet, df_attack, df_all

def main():
	
	args = sys.argv[1:]
	csv_data = args[0]
	csv_in = args[1]
	csv_out = args[2]  
	in_path = "./pred_params/" + csv_in + ".csv"
	
	df_in = pd.read_csv(in_path)

	for index, row in df_in.iterrows():
		#To try different splits
		for i in range(0, 1):
			df_train, df_test = create_df_split(csv_data, row['label'], i)
			x_train, y_train, x_test, y_test = prepare_label(df_train, df_test, row['label'])
			df_out, df_attack, df_all = run_classifier(x_train, y_train, x_test, y_test, row['features'].split(','), row['id'], row['label'])

			out_path = "./out_dir/predictions/" + csv_out + "/" 

			feature_list = ""
			#df_out.to_csv(out_path + '/' + csv_in + row['id'] + '(' + feature_list.join(row['features']) + ')' + '_' + row['label'] + 'packets' + str(i) +'.csv', index=False)
			#df_attack.to_csv(out_path + '/' + csv_in + row['id'] + '(' + feature_list.join(row['features']) + ')' + '_' + row['label'] + 'attacks' + str(i) + '.csv', index=False)
			
			print(out_path)
			print(csv_in)
			print(row['id'])
			print(feature_list.join(row['features']))
			print(row['label'])

			df_all.to_csv(out_path + '/' + csv_in +  row['id'] +  '(' + feature_list.join(row['features']) + ')' + '_' + row['label'] + 'all' + str(i) + '.csv', index=False)


if __name__=="__main__":
	
	ut.limit_gpu_usage()
	
	main()













