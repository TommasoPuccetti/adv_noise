from matplotlib import pyplot as plt
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np


def create_output_row(row, file_name, csv_name, size, correct_shape, succ_idx, miss_idx, end_time, metrics_succ, metrics_miss, dataset_name, model_name): 

	correct_size = correct_shape[0] 
	succ_adv_size = succ_idx.shape[0]
	miss_adv_size = miss_idx.shape[0]
	succ_rate = succ_adv_size / correct_size
	second_image = end_time / size

	add_row = pd.Series([file_name, csv_name, dataset_name, model_name, size, correct_size, succ_adv_size, miss_adv_size, succ_rate, second_image], 
		index=['file_name', 'csv_name', 'dataset_name', 'model_name', 'x_test_size', 'correct_size', 'succ_adv_size', 'miss_adv_size', 'success_rate', 'second_image'])

	row = row.append(add_row)

	for i in metrics_succ:
		add_row = pd.Series([i[1]], index=[i[0]])
		row = row.append(add_row)

	for i in metrics_miss:
		add_row = pd.Series([i[1]], index=[i[0]])
		row = row.append(add_row)
	
	return row


def create_df_out(rows):
 
	df_out = pd.DataFrame()

	#TODO: rimpire dizionario automaticamente con tutti i campi di i

	for i in rows:
		out_dict_succ = {'attack' : i['attack'], 'is_succ': True, 'confidence' : i['confidence'], 'max_iter' : i['max_iter'], 'batch_size' : i['batch_size'],
		 'epsilon' : i['epsilon'], 'epsilon_step' : i['epsilon_step'], 'delta' : i['delta'], 'norm' : i['norm'],'theta' : i['theta'], 'gamma' : i['gamma'], 'eta' : i['eta'],
		 	'max_eval' : i['max_eval'], 'init_eval' : i['init_eval'], 'init_size' : i['init_size'], 'num_trial' : i['num_trial'], 'max_trasl' : i['max_trasl'], 'max_rot' : i['max_rot'],
		 	  'file_name' : i['file_name'] + '_succ', 'csv_name' : i['csv_name'],
		 	 'dataset_name' : i['dataset_name'], 'model_name' : i['model_name'], 'x_test_size' : i['x_test_size'], 'correct_size' : i['correct_size'],
		  'adv_size' : i['succ_adv_size'], 'success_rate' : i['success_rate'], 'second_image' : i['second_image'],
		   'avg_l2' : i['avg_l2_succ'], 'avg_l1' : i['avg_l1_succ'],
		  'avg_linf' : i['avg_linf_succ'], 'avg_l0' : i['avg_l0_succ'],
		   'avg_mse' : i['avg_mse_succ'], 'avg_uqi' : i['avg_uqi_succ'], 'avg_ergas' : i['avg_ergas_succ'],
		    'avg_sam' : i['avg_sam_succ'], 'avg_scc' : i['avg_scc_succ'], 'avg_vifp' : i['avg_vifp_succ'], 'avg_rase' : i['avg_rase_succ'], 'avg_psnrb' : i['avg_psnrb_succ']}

		out_dict_miss = {'attack' : i['attack'], 'is_succ': False, 'confidence' : i['confidence'], 'max_iter' : i['max_iter'], 'batch_size' : i['batch_size'],
		 'epsilon' : i['epsilon'], 'epsilon_step' : i['epsilon_step'], 'delta' : i['delta'], 'norm' : i['norm'],'theta' : i['theta'], 'gamma' : i['gamma'], 'eta' : i['eta'],
		 	'max_eval' : i['max_eval'], 'init_eval' : i['init_eval'], 'init_size' : i['init_size'], 'num_trial' : i['num_trial'], 'max_trasl' : i['max_trasl'], 'max_rot' : i['max_rot'],
		 	 'file_name' : i['file_name'] + '_miss', 'csv_name' : i['csv_name'],
		 	 'dataset_name' : i['dataset_name'], 'model_name' : i['model_name'], 'x_test_size' : i['x_test_size'], 'correct_size' : i['correct_size'],
		  'adv_size' : i['miss_adv_size'], 'success_rate' : i['success_rate'], 'second_image' : i['second_image'], 'avg_l2' : i['avg_l2_miss'], 'avg_l1' : i['avg_l1_miss'],
		  'avg_linf' : i['avg_linf_miss'], 'avg_l0' : i['avg_l0_miss'],
		   'avg_mse' : i['avg_mse_miss'], 'avg_uqi' : i['avg_uqi_miss'], 'avg_ergas' : i['avg_ergas_miss'],
		    'avg_sam' : i['avg_sam_miss'], 'avg_scc' : i['avg_scc_miss'], 'avg_vifp' : i['avg_vifp_miss'], 'avg_rase' : i['avg_rase_miss'], 'avg_psnrb' : i['avg_psnrb_succ']}

	#TODO: sostituire append con concat
		
		df_out = df_out.append(out_dict_succ,  ignore_index = True)
		df_out = df_out.append(out_dict_miss,  ignore_index = True)
	
	return df_out

def create_df_detectors(rows):
		
		df_out = pd.DataFrame()
		
		for i in rows:

			out_dict_succ = {'attack' : i['attack'], 'is_succ': True, 'confidence' : i['confidence'], 'max_iter' : i['max_iter'], 'batch_size' : i['batch_size'],
		 	'epsilon' : i['epsilon'], 'epsilon_step' : i['epsilon_step'], 'delta' : i['delta'], 'norm' : i['norm'],'theta' : i['theta'], 'gamma' : i['gamma'], 'eta' : i['eta'],
		 	'max_eval' : i['max_eval'], 'init_eval' : i['init_eval'], 'init_size' : i['init_size'], 'num_trial' : i['num_trial'], 'max_trasl' : i['max_trasl'], 'max_rot' : i['max_rot'],
		 	 'file_name' : i['file_name'], 
		 	'csv_name' : i['csv_name'],'dataset_name' : i['dataset_name'], 'model_name' : i['model_name'],
		 	'x_test_size' : i['x_test_size'], 'correct_size' : i['correct_size'],
		  	'adv_size' : i['adv_size'], 'success_rate' : i['success_rate'], 'second_image' : i['second_image'], 
		  	'avg_l2' : i['avg_l2'], 'avg_l1' : i['avg_l1'],
		    'avg_linf' : i['avg_linf'], 'avg_l0' : i['avg_l0'],
		  	'avg_mse' : i['avg_mse'], 'avg_uqi' : i['avg_uqi'], 'avg_ergas' : i['avg_ergas'],'avg_sam' : i['avg_sam'], 'avg_scc' : i['avg_scc'], 'avg_vifp' : i['avg_vifp'],
		  	'avg_rase' : i['avg_rase'], 'avg_psnrb' : i['avg_psnrb'],
		  	'magnet_det_rate' : i['magnet_det_rate'],
		    'squeezer_det_rate' : i['squeezer_det_rate']}

			out_dict_miss = {'attack' : i['attack'], 'is_succ': False, 'confidence' : i['confidence'], 'max_iter' : i['max_iter'], 'batch_size' : i['batch_size'],
		 	'epsilon' : i['epsilon'], 'epsilon_step' : i['epsilon_step'], 'delta' : i['delta'], 'norm' : i['norm'],'theta' : i['theta'], 'gamma' : i['gamma'], 'eta' : i['eta'],
		 	'max_eval' : i['max_eval'], 'init_eval' : i['init_eval'], 'init_size' : i['init_size'], 'num_trial' : i['num_trial'], 'max_trasl' : i['max_trasl'], 'max_rot' : i['max_rot'],
		 	 'file_name' : i['file_name'],
		 	'csv_name' : i['csv_name'],'dataset_name' : i['dataset_name'], 'model_name' : i['model_name'],
		 	'x_test_size' : i['x_test_size'], 'correct_size' : i['correct_size'],
		  	'adv_size' : i['adv_size'], 'success_rate' : i['success_rate'], 'second_image' : i['second_image'],
		  	'avg_l2' : i['avg_l2'], 'avg_l1' : i['avg_l1'],
		  	'avg_linf' : i['avg_linf'], 'avg_l0' : i['avg_l0'],
		   	'avg_mse' : i['avg_mse'], 'avg_uqi' : i['avg_uqi'], 'avg_ergas' : i['avg_ergas'], 'avg_psnrb' : i['avg_psnrb'],
		    'avg_sam' : i['avg_sam'], 'avg_scc' : i['avg_scc'], 'avg_vifp' : i['avg_vifp'], 'avg_rase' : i['avg_rase'], 'magnet_det_rate' : i['magnet_det_rate'],
		    'squeezer_det_rate' : i['squeezer_det_rate']}

			df_out = df_out.append(out_dict_succ,  ignore_index = True)
			#df_out = df_out.append(out_dict_miss,  ignore_index = True)

		return df_out

def create_df_out_explode(rows):
 
	df_out = pd.DataFrame()

	#TODO: rimpire dizionario automaticamente con tutti i campi di i

	for i in rows:
		out_dict = {'attack' : i['attack'],  'file_name' : i['file_name'], 'csv_name' : i['csv_name'],
		 	 'dataset_name' : i['dataset_name'], 'model_name' : i['model_name'],
		  'adv_size' : i['adv_size'], 'success_rate' : i['success_rate'], 'L2' : i['L2'], 'L1' : i['L1'],
		  'Linf' : i['Linf'], 'L0' : i['L0'],
		   'mse' : i['mse'], 'uqi' : i['uqi'], 'ergas' : i['ergas'],
		    'sam' : i['sam'], 'scc' : i['scc'], 'vifp' : i['vifp'], 'rase' : i['rase'], 'psnrb' : i['psnrb'], 'magnet_det' : i['magnet_det'], 'squeezer_det' : i['squeezer_det'],
		     'is_succ': i['is_succ'], 'cifar_idx': i['cifar_idx'] }

	#TODO: sostituire append con concat
		
		df_out = df_out.append(out_dict,  ignore_index = True)
	
	return df_out

def save_x_adv(name, model_name, csv_name, x_succ_adv, x_miss_adv, succ_idx, miss_idx):
	
	path = "./out_dir/adv_attacks/" + model_name + "/" + csv_name
	if not os.path.exists(path):
		os.mkdir(path)

	name_succ = name + "_succ"
	name_miss = name + "_miss"
	
	#np.save(path + "/" + name + '_temp_rows',temp_rows)
	
	if succ_idx.shape[0] > 0:
		plt.imshow(x_succ_adv[0])
		plt.savefig(path + "/" + model_name + name_succ + '.png')
		np.save(path + "/" + model_name + name_succ + '_sample', x_succ_adv)
		np.save(path + "/" + model_name + name_succ + '_index', succ_idx)
	
	if miss_idx.shape[0] > 0:
		plt.imshow(x_miss_adv[0])
		plt.savefig(path + "/" + model_name + name_miss + '.png')
		np.save(path + "/"  +  model_name + name_miss + '_sample', x_miss_adv)
		np.save(path + "/"  +  model_name + name_miss + '_index', miss_idx)

