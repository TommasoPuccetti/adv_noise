from sewar.full_ref import uqi
import numpy as np
from sewar import *


def  evaluate_metrics(x_miss_adv, x_succ_adv, x_test, miss_idx, succ_idx):

	#TODO: aggiungere metriche
	#TODO: qui si può stampare i singoli valori delle metriche per ogni img succ e miss stampando liste provvisorie
	
	mse_arr_succ   = []
	uqi_arr_succ   = []
	ergas_arr_succ = []
	sam_arr_succ  = []
	scc_arr_succ   = []
	vifp_arr_succ  = []
	rase_arr_succ  = []
	psnrb_arr_succ  = []
	
	l2 = 0
	l1 = 0
	linf = 0
	l0 = 0
	
	data_succ = []

	if succ_idx.shape[0] > 0:
		x_test_succ = x_test[succ_idx]	

		for x in range(len(x_succ_adv)):
	 		mse_arr_succ.append(mse(x_test_succ[x], x_succ_adv[x]))
	 		uqi_arr_succ.append(uqi(x_test_succ[x], x_succ_adv[x]))
	 		ergas_arr_succ.append(ergas(x_test_succ[x], x_succ_adv[x]))
	 		sam_arr_succ.append(sam(x_test_succ[x], x_succ_adv[x]))
	 		scc_arr_succ.append(scc(x_test_succ[x], x_succ_adv[x]))
	 		vifp_arr_succ.append(vifp(x_test_succ[x], x_succ_adv[x]))
	 		rase_arr_succ.append(rase(x_test_succ[x], x_succ_adv[x]))
	 		psnrb_arr_succ.append(psnrb(x_test_succ[x], x_succ_adv[x]))
	
		l2, l1, linf, l0 = norms_mean(x_test_succ, x_succ_adv)

	data_succ.append(('avg_mse_succ', np.mean(mse_arr_succ)))
	data_succ.append(('avg_uqi_succ', np.mean(uqi_arr_succ)))
	data_succ.append(('avg_ergas_succ', np.mean(ergas_arr_succ)))
	data_succ.append(('avg_sam_succ', np.mean(sam_arr_succ)))
	data_succ.append(('avg_scc_succ', np.mean(scc_arr_succ)))
	data_succ.append(('avg_vifp_succ', np.mean(vifp_arr_succ)))
	data_succ.append(('avg_l2_succ', l2))
	data_succ.append(('avg_l1_succ', l1))
	data_succ.append(('avg_linf_succ', linf))
	data_succ.append(('avg_l0_succ', l0))
	data_succ.append(('avg_rase_succ', np.mean(rase_arr_succ)))
	data_succ.append(('avg_psnrb_succ', np.mean(psnrb_arr_succ)))

	mse_arr_miss   = []
	uqi_arr_miss   = []
	ergas_arr_miss = []
	sam_arr_miss   = []
	scc_arr_miss   = []
	vifp_arr_miss  = []
	rase_arr_miss  = []
	psnrb_arr_miss  = []

	l2 = 0
	l1 = 0
	linf = 0
	l0 = 0

	data_miss = []

	if miss_idx.shape[0] > 0:
		x_test_miss = x_test[miss_idx]

		for x in range(len(x_miss_adv)):
	 		mse_arr_miss.append(mse(x_test_miss[x], x_miss_adv[x]))
	 		uqi_arr_miss.append(uqi(x_test_miss[x], x_miss_adv[x]))
	 		ergas_arr_miss.append(ergas(x_test_miss[x], x_miss_adv[x]))
	 		sam_arr_miss.append(sam(x_test_miss[x], x_miss_adv[x]))
	 		scc_arr_miss.append(scc(x_test_miss[x], x_miss_adv[x]))
	 		vifp_arr_miss.append(vifp(x_test_miss[x], x_miss_adv[x]))
	 		rase_arr_miss.append(rase(x_test_miss[x], x_miss_adv[x]))
	 		psnrb_arr_miss.append(psnrb(x_test_miss[x], x_miss_adv[x]))
		
		l2, l1, linf, l0 = norms_mean(x_test_miss, x_miss_adv)

	data_miss.append(('avg_mse_miss', np.mean(mse_arr_miss)))
	data_miss.append(('avg_uqi_miss', np.mean(uqi_arr_miss)))
	data_miss.append(('avg_ergas_miss', np.mean(ergas_arr_miss)))
	data_miss.append(('avg_sam_miss', np.mean(sam_arr_miss)))
	data_miss.append(('avg_scc_miss', np.mean(scc_arr_miss)))
	data_miss.append(('avg_vifp_miss', np.mean(vifp_arr_miss)))
	data_miss.append(('avg_l2_miss', l2))
	data_miss.append(('avg_l1_miss', l1))
	data_miss.append(('avg_linf_miss', linf))
	data_miss.append(('avg_l0_miss', l0))
	data_miss.append(('avg_rase_miss', np.mean(rase_arr_miss)))
	data_miss.append(('avg_psnr_miss', np.mean(psnrb_arr_miss)))


	return data_succ, data_miss



def norms_mean(x_test, x_adv):
	
	l2 = 0
	l1 = 0
	linf = 0

	for i, x in enumerate(x_adv):
		x_test_lin = np.ndarray.flatten(x_test[i])
		x_adv_lin = np.ndarray.flatten(x_adv[i])
		l2 = l2 + np.linalg.norm(x_test_lin - x_adv_lin)
		l1 = l1 + np.linalg.norm(x_test_lin - x_adv_lin, ord=1)
		linf = linf + np.linalg.norm(x_test_lin - x_adv_lin, ord=np.inf)
		l0 = np.linalg.norm(x_test_lin - x_adv_lin, ord=0) 
	
	l2 = l2/x_adv.shape[0]
	l1 = l1/x_adv.shape[0]
	linf = linf/x_adv.shape[0]
	l0 = l0/x_adv.shape[0]

	return l2, l1, linf, l0