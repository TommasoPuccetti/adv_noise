import sys 
import numpy as np
from sklearn.metrics import *
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import CarliniLInfMethod
from art.attacks.evasion import DeepFool
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import UniversalPerturbation
from art.attacks.evasion import SaliencyMapMethod
from art.attacks.evasion import NewtonFool
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import ElasticNet
from art.attacks.evasion import AdversarialPatchPyTorch
from art.attacks.evasion import ZooAttack
from art.attacks.evasion import BoundaryAttack
from art.attacks.evasion import SimBA
from art.attacks.evasion import GeoDA
from art.attacks.evasion import SpatialTransformation
from matplotlib import pyplot as plt


def select_attack(classifier, param):

	batch_size = int(param['batch_size'])
	atk_name = param['attack']

	if atk_name == 'cw2':
		confidence = float(param['confidence'])
		max_iter = int(param['max_iter'])
		print(param['max_iter'])
		attack = CarliniL2Method(classifier=classifier, confidence=confidence, max_iter=max_iter, batch_size=batch_size)
		name = atk_name + str(confidence) + str(max_iter)

	if atk_name == 'cwinf':
		confidence = float(param['confidence'])
		max_iter = int(param['max_iter'])
		print(param['max_iter'])
		attack = CarliniLInfMethod(classifier=classifier, confidence=confidence, max_iter=max_iter)
		name = atk_name + str(confidence) + str(max_iter)

	if atk_name == 'fgm':
		#TODO: fgsm has minimal perturbation parameter wich introduce eps_step parameter to control the variation of eps at each iteration. ADD these options.
		epsilon = float(param['epsilon'])
		attack = FastGradientMethod(estimator=classifier, eps=epsilon)
		name = atk_name + str(epsilon)

	if atk_name == 'bim':
		epsilon = param['epsilon']
		epsilon_step = param['epsilon_step']
		attack = BasicIterativeMethod(estimator=classifier, eps= epsilon, eps_step=epsilon_step)
		name = atk_name + str(epsilon) + str(epsilon_step)

	if atk_name == 'deep':
		max_iter = int(param['max_iter'])
		epsilon = float(param['epsilon'])
		attack = DeepFool(classifier=classifier, max_iter= max_iter, epsilon=epsilon)
		name = atk_name + str(epsilon) + str(max_iter)

	if atk_name == 'pgd':
		max_iter = int(param['max_iter'])
		epsilon = float(param['epsilon'])
		attack = AutoProjectedGradientDescent(estimator=classifier, max_iter= max_iter, eps=epsilon, loss_type="cross_entropy")
		name = atk_name + str(epsilon) + str(max_iter)
	
	if atk_name == 'uni':
		max_iter = int(param['max_iter'])
		delta = float(param['delta'])
		epsilon = float(param['epsilon'])
		norm = float(param['norm'])
		attack = UniversalPerturbation(classifier=classifier, max_iter=max_iter, delta=delta, eps=epsilon, norm=norm)
		name = atk_name + str(epsilon) + str(max_iter) + str(delta) + str(norm)

	if atk_name == 'jsma':
		theta = float(param['theta'])
		gamma = float(param['gamma'])
		attack = SaliencyMapMethod(classifier=classifier, theta=theta, gamma=gamma)
		name = atk_name + str(theta) + str(gamma) 

	if atk_name == 'newton':
		max_iter = int(param['max_iter'])
		eta = float(param['eta'])
		attack = NewtonFool(classifier=classifier, max_iter=max_iter, eta=eta)
		name = atk_name + str(eta) + str(max_iter) 

	if atk_name == 'hopskip':
		norm = param['norm']
		max_iter = int(param['max_iter'])
		max_eval = int(param['max_eval'])
		init_eval = int(param['init_eval'])
		init_size = int(param['init_size'])
		attack = HopSkipJump(classifier=classifier, max_iter=max_iter, max_eval=max_eval, init_eval=init_eval, init_size=init_size)
		name = atk_name + str(norm) + str(max_iter) + str(max_eval) + str(init_eval) + str(init_size)

	if atk_name == 'elastic':
		max_iter = int(param['max_iter'])
		confidence = float(param['confidence'])
		attack = ElasticNet(classifier=classifier, max_iter=max_iter, confidence=confidence)
		name = atk_name + str(max_iter) + str(confidence)

	#TODO: Adversarial patch al momento non è supportata è 2 step con applicazione
	if atk_name == 'patch':
		lr = float(param['lr'])
		max_iter = int(param['max_iter'])
		attack = AdversarialPatchPyTorch(classifier=classifier)
		name = atk_name + str(max_iter) + str(lr)

	if atk_name == 'zoo':
		max_iter = int(param['max_iter'])
		confidence = float(param['confidence'])
		attack = ZooAttack(classifier=classifier, max_iter=max_iter, confidence=confidence)
		name = atk_name + str(max_iter) + str(confidence)

	if atk_name == 'boundary':
		targeted = False
		delta = float(param['delta'])
		max_iter = int(param['max_iter'])
		epsilon = float(param['epsilon'])
		num_trial = int(param['num_trial'])
		attack = BoundaryAttack(estimator=classifier, max_iter=max_iter, delta=delta, epsilon=epsilon, num_trial=num_trial, targeted=False)
		name = atk_name + str(max_iter) + str(delta) + str(epsilon) + str(num_trial)

	#not working	
	if atk_name == 'simba':
		max_iter = int(param['max_iter'])
		epsilon = float(param['epsilon'])
		attack = SimBA(classifier=classifier, max_iter= max_iter, epsilon=epsilon)
		name = atk_name + str(epsilon) + str(max_iter)

	#TODO Capire quali sono i parametri di generazione giusti 
	if atk_name == 'spatial':
		max_trasl = float(param['max_trasl'])
		max_rot = float(param['max_rot'])
		attack = SpatialTransformation(classifier=classifier, max_translation=max_trasl, max_rotation=max_rot) 
		name = atk_name + str(max_rot) + str(max_trasl)

	#not working
	if atk_name == 'geo':
		max_iter = int(param['max_iter'])
		epsilon = float(param['epsilon'])
		attack = GeoDA(estimator=classifier, max_iter= max_iter, bin_search_tol=epsilon)
		name = atk_name + str(epsilon) + str(max_iter)
	
	return attack, name


def get_adv_indexes(y_pred, y_pred_adv, y_test):
	
	print("Total test set images " + str(y_test.shape))
	
	y_bool = (y_pred == y_test[:len(y_pred)])
	correct_idx = np.asarray([i for i, val in enumerate(y_bool) if val])
	
	print("Shape of correctly classified sample " + str(correct_idx.shape))

	y_bool_adv = (y_pred_adv == y_test[:len(y_pred_adv)])
	succ_adv_idx = np.asarray([j for j, val1 in enumerate(y_bool_adv) if not val1])
	miss_adv_idx = np.asarray([k for k, val2 in enumerate(y_bool_adv) if val2])
	
	print("Sucessful adversarial attacks on all the test set images " + str(succ_adv_idx.shape))
	print("Missed adverarial attacks on all the test set images" + str(miss_adv_idx.shape))

	succ_idx = np.intersect1d(succ_adv_idx, correct_idx)
	miss_idx = np.intersect1d(miss_adv_idx, correct_idx)
	
	print("Shape of total successful selected attack samples " + str(succ_idx.shape))
	print("Shape of total missed attack samples " + str(miss_idx.shape))

	return succ_idx, miss_idx, correct_idx.shape


def select_adv_samples(classifier, y_test, x_test_adv, y_pred):
	
	y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)

	succ_idx, miss_idx, correct_shape = get_adv_indexes(y_pred, y_pred_adv, y_test)
	
	if miss_idx.shape[0] > 0:
		x_miss_adv = x_test_adv[miss_idx]
		acc = accuracy_score(y_pred_adv[miss_idx], y_test[miss_idx])
		print("Accuracy score for missed attacks generated from correctly classified test samples " + str(acc))
	else:
		x_miss_adv = np.asarray([])
		miss_idx = np.asarray([])

	if succ_idx.shape[0] > 0:
		x_succ_adv = x_test_adv[succ_idx]
		acc = accuracy_score(y_pred_adv[succ_idx], y_test[succ_idx])
		print("Accuracy score for succesfull attacks generated from correctly classified test samples " + str(acc) + "\n")
	else: 
		x_succ_adv = np.asarray([])
		succ_idx = np.asarray([])
		
	return x_miss_adv, x_succ_adv, succ_idx, miss_idx, correct_shape

