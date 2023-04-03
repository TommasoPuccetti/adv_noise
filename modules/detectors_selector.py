from detectors.magnet_mnist import MagNetDetector as MagNetDetectorMNIST
from detectors.magnet_cifar import MagNetDetector as MagNetDetectorCIFAR
from detectors.feature_squeezing import FeatureSqueezingDetector


def select_detectors(dataset_name, model, model_name):

	if dataset_name == "cifar":
		lambda_size = 512
		if model_name == "densenet":
			lambda_size = 448
		magnet_detector = MagNetDetectorCIFAR(model, "MagNet", lambda_size)
		squeezer_args = "FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1"
		squeezer_data = "CIFAR"
	else:
		if dataset_name == "mnist":
			magnet_detector = MagNetDetectorMNIST(model, "MagNet")
			squeezer_args = "FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1"
			squeezer_data = "MNIST"

	squeezer = FeatureSqueezingDetector(model, squeezer_args, squeezer_data)

	return magnet_detector, squeezer