# On Measuring Adversarial Perturbations to Select Diverse Attack Classes

This repository reproduce the dataset generation for the paper: On Measuring Adversarial Perturbations to Select Diverse Attack Classes

____INSTALLATION:____

To install the required packages and GPU support please load the Anaconda environment  "adv_gen":

	1) conda env create -f conda_env.yml
	2) conda activate generation
	3) conda env list 

____GENERATE ADVERSARIAL ATTACKS____

The main.py file generate adversarial attacks using the list and the parameters listed in the selected csv file in the "gen_param" folder. 

The attacks are selected in the ART Toolbox (documentation v1.10.1, https://adversarial-robustness-toolbox.readthedocs.io/en/latest/).
The script exclude the legitimate sample that are missclassified originally by the model.   

- To execute attacks generations loop in main.py:
	
	python main.py -dataset -model -csv_name -img_size -log_all

	Parameters:
		- dataset: name of the dataset
			-cifar
			-mnist
		- model: name of the classifier model
			-carlini
			-clever
			-conv12
			-densenet
		- csv_name: name of the input csv in the gen_params folder
		- img_size: size of the starting legitimate sample set
		
		example: python main.py cifar conv12 gen_list 100

The program save its output in the ./out_dir/model directory with the same name of the input csv file. The sub folder contains all the adversarial set generated from the attacks listed in the input csv, the indexes that trace their position in the test set of the selected dataset and a sample adv image for each attack. 
Also an output csv is generated collecting all the defined mean metrics for each dataset.

____RUN DETECTORS AND CREATE THE FINAL DATASET____

run 
	"python explode_atk.py"

The script takes the file path of a csv file obtained after the generation of an attack set and computes both L-norms and iamge quality metrics on all the images in the attack set. It also exercise both Magnet and Squeezer detectors on each image. The output csv file associate, for each image of each attack set, the metrics and the detection label of both the detectors. 
	
The path of the "output_csv" can be changed in the code to indicate a specific output file. 

____PREDICT THE SUCCESS RATE OF THE ATTACK____

A regression analysis to predict the success rate of an attack using distance metrics as input features. Noticeably, the regressor can predict accurately the success rate of the attack by solely relying on distance metrics. 

run
   	"predict_success_rate.py"

The script take the file path of the final tabular dataset and predicts the success rate of the attack for a specific configuration, using the distance metrics computed on a sample image. The script take as input a csv file that specifies the experiment configuration (feature of the tabular dataset to be used for the prediction). 

	Parameters:
		-tabular_dataset
		-experiment_configurations
		-output_folder
		
The scripts generate in the output folder the 3 files for each experiment defined in experiment_configurations csv:
	
	- Results for each attack configurations ("*packets.csv")
	- Results for each attack type ("*attacks.csv")
	- Results for all the images ("*all.csv")
	
	PREDICT THE SUCCESS RATE UNKNOWN
		We repeat the study by training the regressor multiple times using adversarial images generated from the same attack, and testing on adversarial images generated using the other attacks. This shows the mutual predictability of the success rate between attacks, which we can use to understand if two attacks are similar.
	
	run
		"loo_predict_success_rate.py"
	
	The script is a modified version of the predict_success_rate.py and can be exercised using the same input parameters
	
____PREDICT THE DETECTION OUTPUT OF THE DETECTORS____

A binary classification analysis to predict the detection of an attack using as input features the distance metrics, using 2 detectors from the state of the art. Attacks with similar distance metrics values are predicted to be detected similarly by a defense.

run
   	"predict_detector_output.py"

The script take the file path of the final tabular dataset and predicts the detector output for a given adversarial image, using the distance metrics computed on a sample image. The script take as input a csv file that specifies the experiment configuration (feature of the tabular dataset to be used for the prediction). 

	Parameters:
		-tabular_dataset
		-experiment_configurations
		-output_folder
		
The scripts generate in the output folder the 3 files for each experiment defined in experiment_configurations csv:
	
	- Results for each attack configurations ("*packets.csv")
	- Results for each attack type ("*attacks.csv")
	- Results for all the images ("*all.csv")	

	PREDICT THE DETECTION OUTPUT UNKNOWN
		We repeat the study by training the classifier multiple times using adversarial images generated from the same attack, and testing on adversarial images generated using the other attacks. This shows the mutual predictability of the detection output between attacks, which we can use to understand if two attacks are similar.

	run
		"loo_predict_detector_output.py"
	
	The script is a modified version of the predict_detector_output.py and can be exercised using the same input parameters.

____DETECTORS____

| Detector          | # References                             | 
|-------------------|------------------------------------------|
| MagNet            | https://arxiv.org/abs/1705.09064         |
| Feature Squeezing | https://github.com/mzweilin/EvadeML-Zoo  |

____TARGET MODELS____

## Dataset: MNIST

| Model Name | # Trainable Parameters  | Testing Accuracy | Alias   |
|------------|-------------------------|------------------|---------|
| Carlini    |  312,202                |     0.9943       | MNIST-1 |
| Cleverhans |  710,218                |     0.9919       | MNIST-2 |

## Dataset: CIFAR-10

|      Model Name     |  # Trainable Parameters  | Testing Accuracy | Alias   |
|---------------------|--------------------------|------------------|---------|
| DenseNet(L=40,k=12) | 1,019,722                |     0.9484       | CIFAR-1 | 
| ConvNet12           | 2,919,082		 |     0.8514	    | CIFAR-2 |
| ResNet50	      | 26,162,698		 |     0.9356       | CIFAR-3 |
