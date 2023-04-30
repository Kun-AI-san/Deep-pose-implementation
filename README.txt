################ README ##############################
This README will explain the basic instructions on how to run the complete DL pipeline for the implementation of DeepPose.
The following steps should be followed in sequence and all the required dependancies mentioned are provided in the code files.
1. Run the LSP_data.py:
	For this file, the Leeds Sports Dataset(LSP) and Extended Leeds Sports Dataset (LSPe) are required.
	These files are already provided in the code files but you can also find LSPe dataset in the following link
		a) LSPe: http://sam.johnson.io/research/lspet.html
	The LSP_data.py proprocesses the images and their respective joint annotations by cropping, random translation, flips, etc. The file saves
	training data and testing data onto your system as "train_images.npy", "train_labels.npy" for training and "test_images.npy", "test_labels.npy"
	as test files.
	To run the file, use command: "py LSP_data.py" (For Windows) or "python LSP_data.py" (for Linux). You can use python IDEs to run the code too.

2. Run the Train_stage_1.py file:
	This file imports the DNN model mentioned in "Resnet_stage_1.py" file and trains this network on the saved Training data.
	This file also saves the trained DNN model for the next staged layer of training.
	To run the file, use command:  "py Train_stage_1.py" (For Windows) or "python Train_stage_1.py" (for Linux). You can use python IDEs to run the code too.
	For the ease of use, we are including an output file called "Generate_train_output.py" of which gives output "training_output.npy" of the model that we trained using the above code for you to verify the other files as this training takes a long time.

3. Run the test_stage_1.py file:
	This file imports the DNN model mentioned in "Resnet_stage_1.py" file and tests this network on the saved test data.
	To run the file, use command:  "py test_stage_1.py" (For Windows) or "python test_stage_1.py " (for Linux). You can use python IDEs to run the code too.

4. Run the Generate_train_output.py file:
	As explained in step 2, this file is necessary for the input used for stage 2 training.
	To run the file, use command:  "Generate_train_output.py" (For Windows) or "Generate_train_output.py " (for Linux). You can use python IDEs to run the code too.
	
5. Run the Train_stage_2.py file:
	This file imports the DNN "Resnet_stage_1" model output "training_output.npy" file mentioned in step 2 file and trains the "Alex_net_c1" network located i "Cascade_1_net.py" file on the saved Training data
	for each joint. Another Important aspect of this code is that it augments the already existing data by adding 20 simulated predictions for each joint in each image.
	This is possible due to the "deltas.npy" file. Code comments are mentioned in the code file in case you want to make your own deltas.npy file. But this file already provided in the code files.
	This file saves 14 different Alex_net_c1 models each trained on a single joint of the dataset.
	To run the file, use command:  "py Train_stage_2.py" (For Windows) or "python Train_stage_2.py" (for Linux). You can use python IDEs to run the code too.

6. Run the test_stage_2.py file:
	This file is used to evaluate or test the trained models from step 5 on the test data.
	o run the file, use command:  "py test_stage_2.py" (For Windows) or "python test_stage_2.py" (for Linux). You can use python IDEs to run the code too.

Additional important files:
1. Resnet_stage_1.py
2. Cascade_1_net.py
3. train_output.npy
4. deltas.npy
5. images_1 folder and images_2 folder which contain the dataset images.
6. joints_1.mat and joints_2.mat files that contain joint annotations for their respective images.

Important packages:
1. PyTorch
2. NumPy
3. CUDA
4. torchvision

################ DISCLAIMER ##############################
The Cascade_1_net.py contains a version of AlexNet (Alex_net_c1() class) which is the exact model used by the well established paper DeepPose.
I do not claim this section of the code in "Cascade_1_net.y" to by my own.

References to the above disclaimer:
[1] Toshev, A. and Szegedy, C., 2014. Deeppose: Human pose estimation via deep neural networks. 
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1653-1660).
    https://arxiv.org/abs/1312.4659