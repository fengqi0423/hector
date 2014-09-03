Neural Network in Hector

Neural network in project Hector is still being developed. I will submit merge request to xlvector once several major issues are solved. On high level, this implementation uses backpropagation, sigmoid activation functions, SGD, and square error cost. Long term goal is to support deep learning algorithms.

Current status:

	1. Method "ann" in neural_network.go is deprecated. This is the first version of two layer neural network I contributed to xlvector long time ago. I've probably messed up "ann" after recently forking it from xlvector's repo, since it no longer passes the XOR test. It differs from xlvector's version in that it supports hidden layer dropout.

	2. Method "dnn" has replaced "ann". Its stable version is in branch "v1". It differs from ann in that it
		- can be configured to have any number of layers
		- supports dropout in both hidden and input layers
		- supports momentum
		- can output probabilities instead of diescrete labels

	3. Method "dnn" in master branch is being actively developed (should have used a develop branch). It differs from the stable version in that 
		- all matrices and vectors are replaced by arrays (golang slices) to better control memory allocation (not fully successful yet)
		- mini-batch is supported
		- it initializes weights from a distribution of Gauss(0, 1/sqrt(NumOfInputNodes)). This turned out to be a bad decision for sigmoid activation, and needs to be fixed.

TODOs for dnn

	1. support different activation functions, tanh, softsign, rectifier, etc.

	2. switch to Glorot and Bengio's normalized initialization

	3. fully reuse arrays to avoid memory allocation, or alternatively, figure out a way for better config GC (doesn't seem possible given current golang)

	4. fully use array to avoid map lookups, yet keep leveraging input sparsity (this is almost done - 20% speed up compared to v1 branch)

	5. think of a way to use goroutine and more CPU cores.
		a. committee and Bayesian neural networks
		b. parallel processing for a mini-batch
		c. parallel processing for forward and backward computations of neurons in the same layer (this failed to bring cpu usage to higher than 100% in "ann")

	6. support autoencoder or RBM pre-training

	7. support different cost functions

Setup:

	Neural Network does not require further setup once you've setup Hector. To setup Hector, follow golang's folder structure (https://golang.org/doc/code.html).

	Below steps will clone hector into ~/go_workspace and install its binaries to /usr/local/bin
	mkdir -p ~/go_workplace/src/github.com/xlvector/
	cd ~/go_workplace/src/github.com/xlvector/
	git clone <this repo>
	cd hector
	git checkout v1
	cd bin
	GOPATH=~/go_workspace ./install

Use neural network:
	Input data files should be in svmlight format. Output can be class 0 or 1. It supports multi-class classification, but AUC evaluations is only done for class 1. Feature ids should start from 1, and increase by 1 for each new feature.

	An example command line is:
	hector-mc-run \
		--v 1 \
		--method dnn \
		--global 0 \
		--prob-of-class 1 \
		--steps 10 \
		--hidden 20,100,50 \
		--learning-rate 0.1 \
		--learning-rate-discount 0.95 \
		--dropout-rate 0.5 \
		--dropout-rate-input 0.3 \
		--regularization 0.0 \
		--momentum 0.5 \
		--train build/feature/feature5.trn1.sps.scale \
		--test build/feature/feature5.val1.sps.scale \
		--pred build/val1/dnn_20_10_0.1_0.95_0.5_0.3_0.0_0.5_feature5.val1.yht \
		--validation build/feature/feature5.val1.sps.scale

	Always use hector-mc-run, which means multi-class run.
	Parameters (available in stable version):
		--v: verbose. Output training progress to console. After each iteration, evaluate AUC, if a validation data file is given
		--method: should always be "dnn", for neural network
		--global: add the offset 1 to each input vector, in most scenarios you want to set it to 1
		--prob-of-class: output probabiliy for one class (instead of which class)
		--steps: a.k.a. training epoches
		--hidden: number of hidden neurons, separated by commas, and ordered from the closest to inputs to the closest to outputs. One more offset neuron which is always activated to be 1 is automatically added to each layer.
		--learning-rate: learning rate in backpropagation
		--learning-rate-discount: this is a bad naming. The learning rate is multiplied by this value after each epoch.
		--dropout-rate: probability of dropping out a hidden neuron
		--dropout-rate-input: probability of dropping out an input neuron
		--regularization: L2 regularization scale
		--momentum: parameter change dw at time t+1 will be modified to dw(t+1) = (1-m)*dw(t+1)+m*dw(t), where m is momentum
		--train: training data file
		--test: testing data file
		--validation: validation data file (only useful in verbose mode to evaluate AUC during training)
		--pred: prediction output file
