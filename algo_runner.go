/*
Package hector is a golang based machine learning lib. It intend to implement all famous machine learning algoirhtms by golang.
Currently, it only support algorithms which can solve binary classification problems. Supported algorithms include:
1. Decision Tree (CART, Random Forest, GBDT)
2. Logistic Regression
3. SVM
4. Neural Network
*/
package hector

import (
	"strconv"
	"os"
)

func AlgorithmRun(classifier Classifier, train_path string, test_path string, pred_path string, params map[string]string) (float64, []*LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewDataSet()

	err := train_dataset.Load(train_path, global)
	
	if err != nil{
		return 0.5, nil, err
	}
	
	test_dataset := NewDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil{
		return 0.5, nil, err
	}
	classifier.Init(params)
	auc, predictions := AlgorithmRunOnDataSet(classifier, train_dataset, test_dataset, pred_path, params)

	return auc, predictions, nil
}

func AlgorithmTrain(classifier Classifier, train_path string, params map[string]string) (error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewDataSet()

	err := train_dataset.Load(train_path, global)
	
	if err != nil{
		return err
	}

	classifier.Init(params)
	classifier.Train(train_dataset)

	model_path, _ := params["model"]

	if model_path != "" {
		classifier.SaveModel(model_path)
	}

	return nil
}

func AlgorithmTest(classifier Classifier, test_path string, pred_path string, params map[string]string) (float64, []*LabelPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	
	model_path, _ := params["model"]
	classifier.Init(params)
	if model_path != "" {
		classifier.LoadModel(model_path)
	} else {
		return 0.0, nil, nil
	}

	test_dataset := NewDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil{
		return 0.0, nil, err
	}
	
	auc, predictions := AlgorithmRunOnDataSet(classifier, nil, test_dataset, pred_path, params)

	return auc, predictions, nil
}

func AlgorithmRunOnDataSet(classifier Classifier, train_dataset, test_dataset *DataSet, pred_path string, params map[string]string) (float64, []*LabelPrediction) {
	
	if train_dataset != nil {
		classifier.Train(train_dataset)
	}

	predictions := []*LabelPrediction{}
	var pred_file *os.File
	if pred_path != ""{
		pred_file, _ = os.Create(pred_path)
	}
	for _,sample := range test_dataset.Samples {
		prediction := classifier.Predict(sample)
		if pred_file != nil{
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		predictions = append(predictions, &(LabelPrediction{Label: sample.Label, Prediction: prediction}))
	}
	if pred_path != ""{
		defer pred_file.Close()
	}
		
	auc := AUC(predictions)
	return auc, predictions
}


/* Regression */
func RegAlgorithmRun(regressor Regressor, train_path string, test_path string, pred_path string, params map[string]string) (float64, []*RealPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewRealDataSet()

	err := train_dataset.Load(train_path, global)
	
	if err != nil{
		return 0.5, nil, err
	}
	
	test_dataset := NewRealDataSet()
	err = test_dataset.Load(test_path, global)
	if err != nil{
		return 0.5, nil, err
	}
	regressor.Init(params)
	rmse, predictions := RegAlgorithmRunOnDataSet(regressor, train_dataset, test_dataset, pred_path, params)

	return rmse, predictions, nil
}

func RegAlgorithmTrain(regressor Regressor, train_path string, params map[string]string) (error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	train_dataset := NewRealDataSet()

	err := train_dataset.Load(train_path, global)
	
	if err != nil{
		return err
	}

	regressor.Init(params)
	regressor.Train(train_dataset)

	model_path, _ := params["model"]

	if model_path != "" {
		regressor.SaveModel(model_path)
	}

	return nil
}

func RegAlgorithmTest(regressor Regressor, test_path string, pred_path string, params map[string]string) (float64, []*RealPrediction, error) {
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	
	model_path, _ := params["model"]
	regressor.Init(params)
	if model_path != "" {
		regressor.LoadModel(model_path)
	} else {
		return 0.0, nil, nil
	}

	test_dataset := NewRealDataSet()
	err := test_dataset.Load(test_path, global)
	if err != nil{
		return 0.0, nil, err
	}
	
	rmse, predictions := RegAlgorithmRunOnDataSet(regressor, nil, test_dataset, pred_path, params)

	return rmse, predictions, nil
}

func RegAlgorithmRunOnDataSet(regressor Regressor, train_dataset, test_dataset *RealDataSet, pred_path string, params map[string]string) (float64, []*RealPrediction) {
	
	if train_dataset != nil {
		regressor.Train(train_dataset)
	}

	predictions := []*RealPrediction{}
	var pred_file *os.File
	if pred_path != ""{
		pred_file, _ = os.Create(pred_path)
	}
	for _,sample := range test_dataset.Samples {
		prediction := regressor.Predict(sample)
		if pred_file != nil{
			pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
		}
		predictions = append(predictions, &RealPrediction{Value: sample.Value, Prediction: prediction})
	}
	if pred_path != ""{
		defer pred_file.Close()
	}
		
	rmse := RegRMSE(predictions)
	return rmse, predictions
}

