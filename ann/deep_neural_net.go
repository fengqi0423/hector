package ann

import (
	"fmt"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"github.com/xlvector/hector/eval"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

type DeepNetParams struct {
	LearningRate         float64
	LearningRateDiscount float64
	Regularization       float64
	Hidden               []int64
	Classes              int64
	Epoches              int64
	Verbose              int64
	Dropout_rate_input   float64 // Input layer dropout rate
	Dropout_rate         float64 // Hidden layer dropout rate
}

type DeepNet struct {
	Weights  []*core.Matrix
	Params   DeepNetParams
	ValidationSet  *core.DataSet
}

func (algo *DeepNet) RandomInitVector(dim int64) *core.Vector {
	v := core.NewVector()
	d := math.Sqrt(float64(dim))
	var i int64
	for i = 0; i < dim; i++ {
		v.SetValue(i, (rand.Float64()-0.5)/d)
	}
	return v
}

func (algo *DeepNet) SaveModel(path string) {

}

func (algo *DeepNet) LoadModel(path string) {

}

func (algo *DeepNet) Init(params map[string]string) {
	algo.Params.LearningRate, _         = strconv.ParseFloat(params["learning-rate"], 64)
	algo.Params.LearningRateDiscount, _ = strconv.ParseFloat(params["learning-rate-discount"], 64)
	algo.Params.Regularization, _       = strconv.ParseFloat(params["regularization"], 64)
	algo.Params.Dropout_rate, _         = strconv.ParseFloat(params["dropout-rate"], 64)
	algo.Params.Dropout_rate_input, _   = strconv.ParseFloat(params["input-dropout-rate"], 64)

	algo.Params.Classes, _ = strconv.ParseInt(params["classes"], 10, 32)
	algo.Params.Epoches, _ = strconv.ParseInt(params["steps"], 10, 32)
	algo.Params.Verbose, _ = strconv.ParseInt(params["verbose"], 10, 32)

	hidden := strings.Split(params["hidden"], ",")
	algo.Params.Hidden = make([]int64, len(hidden))
	algo.Weights = make([]*core.Matrix, len(hidden)+1)
	for i := range hidden {
	 	algo.Params.Hidden[i], _ = strconv.ParseInt(hidden[i], 10, 32)
	}

	global_bias, _ := strconv.ParseInt(params["global"], 10, 64)
	validation_set := core.NewDataSet()
	validation_path, ok := params["validation_path"]

	if algo.Params.Verbose == 1 && ok {
		err := validation_set.Load(validation_path, global_bias)
		if err != nil {
			validation_set = nil
		}
		algo.ValidationSet = validation_set
	}
}

func (algo *DeepNet) PredictMultiClass(sample *core.Sample) *core.ArrayVector {
	// Input layer -> first hidden layer
	h := core.NewVector()
	weights := algo.Weights[0]
	for i:=int64(0); i < algo.Params.Hidden[0]; i++ {
		sum := float64(0.0)
		for _, f := range sample.Features {
			sum += f.Value * weights.GetValue(i, f.Id)
		}
		h.SetValue(i, util.Sigmoid(sum))
	}

	var y *core.Vector
	var dim int64
	for l := 1; l < len(algo.Weights); l++ {
		weights = algo.Weights[l]
		y = core.NewVector()
		h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer

		if l == len(algo.Weights)-1 {
			dim = algo.Params.Classes
		} else {
			dim = algo.Params.Hidden[l]
		}
        
		for i := int64(0); i <= dim; i++ {
			sum := float64(0.0)
			for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
				sum += h.GetValue(j) * weights.GetValue(i, j)
			}
			y.SetValue(i, sum)
		}
		h = y
	}

	z := core.NewArrayVector()
	for k, v := range y.Data {
		z.SetValue(int(k), v)
	}

	z = z.SoftMaxNorm()
	return z
}


func (algo *DeepNet) PredictMultiClassWithDropout(sample *core.Sample, dropout [][]int) []*core.Vector {
	// Input layer -> first hidden layer
	ret := make([]*core.Vector, len(algo.Weights))
	h := core.NewVector()
	weights := algo.Weights[0]
	in_dropout := dropout[0]
	out_dropput := dropout[1]
	for i:=int64(0); i < algo.Params.Hidden[0]; i++ {
		if out_dropput[i] == 1 {
			h.SetValue(i, 0)
		} else {
			sum := float64(0.0)
			for _, f := range sample.Features {
				if in_dropout[f.Id] != 0 {
					sum += f.Value * weights.GetValue(i, f.Id)
				}
			}
			h.SetValue(i, util.Sigmoid(sum))
		}
	}
	ret[0] = h.Copy()

	var y *core.Vector
	var dim int64
	for l := 1; l < len(algo.Weights); l++ {
		in_dropout = dropout[l]
		weights = algo.Weights[l]
		y = core.NewVector()
		h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer

		if l == len(algo.Weights)-1 {
			dim = algo.Params.Classes
			for i := int64(0); i <= dim; i++ {
				sum := float64(0.0)
				for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
					if in_dropout[j] != 0 {
						sum += h.GetValue(j) * weights.GetValue(i, j)
					}
				}
				y.SetValue(i, sum)
			}
		} else {
			out_dropput = dropout[l+1]
			for i := int64(0); i <= dim; i++ {
				if out_dropput[i] == 1 {
					y.SetValue(i, 0)
				} else {
					sum := float64(0.0)
					for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
						if in_dropout[j] != 0 {
							sum += h.GetValue(j) * weights.GetValue(i, j)
						}
					}
					y.SetValue(i, sum)
				}
			}
			dim = algo.Params.Hidden[l]
		}
		h = y
        ret[l] = h.Copy()
	}

	return ret // Contains activities of hidden layers and the final output layer
}

func (algo *DeepNet) Train(dataset *core.DataSet) {
	var weights *core.Matrix
	var dim int64

	// Initialize the first layer of weights
	algo.Weights[0] = core.NewMatrix()
	weights = algo.Weights[0]
	for i := int64(0); i < algo.Params.Hidden[0]; i++ {
		weights.Data[i] = core.NewVector()
	}
	initalized := make(map[int64]int)
	max_label := int64(0)
	for _, sample := range dataset.Samples {
		for _, f := range sample.Features {
			_, ok := initalized[f.Id]
			if !ok {
				for i := int64(0); i < algo.Params.Hidden[0]; i++ {
					weights.SetValue(i, f.Id, (rand.Float64()-0.5)/math.Sqrt(float64(algo.Params.Hidden[0])))
				}
				initalized[f.Id] = 1
				if f.Id > max_label {
					max_label = f.Id
				}
			}
		}
	}
	// Initialize other layers
	for l := 1; l < len(algo.Weights); l++ {
		algo.Weights[l] = core.NewMatrix()
		weights = algo.Weights[l]
		if l == len(algo.Weights)-1 {
			dim = algo.Params.Classes
		} else {
			dim = algo.Params.Hidden[l]
		}
		for i := int64(0); i < dim; i++ {
			weights.Data[i] = algo.RandomInitVector(algo.Params.Hidden[l-1] + 1)
		}
	}

	dropout := make([][]int, len(algo.Params.Hidden)+1)
	dropout[0] = make([]int, max_label+1) // initialized to zero. +1 for the offset
	for i := 1; i < len(algo.Params.Hidden)+1; i++ {
		dropout[i] = make([]int, algo.Params.Hidden[i]+1) // initialized to zero. +1 for the offset
	}

    L := len(algo.Weights)
	total := len(dataset.Samples)
	for epoch := int64(0); epoch < algo.Params.Epoches; epoch++ {
		if algo.Params.Verbose <= 0 {
			fmt.Printf(".")
		}
		counter := 0
		for _, sample := range dataset.Samples {
			if algo.Params.Dropout_rate_input > 0.0 {
				for i:=int64(0); i<max_label+1; i++{
					dropout[0][i] = rand.Intn(2)
				}
			}

			if algo.Params.Dropout_rate > 0.0 {
				for j:=1; j<len(algo.Params.Hidden)+1; j++ { 
					for i := int64(0); i <= algo.Params.Hidden[j]+1; i++ {
						dropout[j][i] = rand.Intn(2)
					}
				}
			}

			y := algo.PredictMultiClassWithDropout(sample, dropout)

			// Output layer error signal
			dy := core.NewVector()
			y_true := y[L-1].GetValue(int64(sample.Label))
			dy.SetValue(int64(sample.Label), y_true*(1-y_true))

			for l := L-1; l > 0 ; l-- { // Weights layer 1 to L-1, no layer 0 yet
				weights = algo.Weights[l]
				h  := y[l-1]
				dh := algo.Params.Hidden[l-1] // Dim of lower hidden layer
				var dg int64                    // Dim of upper hidden layer
				if l == L-1 {
					dg = algo.Params.Classes
				} else {
					dg = algo.Params.Hidden[l]
				}

				dyy := core.NewVector()
				for i:=int64(0); i<dh; i++{
					sum := 0.0
					for j:=int64(0); j<dg; j++{
						sum += dy.GetValue(j) * h.GetValue(i) * (1-h.GetValue(i)) * weights.GetValue(i, j)
					}
					dyy.SetValue(i, sum)
				}

				for i:=int64(0); i<dg; i++{
					for j:=int64(0); j<dh+1; j++{
						dw := dy.GetValue(i)*h.GetValue(j)
						w  := weights.GetValue(j, i) + algo.Params.LearningRate*dw
						weights.SetValue(j, i, w)
					}
				}
				dy = dyy
			}

			// Weight layer 0 delta
			weights = algo.Weights[0]
			for i:=int64(0); i<algo.Params.Hidden[0]; i++{
				for _, f := range sample.Features {
					dw := dy.GetValue(i)*f.Value
					w  := weights.GetValue(f.Id, i) + algo.Params.LearningRate*dw
					weights.SetValue(f.Id, i, w)
				}
			}

			counter++
			if algo.Params.Verbose > 0 && counter%2000 == 0 {
				fmt.Printf("Epoch %d %f%%\n", epoch+1, float64(counter)/float64(total)*100)
			}
		}

		if algo.Params.Verbose > 0 && algo.ValidationSet != nil {
			algo.Evaluate(algo.ValidationSet)
		}
		algo.Params.LearningRate *= algo.Params.LearningRateDiscount
	}
	if algo.Params.Dropout_rate_input != 0.0 {
		algo.Weights[0].Scale(1-algo.Params.Dropout_rate_input)
	}
	if algo.Params.Dropout_rate != 0.0 {
		for i:=1; i<len(algo.Params.Hidden)+1; i++ {
			algo.Weights[i].Scale(1-algo.Params.Dropout_rate)
		}
	}
	fmt.Println()
}

func (algo *DeepNet) Predict(sample *core.Sample) float64 {
	z := algo.PredictMultiClass(sample)
	return z.GetValue(1)
}

func (algo *DeepNet) Evaluate(dataset *core.DataSet) {
	accuracy := 0.0
	total := 0.0
	predictions := []*eval.LabelPrediction{}
	for _, sample := range dataset.Samples {
		prediction := algo.PredictMultiClass(sample)
		label, _ := prediction.KeyWithMaxValue()
		if int(label) == sample.Label {
			accuracy += 1.0
		}
		total += 1.0
		predictions = append(predictions, &(eval.LabelPrediction{Label: sample.Label, Prediction: prediction.GetValue(1)}))
	}
	fmt.Printf("accuracy %f%%\n", accuracy/total*100)
	auc := eval.AUC(predictions)
	fmt.Printf("AUC of class 1: %f\n", auc)
}
