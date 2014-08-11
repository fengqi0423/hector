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
	for i := int64(0); i < dim; i++ {
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
	validation_path, ok := params["validation_path"]

	if algo.Params.Verbose == 1 && ok {
		validation_set := core.NewDataSet()
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
	for l := 1; l < len(algo.Weights)-1; l++ {
		weights = algo.Weights[l]
		y = core.NewVector()
		h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer

		for i := int64(0); i <= algo.Params.Hidden[l]; i++ {
			sum := float64(0.0)
			for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
				sum += h.GetValue(j) * weights.GetValue(i, j)
			}
			y.SetValue(i, util.Sigmoid(sum))
		}
		h = y
	}

	l := len(algo.Weights)-1
	weights = algo.Weights[l]
	y = core.NewVector()
	h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer

	for i := int64(0); i <= algo.Params.Classes; i++ {
		sum := float64(0.0)
		for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
			sum += h.GetValue(j) * weights.GetValue(i, j)
		}
		y.SetValue(i, sum)
	}
	z := core.NewArrayVector()
	for k, v := range y.Data {
		z.SetValue(int(k), v)
	}

	z = z.SoftMaxNorm()
	return z
}


func (algo *DeepNet) PredictMultiClassWithDropout(sample *core.Sample, dropout []*core.Vector) []*core.Vector {
	// Input layer -> first hidden layer
	L := len(algo.Weights)
	ret := make([]*core.Vector, L)
	h := core.NewVector()
	weights := algo.Weights[0]
	in_dropout := dropout[0]
	out_dropput := dropout[1]
	for i:=int64(0); i < algo.Params.Hidden[0]; i++ {
		if out_dropput.GetValue(i) == 1 {
			h.SetValue(i, 0)
		} else {
			sum := float64(0.0)
			for _, f := range sample.Features {
				if in_dropout.GetValue(f.Id) == 0 {
					sum += f.Value * weights.GetValue(i, f.Id)
				}
			}
			h.SetValue(i, util.Sigmoid(sum))
		}
	}

	var y *core.Vector
	for l := 1; l < L-1; l++ {
		in_dropout = dropout[l]
		weights = algo.Weights[l]
		y = core.NewVector()
		h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer
		ret[l-1] = h
		out_dropput = dropout[l+1]
		for i := int64(0); i <= algo.Params.Hidden[l]; i++ {
			if out_dropput.GetValue(i) == 1 {
				y.SetValue(i, 0)
			} else {
				sum := float64(0.0)
				for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
					if in_dropout.GetValue(j) == 0 {
						sum += h.GetValue(j) * weights.GetValue(i, j)
					}
				}
				y.SetValue(i, util.Sigmoid(sum))
			}
		}
		h = y
	}

	l := L-1
	in_dropout = dropout[l]
	weights = algo.Weights[l]
	y = core.NewVector()
	h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer
	ret[l-1] = h

	for i := int64(0); i <= algo.Params.Classes; i++ {
		sum := float64(0.0)
		for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
			if in_dropout.GetValue(j) == 0 {
				sum += h.GetValue(j) * weights.GetValue(i, j)
			}
		}
		y.SetValue(i, sum)
	}

	ret[L-1] = y.SoftMaxNorm() // Output layer

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
			weights.Data[i] = algo.RandomInitVector(dim)
		}
	}

    L := len(algo.Weights)
	total := len(dataset.Samples)
	for epoch := int64(0); epoch < algo.Params.Epoches; epoch++ {
		if algo.Params.Verbose <= 0 {
			fmt.Printf(".")
		}
		counter := 0
		for _, sample := range dataset.Samples {
			dropout := make([]*core.Vector, L)
			for i := 0; i < L; i++ {
				dropout[i] = core.NewVector()
			}

			if algo.Params.Dropout_rate_input > 0.0 {
				for i:=int64(0); i<max_label+1; i++{
					if rand.Float64() < algo.Params.Dropout_rate_input {
						dropout[0].SetValue(i, 1)
					}
				}
			}

			if algo.Params.Dropout_rate > 0.0 {
				for j:=1; j<L; j++ { 
					for i := int64(0); i <= algo.Params.Hidden[j-1]; i++ {
						if rand.Float64() < algo.Params.Dropout_rate {
							dropout[j].SetValue(i, 1)
						}
					}
				}
			}

			y := algo.PredictMultiClassWithDropout(sample, dropout)

			// Output layer error signal
			dy := core.NewVector()
			for i:=int64(0); i<algo.Params.Classes; i++ {
				y_true := y[L-1].GetValue(i)
				if i == int64(sample.Label) {
					dy.SetValue(i, 1-y_true)
				} else {
					dy.SetValue(i, -y_true)					
				}
			}

			var dropg *core.Vector // upper layer node dropout
			var droph *core.Vector // lower layer node dropout
			for l := L-1; l > 0 ; l-- { // Weights layer 1 to L-1, no layer 0 yet
				weights = algo.Weights[l]
				if l == L-1 {
					dropg = core.NewVector() // No dropout for the output layer
				} else {
					droph = dropout[l+1]
				}
				droph := dropout[l]
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
					if droph.GetValue(i) == 0 {
						for j:=int64(0); j<dg; j++{
							if dropg.GetValue(j) == 0 {
								sum += dy.GetValue(j) * h.GetValue(i) * (1-h.GetValue(i)) * weights.GetValue(j, i)
							}
						}
					}
					dyy.SetValue(i, sum)
				}

				for i:=int64(0); i<dg; i++{
					if dropg.GetValue(i) == 0 {
						for j:=int64(0); j<dh+1; j++{
							if droph.GetValue(j) == 0 {
								dw := dy.GetValue(i)*h.GetValue(j)
								w  := weights.GetValue(i, j) + algo.Params.LearningRate*dw
								weights.SetValue(i, j, w)
							}
						}
					}
				}
				dy = dyy
			}

			// Weight layer 0 delta
			dropg = dropout[1]
			droph = dropout[0]
			weights = algo.Weights[0]
			for i:=int64(0); i<algo.Params.Hidden[0]; i++{
				if dropg.GetValue(i) == 0 {
					for _, f := range sample.Features {
						if droph.GetValue(f.Id) == 0 {
							dw := dy.GetValue(i)*f.Value
							w  := weights.GetValue(i, f.Id) + algo.Params.LearningRate*dw
							weights.SetValue(i, f.Id, w)
						}
					}
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
