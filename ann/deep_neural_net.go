package ann

import (
	"os"
	"fmt"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"github.com/xlvector/hector/eval"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"bufio"
)

type DeepNetParams struct {
	LearningRate         float64 // 0.1
	LearningRateDiscount float64  // 0.95 0.99
	Regularization       float64
	Momentum             float64
	Hidden               []int64 // [10,5,3]
	Classes              int64 // 2
	InputDim             int64
	Epoches              int64  // Steps
	Verbose              int64  // 1 0
	Dropout_rate_input   float64 // Input layer dropout rate 0.3
	Dropout_rate         float64 // Hidden layer dropout rate 0.5
}

type DeepNet struct {
	LoadedModel     bool
	Weights        []*core.Matrix
	Params         DeepNetParams
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
	// Saves model architecture hidden neurons + output neurons
	// And weights
	sb := util.StringBuilder{}
	sb.Int64(algo.Params.InputDim)
	sb.Write("\n")
	for i:=0;i<len(algo.Params.Hidden);i++{
		sb.Int64(algo.Params.Hidden[i])
		if i == len(algo.Params.Hidden)-1 {
			sb.Write("\n")
		} else {
			sb.Write(",")
		}
	}
	sb.Int64(algo.Params.Classes)
	sb.Write("\n")
	for i:=0; i<len(algo.Params.Hidden)+1; i++ {
		weights := algo.Weights[i]
		var up, down int64
		if i == len(algo.Params.Hidden) {
			up = algo.Params.Classes
		} else {
			up = algo.Params.Hidden[i]
		}
		if i == 0 {
			down = algo.Params.InputDim
		} else {
			down = algo.Params.Hidden[i-1]
		}
		for p:=int64(0); p<up; p++{
			sb.Int64(p)
			sb.Write(" ")
			for q:=int64(0); q<down+1; q++{
				sb.Int64(q)
				sb.Write(":")
				sb.Float(weights.GetValue(p, q))
				sb.Write(" ")
			}
			sb.Write("\n")
		}
	}
	sb.WriteToFile(path)
}

func (algo *DeepNet) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()
	scanner := bufio.NewScanner(file)
	// input
	scanner.Scan()
	algo.Params.InputDim, _ = strconv.ParseInt(scanner.Text(), 10, 32)
    // hidden structure
	scanner.Scan()
	hidden := strings.Split(scanner.Text(), ",")
	algo.Params.Hidden = make([]int64, len(hidden))
	algo.Weights = make([]*core.Matrix, len(hidden)+1)
	for i := range hidden {
	 	algo.Params.Hidden[i], _ = strconv.ParseInt(hidden[i], 10, 32)
	}
	// output
	scanner.Scan()
	algo.Params.Classes, _ = strconv.ParseInt(scanner.Text(), 10, 32)
	//Weights
	for i:=0; i<len(algo.Params.Hidden)+1; i++ {
		algo.Weights[i] = core.NewMatrix()
		weights := algo.Weights[i]
		var up, down int64
		if i == len(algo.Params.Hidden) {
			up = algo.Params.Classes
		} else {
			up = algo.Params.Hidden[i]
		}
		if i == 0 {
			down = algo.Params.InputDim
		} else {
			down = algo.Params.Hidden[i-1]
		}
		for p:=int64(0); p<up; p++{
			scanner.Scan()
			parts := strings.Split(scanner.Text(), " ")
			for q:=int64(0); q<down+1; q++{
				u := parts[q+1]
				v, _ := strconv.ParseFloat(strings.Split(u, ":")[1], 64)
				weights.SetValue(p, q, v)
			}
		}
	}
	algo.LoadedModel = true
}

func (algo *DeepNet) Init(params map[string]string) {
	algo.Params.LearningRate, _         = strconv.ParseFloat(params["learning-rate"], 64)
	algo.Params.LearningRateDiscount, _ = strconv.ParseFloat(params["learning-rate-discount"], 64)
	algo.Params.Regularization, _       = strconv.ParseFloat(params["regularization"], 64)
	algo.Params.Dropout_rate, _         = strconv.ParseFloat(params["dropout-rate"], 64)
	algo.Params.Dropout_rate_input, _   = strconv.ParseFloat(params["input-dropout-rate"], 64)
	algo.Params.Momentum, _             = strconv.ParseFloat(params["momentum"], 64)

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
	algo.LoadedModel = false
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

		for i := int64(0); i < algo.Params.Hidden[l]; i++ {
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

	for i := int64(0); i < algo.Params.Classes; i++ {
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
		out_dropput = dropout[l+1]
		weights = algo.Weights[l]
		y = core.NewVector()
		h.SetValue(algo.Params.Hidden[l-1], 1) // Offset neuron for hidden layer
		ret[l-1] = h
		for i := int64(0); i < algo.Params.Hidden[l]; i++ {
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

	for i := int64(0); i < algo.Params.Classes; i++ {
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
	var mv, ft float64 
	L := len(algo.Weights)
	total := len(dataset.Samples)
	pdw := make([]*core.Matrix, L) // previous dw, for momentum

	if !algo.LoadedModel {
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
						weights.SetValue(i, f.Id, (rand.Float64()-0.5)/math.Sqrt(float64(algo.Params.Hidden[0]))) // should use input dim
					}
					initalized[f.Id] = 1
					if f.Id > max_label {
						max_label = f.Id
					}
				}
			}
		}
		algo.Params.InputDim = max_label
		// Initialize other layers
		for l := 1; l < L; l++ {
			algo.Weights[l] = core.NewMatrix()
			weights = algo.Weights[l]
			if l == L-1 {
				dim = algo.Params.Classes
			} else {
				dim = algo.Params.Hidden[l]
			}
			for i := int64(0); i < dim; i++ {
				weights.Data[i] = algo.RandomInitVector(dim)//this should be input layer dim?
			}
		}
	}

	if algo.Params.Momentum > 0 {
		for i:=0; i<L; i++{
			pdw[i] = core.NewMatrix()
		}
		mv = algo.Params.Momentum
		ft = 1 - algo.Params.Momentum
	}

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
				for i:=int64(0); i<=algo.Params.InputDim; i++{
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
				y_hat := y[L-1].GetValue(i)
				if i == int64(sample.Label) {
					dy.SetValue(i, 1-y_hat)
				} else {
					dy.SetValue(i, -y_hat)					
				}
			}

			var dropg *core.Vector // upper layer node dropout
			var droph *core.Vector // lower layer node dropout
			for l := L-1; l > 0 ; l-- { // Weights layer 1 to L-1, no layer 0 yet
				weights = algo.Weights[l]
				if l == L-1 {
					dropg = core.NewVector() // No dropout for the output layer
				} else {
					dropg = dropout[l+1]
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
								wp := weights.GetValue(i, j)
								dw := dy.GetValue(i)*h.GetValue(j) - algo.Params.Regularization * wp
								if algo.Params.Momentum > 0 {
									dw = pdw[l].GetValue(i, j)*mv + dw*ft
									pdw[l].SetValue(i, j, dw)
								}
								w  := wp + algo.Params.LearningRate*dw
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
							wp := weights.GetValue(i, f.Id)
							dw := dy.GetValue(i)*f.Value - algo.Params.Regularization * wp
							if algo.Params.Momentum > 0 {
								dw = pdw[0].GetValue(i, f.Id)*mv + dw*ft
								pdw[0].SetValue(i, f.Id, dw)
							}
							w  := wp + algo.Params.LearningRate*dw
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
		for i:=1; i<L; i++ {
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
