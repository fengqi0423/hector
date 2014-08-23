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
	LearningRate         float64
	LearningRateDiscount float64
	Regularization       float64
	Momentum             float64
	Batch                int64
	Hidden               []int64
	Classes              int64
	InputDim             int64
	Epoches              int64
	Verbose              int64
	Dropout_rate_input   float64
	Dropout_rate         float64
}

type DeepNet struct {
	LoadedModel     bool
	Weights        [][][]float64
	Params         DeepNetParams
	ValidationSet  *core.DataSet
}

func (algo *DeepNet) RandomInitArray(input_dim int64) []float64 {
	w := make([]float64, input_dim)
	d := math.Sqrt(float64(input_dim))
	for i:=int64(0); i < input_dim; i++ {
		w[i] = (rand.Float64()-0.5)/d
	}
	return w
}

func (algo *DeepNet) ScaleWeights(weights [][]float64, scale float64) {
	d1 := len(weights)
	d2 := len(weights[0])
	for i:=0; i<d1; i++{
		for j:=0; j<d2; j++{
			weights[i][j] = weights[i][j]*scale		
		}
	}
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
				sb.Float(weights[p][q])
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
	algo.Weights = make([][][]float64, len(hidden)+1)
	for i := range hidden {
	 	algo.Params.Hidden[i], _ = strconv.ParseInt(hidden[i], 10, 32)
	}
	// output
	scanner.Scan()
	algo.Params.Classes, _ = strconv.ParseInt(scanner.Text(), 10, 32)
	//Weights
	for i:=0; i<len(algo.Params.Hidden)+1; i++ {
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
		algo.Weights[i] = make([][]float64, up)
		weights := algo.Weights[i]
		for p:=int64(0); p<up; p++{
			scanner.Scan()
			parts := strings.Split(scanner.Text(), " ")
			weights[p] = make([]float64, down+1)
			for q:=int64(0); q<down+1; q++{
				u := parts[q+1]
				v, _ := strconv.ParseFloat(strings.Split(u, ":")[1], 64)
				weights[p][q] = v
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
	algo.Params.Batch  , _ = strconv.ParseInt(params["batch"], 10, 32)

	hidden := strings.Split(params["hidden"], ",")
	algo.Params.Hidden = make([]int64, len(hidden))
	algo.Weights = make([][][]float64, len(hidden)+1)
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
	h := core.NewArrayVector()
	weights := algo.Weights[0]
	for i:=int64(0); i < algo.Params.Hidden[0]; i++ {
		sum := float64(0.0)
		for _, f := range sample.Features {
			sum += f.Value * weights[i][f.Id]
		}
		h.SetValue(int(i), util.Sigmoid(sum))
	}

	var y *core.ArrayVector
	var dim int64
	L := len(algo.Weights)
	for l := 1; l < L; l++ {
		if l == L-1 {
			dim = algo.Params.Classes
		} else {
			dim = algo.Params.Hidden[l]
		}

		weights = algo.Weights[l]
		y = core.NewArrayVector()
		h.SetValue(int(algo.Params.Hidden[l-1]), 1) // Offset neuron for hidden layer

		for i := int64(0); i < dim; i++ {
			sum := float64(0.0)
			for j := int64(0); j <= algo.Params.Hidden[l-1]; j++ {
				sum += h.GetValue(int(j)) * weights[i][j]
			}
			y.SetValue(int(i), util.Sigmoid(sum))
		}
		h = y
	}

	return y.SoftMaxNorm()
}


func (algo *DeepNet) PredictMultiClassWithDropout(sample *core.Sample, dropout [][]int) []*core.ArrayVector {
	// Input layer -> first hidden layer
	L := len(algo.Weights)
	ret := make([]*core.ArrayVector, L)
	h := core.NewArrayVector()
	weights := algo.Weights[0]
	in_dropout := dropout[0]
	out_dropput := dropout[1]
	for i:=0; i < int(algo.Params.Hidden[0]); i++ {
		if out_dropput[i] == 1 {
			h.SetValue(i, 0)
		} else {
			sum := float64(0.0)
			for _, f := range sample.Features {
				if in_dropout[f.Id] == 0 {
					sum += f.Value * weights[i][f.Id]
				}
			}
			h.SetValue(i, util.Sigmoid(sum))
		}
	}

	var y *core.ArrayVector
	var dim int64
	for l := 1; l < L; l++ {
		in_dropout = dropout[l]
		if l == L-1 {
			out_dropput = make([]int, algo.Params.Classes)
		} else {
			out_dropput = dropout[l+1]
		}
		weights = algo.Weights[l]
		y = core.NewArrayVector()
		h.SetValue(int(algo.Params.Hidden[l-1]), 1) // Offset neuron for hidden layer
		ret[l-1] = h
		if l == L-1 {
			dim = algo.Params.Classes
		} else {
			dim = algo.Params.Hidden[l]
		}
		for i := 0; i < int(dim); i++ {
			if out_dropput[i] == 1 {
				y.SetValue(i, 0)
			} else {
				sum := float64(0.0)
				for j := 0; j <= int(algo.Params.Hidden[l-1]); j++ {
					if in_dropout[j] == 0 {
						sum += h.GetValue(j) * weights[i][j]
					}
				}
				y.SetValue(i, util.Sigmoid(sum))
			}
		}
		h = y
	}

	ret[L-1] = y.SoftMaxNorm() // Output layer

	return ret // Contains activities of hidden layers and the final output layer
}

func (algo *DeepNet) GetDelta(samples []*core.Sample, dropout [][]int) [][][]float64{
	// Give a batch of samples, return accumulated dw without changing w
	L := len(algo.Params.Hidden)+1
	adws := make([][][]float64, L)
	var weights [][]float64
	var adw [][]float64
	var in_dim, out_dim int64
	for i := 0; i<L; i++ {
		if i == 0 {
			in_dim = algo.Params.InputDim
		} else {
			in_dim = algo.Params.Hidden[i-1]
		}
		if i == L-1 {
			out_dim = algo.Params.Classes
		} else {
			out_dim = algo.Params.Hidden[i]
		}
		adws[i] = make([][]float64, out_dim)
		for j:=int64(0); j<out_dim; j++ {
			adws[i][j] = make([]float64, in_dim+1)
		}
	}
	for _, sample := range samples {
		y := algo.PredictMultiClassWithDropout(sample, dropout)

		// Output layer error signal
		dy := core.NewArrayVector()
		for i:=0; i<int(algo.Params.Classes); i++ {
			y_hat := y[L-1].GetValue(i)
			if i == sample.Label {
				dy.SetValue(i, 1-y_hat)
			} else {
				dy.SetValue(i, -y_hat)					
			}
		}

		var dropg []int // upper layer node dropout
		var droph []int // lower layer node dropout
		for l := L-1; l > 0 ; l-- { // Weights layer 1 to L-1, no layer 0 yet
			weights = algo.Weights[l]
			adw = adws[l]
			h  := y[l-1]
			dh := int(algo.Params.Hidden[l-1]) // Dim of lower hidden layer
			var dg int                    // Dim of upper hidden layer
			if l == L-1 {
				dropg = make([]int, algo.Params.Classes) // No dropout for the output layer
				dg = int(algo.Params.Classes)
			} else {
				dropg = dropout[l+1]
				dg = int(algo.Params.Hidden[l])
			}
			droph := dropout[l]

			dyy := core.NewArrayVector()
			for i:=0; i<dh; i++{
				sum := 0.0
				if droph[i] == 0 {
					for j:=0; j<dg; j++{
						if dropg[j] == 0 {
							sum += dy.GetValue(j) * h.GetValue(i) * (1-h.GetValue(i)) * weights[j][i]
						}
					}
				}
				dyy.SetValue(i, sum)
			}

			for i:=0; i<dg; i++{
				if dropg[i] == 0 {
					for j:=0; j<dh+1; j++{
						if droph[j] == 0 {
							dw := dy.GetValue(i)*h.GetValue(j)
							adw[i][j] = adw[i][j]+dw
						}
					}
				}
			}
			dy = dyy
		}

		// Weight layer 0 delta
		dropg = dropout[1]
		droph = dropout[0]
		adw = adws[0]
		for i:=0; i<int(algo.Params.Hidden[0]); i++{
			if dropg[i] == 0 {
				for _, f := range sample.Features {
					if droph[f.Id] == 0 {
						dw := dy.GetValue(i)*f.Value
						adw[i][f.Id] = adw[i][f.Id]+dw
					}
				}
			}
		}
	}
	return adws
}

func (algo *DeepNet) Train(dataset *core.DataSet) {
	var weights [][]float64
	var dweights [][]float64
	var pdweights [][]float64
	var dWeights [][][]float64
	var in_dim, out_dim int64
	var mv, ft float64 
	L := len(algo.Weights)
	total := int64(len(dataset.Samples))
	previousdWeights := make([][][]float64, L)

	if !algo.LoadedModel {
		max_label := int64(0)
		for _, sample := range dataset.Samples {
			for _, f := range sample.Features {
				if f.Id > max_label {
					max_label = f.Id
				}
			}
		}
		algo.Params.InputDim = max_label
		fmt.Printf("Found %d input dimensions.\n", algo.Params.InputDim)

		for l := 0; l < L; l++ {
			if l == L-1 {
				out_dim = algo.Params.Classes
			} else {
				out_dim = algo.Params.Hidden[l]
			}
			if l == 0 {
				in_dim = algo.Params.InputDim
			} else {
				in_dim = algo.Params.Hidden[l-1]
			}
			algo.Weights[l] = make([][]float64, out_dim)
			previousdWeights[l] = make([][]float64, out_dim)
			for i := int64(0); i < out_dim; i++ {
				algo.Weights[l][i] = algo.RandomInitArray(in_dim+1)
				previousdWeights[l][i] = make([]float64, in_dim+1)
			}
		}
	}

	if algo.Params.Momentum > 0 {
		mv = algo.Params.Momentum
		ft = 1 - algo.Params.Momentum
	} else {
		mv = 0
		ft = 1
	}

	dropout := make([][]int, L)
	dropout[0] = make([]int, algo.Params.InputDim+1)
	for i := 1; i < L; i++ {
		dropout[i] = make([]int, algo.Params.Hidden[i-1]+1)
	}

	for epoch := int64(0); epoch < algo.Params.Epoches; epoch++ {
		if algo.Params.Verbose <= 0 {
			fmt.Printf(".")
		}
		counter := 0
		for i := int64(0); i < total; i += algo.Params.Batch {
			var samples []*core.Sample
			if i + algo.Params.Batch <= total {
				samples = dataset.Samples[i:i+algo.Params.Batch]
			} else {
				samples = dataset.Samples[i:total]
			}


			if algo.Params.Dropout_rate_input > 0.0 {
				for i:=int64(0); i<=algo.Params.InputDim; i++{
					if rand.Float64() < algo.Params.Dropout_rate_input {
						dropout[0][i] = 1
					} else {
						dropout[0][i] = 0
					}
				}
			}

			if algo.Params.Dropout_rate > 0.0 {
				for j:=1; j<L; j++ { 
					for i := int64(0); i <= algo.Params.Hidden[j-1]; i++ {
						if rand.Float64() < algo.Params.Dropout_rate {
							dropout[j][i] = 1
						} else {
							dropout[j][i] = 0
						}
					}
				}
			}

			dWeights = algo.GetDelta(samples, dropout)

			for i:=0; i<L; i++ {
				weights   = algo.Weights[i]
				dweights  = dWeights[i]
				pdweights = previousdWeights[i]
				if i == L-1 {
					out_dim = algo.Params.Classes
				} else {
					out_dim = algo.Params.Hidden[i]
				}
				if i == 0 {
					in_dim = algo.Params.InputDim
				} else {
					in_dim = algo.Params.Hidden[i-1]
				}
				for p:=int64(0); p<out_dim; p++ {
					for q:=int64(0); q<in_dim+1; q++ {
						dw := dweights[p][q]
						if dw != 0 { // Especially for layer 0 - When input is sparse, avoid lots of zero updates
							w  := weights[p][q]
							if algo.Params.Momentum > 0 {
								pdw := pdweights[p][q]
								dw = pdw * mv + dw * ft
							}
							w = w + algo.Params.LearningRate * dw - algo.Params.Regularization * w
							weights[p][q] = w
						}
					}
				}
			}

			previousdWeights = dWeights

			counter += int(algo.Params.Batch)
			if algo.Params.Verbose > 0 && counter % (10*int(algo.Params.Batch)) == 0 {
				fmt.Printf("Epoch %d %f%%\n", epoch+1, float64(counter)/float64(total)*100)
			}
		}

		if algo.Params.Verbose > 0 && algo.ValidationSet != nil {
			algo.Evaluate(algo.ValidationSet)
		}
		algo.Params.LearningRate *= algo.Params.LearningRateDiscount
	}
	if algo.Params.Dropout_rate_input != 0.0 {
		algo.ScaleWeights(algo.Weights[0], (1-algo.Params.Dropout_rate_input))
	}
	if algo.Params.Dropout_rate != 0.0 {
		for i:=1; i<L; i++ {
			algo.ScaleWeights(algo.Weights[i], 1-algo.Params.Dropout_rate)
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
