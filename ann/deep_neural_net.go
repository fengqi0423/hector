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
	"runtime"
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
	Activation           int64
	InitWeight           int64
}

type DeepNet struct {
	LoadedModel     bool
	Weights        [][][]float64
	Params         DeepNetParams
	ValidationSet  *core.DataSet
}

func (algo *DeepNet) RandomArray(lower float64, upper float64, size int64) []float64 {
	// Generate an array of random numbers evenly sampled between lower and upper bound
	a := make([]float64, size)
	l := upper - lower
	for i:=int64(0); i < size; i++ {
		a[i] = lower + rand.Float64()*l
	}
	return a
}

func (algo *DeepNet) InitializeWeights(input_dim int64, output_dim int64) [][]float64 {
	w := make([][]float64, output_dim)
	lower := -1.0
	upper := 1.0
	switch {
	case algo.Params.InitWeight == 0: 
		upper = 1.0/math.Sqrt(float64(input_dim))
		lower = -upper
	case algo.Params.InitWeight == 1:  // normalized initialization Clorot and Bengio 2010
		upper = math.Sqrt(6.0/float64(input_dim + output_dim))
		lower = -upper
	case algo.Params.InitWeight == 2: // my experience this works better with sigmoid activation
		upper = 1.0/math.Sqrt(float64(output_dim))
		lower = -upper
	}

	for i:=int64(0); i<output_dim; i++ {
		w[i] = algo.RandomArray(lower, upper, input_dim)
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

	algo.Params.Classes   , _ = strconv.ParseInt(params["classes"], 10, 32)
	algo.Params.Epoches   , _ = strconv.ParseInt(params["steps"], 10, 32)
	algo.Params.Verbose   , _ = strconv.ParseInt(params["verbose"], 10, 32)
	algo.Params.Batch     , _ = strconv.ParseInt(params["batch"], 10, 32)
	algo.Params.Activation, _ = strconv.ParseInt(params["activation"], 10, 32)
	algo.Params.InitWeight, _ = strconv.ParseInt(params["init-weight"], 10, 32)

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

func (algo *DeepNet) Activate(x float64) float64 {
	switch {
	case algo.Params.Activation == 1:
		return util.Sigmoid(x)
	case algo.Params.Activation == 2:
		return util.Tanh(x)
	}
	return util.Sigmoid(x)
}

func (algo *DeepNet) DerivativeOfActivation(x float64) float64 {
	switch {
	case algo.Params.Activation == 1:
		return 1 * (1-x)
	case algo.Params.Activation == 2:
		return 1 - x*x
	}
	return 1 * (1-x)
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
		h.SetValue(int(i), algo.Activate(sum))
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
			if l == L-1 {
				y.SetValue(int(i), sum)
			} else {
				y.SetValue(int(i), algo.Activate(sum))
			}
		}
		h = y
	}

	return y.SoftMaxNorm()
}


func (algo *DeepNet) PredictMultiClassWithDropout(sample *core.Sample, dropout [][]int, activities [][]float64) {
	// Input layer -> first hidden layer
	L := len(algo.Weights)
	h := activities[0]
	g := activities[0]
	weights := algo.Weights[0]
	in_dropout  := dropout[0]
	out_dropput := dropout[1]
	for i:=0; i < int(algo.Params.Hidden[0]); i++ {
		if out_dropput[i] == 0 {
			sum := float64(0.0)
			for _, f := range sample.Features {
				if in_dropout[f.Id] == 0 {
					sum += f.Value * weights[i][f.Id]
				}
			}
			g[i] = algo.Activate(sum)
		} else {
			g[i] = 0
		}
	}

	var dim int64
	for l := 1; l < L; l++ {
		in_dropout = dropout[l]
		out_dropput = dropout[l+1]
		weights = algo.Weights[l]
		g = activities[l]
		h = activities[l-1]
		h[algo.Params.Hidden[l-1]] = 1 // Offset neuron for hidden layer

		if l == L-1 {
			dim = algo.Params.Classes
		} else {
			dim = algo.Params.Hidden[l]
		}
		for i := 0; i < int(dim); i++ {
			if out_dropput[i] == 1 {
				g[i] = 0
			} else {
				sum := float64(0.0)
				for j := 0; j <= int(algo.Params.Hidden[l-1]); j++ {
					if in_dropout[j] == 0 {
						sum += h[j] * weights[i][j]
					}
				}
				if l == L - 1 {
					g[i] = sum
				} else {
					g[i] = algo.Activate(sum)
				}
			}
		}
	}

	// Softmax output layer
	sum := 0.0
	for i:=int64(0); i<algo.Params.Classes; i++{
		g[i] = math.Exp(g[i])
		sum += g[i]
	}
	for i:=int64(0); i<algo.Params.Classes; i++{
		g[i] = g[i]/sum
	}
}

func (algo *DeepNet) GetDelta(samples []*core.Sample, dropout [][]int, adws [][][]float64, signal [][]float64, activities [][]float64, feature_ids map[int64]int){
	// Give a batch of samples, return accumulated dw without changing w
	L := len(algo.Weights)
	var weights [][]float64
	var adw [][]float64

	for _, sample := range samples {
		algo.PredictMultiClassWithDropout(sample, dropout, activities)

		// Output layer error signal
		for i:=0; i<int(algo.Params.Classes); i++ {
			y_hat := activities[L-1][i]
			if i == sample.Label {
				signal[L-1][i] = 1-y_hat
			} else {
				signal[L-1][i] = -y_hat
			}
		}

		var dropg []int // upper layer node dropout
		var droph []int // lower layer node dropout
		for l := L-1; l > 0 ; l-- { // Weights layer 1 to L-1, no layer 0 yet
			weights = algo.Weights[l]
			adw = adws[l]
			h  := activities[l-1]
			dh := int(algo.Params.Hidden[l-1]) // Dim of lower hidden layer
			var dg int                    // Dim of upper hidden layer
			if l == L-1 {
				dg = int(algo.Params.Classes)
			} else {
				dg = int(algo.Params.Hidden[l])
			}
			droph = dropout[l]
			dropg = dropout[l+1]
			dyh := signal[l-1]
			dyg := signal[l]
			for i:=0; i<dh; i++{
				sum := 0.0
				if droph[i] == 0 {
					for j:=0; j<dg; j++{
						if dropg[j] == 0 {
							sum += dyg[j] * algo.DerivativeOfActivation(h[i]) * weights[j][i]
						}
					}
					dyh[i] = sum
				}
			}

			for i:=0; i<dg; i++{
				if dropg[i] == 0 {
					for j:=0; j<=dh; j++{
						if droph[j] == 0 {
							dw := dyg[i]*h[j]
							adw[i][j] = adw[i][j]+dw
						}
					}
				}
			}
		}

		// Weight layer 0 delta
		dropg = dropout[1]
		droph = dropout[0]
		adw = adws[0]
		for i:=0; i<int(algo.Params.Hidden[0]); i++{
			if dropg[i] == 0 {
				for _, f := range sample.Features {
					if droph[f.Id] == 0 {
						feature_ids[f.Id] = 1
						dw := signal[0][i] * f.Value
						adw[i][f.Id] = adw[i][f.Id]+dw
					}
				}
				if droph[algo.Params.InputDim] == 0 {
					feature_ids[algo.Params.InputDim] = 1
					dw := signal[0][i]
					adw[i][algo.Params.InputDim] = adw[i][algo.Params.InputDim]+dw
				}
			}
		}

		// clear activity buffer - it only stores hidden and output layer, so this should be fast
		for i:=0; i<len(algo.Params.Hidden); i++ {
			for j:=int64(0); j<=algo.Params.Hidden[i]; j++ {
				activities[i][j] = 0
			}
		}
		for j:=int64(0); j<algo.Params.Classes; j++ {
			activities[L-1][j] = 0
		}
	}
}

func (algo *DeepNet) Train(dataset *core.DataSet) {
	gc_period := 10000
	var weights [][]float64
	var dweights [][]float64
	var pdweights [][]float64
	var in_dim, out_dim int64
	var mv, ft float64 
	L := len(algo.Weights)
	total := int64(len(dataset.Samples))
	report_count := int(total) / 100
	previousdWeights := make([][][]float64, L)
	dWeights := make([][][]float64, L)

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
			dWeights[l]         = make([][]float64, out_dim)
			algo.Weights[l]     = make([][]float64, out_dim)
			previousdWeights[l] = make([][]float64, out_dim)
			for i := int64(0); i < out_dim; i++ {
				dWeights[l][i]         = make([]float64, in_dim+1)
				previousdWeights[l][i] = make([]float64, in_dim+1)
			}
			algo.Weights[l] = algo.InitializeWeights(in_dim+1, out_dim)
		}
	}

	// momentum init
	if algo.Params.Momentum > 0 {
		mv = algo.Params.Momentum
		ft = 1 - algo.Params.Momentum
	} else {
		mv = 0
		ft = 1
	}

	//dropout init
	dropout := make([][]int, L+1)
	dropout[0] = make([]int, algo.Params.InputDim+1)
	for i := 1; i < L; i++ {
		dropout[i] = make([]int, algo.Params.Hidden[i-1]+1)
	}
	dropout[L] = make([]int, algo.Params.Classes)

	// buffer for backpropagation
	signal     := make([][]float64, L)
	activities := make([][]float64, L)
	for i := 0; i<L-1; i++ {
		signal[i]     = make([]float64, algo.Params.Hidden[i])
		activities[i] = make([]float64, algo.Params.Hidden[i]+1)
	}
	signal[L-1]     = make([]float64, algo.Params.Classes)
	activities[L-1] = make([]float64, algo.Params.Classes)

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

			feature_ids := make(map[int64]int)
			algo.GetDelta(samples, dropout, dWeights, signal, activities, feature_ids)

			weights   = algo.Weights[0]
			dweights  = dWeights[0]
			pdweights = previousdWeights[0]
			for p:=int64(0); p<algo.Params.Hidden[0]; p++ {
				for q, _ := range feature_ids {
					dw := dweights[p][q]
					w  := weights[p][q]
					if algo.Params.Momentum > 0 {
						pdw := pdweights[p][q]
						dw = pdw * mv + dw * ft
					}
					w = w + algo.Params.LearningRate * dw - algo.Params.Regularization * w
					weights[p][q] = w
					pdweights[p][q] = dw

					// clear cache
					dweights[p][q] = 0
				}
			}

			for i:=1; i<L; i++ {
				weights   = algo.Weights[i]
				dweights  = dWeights[i]
				pdweights = previousdWeights[i]
				if i == L-1 {
					out_dim = algo.Params.Classes
				} else {
					out_dim = algo.Params.Hidden[i]
				}
				in_dim = algo.Params.Hidden[i-1]
				for p:=int64(0); p<out_dim; p++ {
					for q:=int64(0); q<=in_dim; q++ {
						dw := dweights[p][q]
						if dw != 0 { // Especially for layer 0 - When input is sparse, avoid lots of zero updates
							w  := weights[p][q]
							if algo.Params.Momentum > 0 {
								pdw := pdweights[p][q]
								dw = pdw * mv + dw * ft
							}
							w = w + algo.Params.LearningRate * dw - algo.Params.Regularization * w
							weights[p][q] = w
							pdweights[p][q] = dw
							dweights[p][q] = 0 // clear cache
						} else {
							pdweights[p][q] = 0
						}
					}
				}
			}			

			counter += int(algo.Params.Batch)
			if algo.Params.Verbose > 0 && (counter % report_count == 0) {
				fmt.Printf("Epoch %d %f%%\n", epoch+1, float64(counter)/float64(total)*100)
			}
			if counter % gc_period == 0 {
				runtime.GC()
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
