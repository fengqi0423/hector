package ann

import (
	"fmt"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"github.com/xlvector/hector/eval"
	"math"
	"math/rand"
	"strconv"
)

type NeuralNetworkParams struct {
	LearningRate         float64
	LearningRateDiscount float64
	Regularization       float64
	Hidden               int64
	Steps                int
	Verbose              int
	w1                   float64
	Dropout              int
	Dropout_rate1        float64 // Input layer dropout rate
	Dropout_rate2        float64 // Hidden layer dropout rate
}

type TwoLayerWeights struct {
	L1 *core.Matrix
	L2 *core.Matrix
}

/*
Please refer to this chapter to know algorithm details :
http://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
*/
type NeuralNetwork struct {
	Model    TwoLayerWeights
	MaxLabel int64
	Params   NeuralNetworkParams
	TestSet  *core.DataSet
}

func RandomInitVector(dim int64) *core.Vector {
	v := core.NewVector()
	var i int64
	for i = 0; i < dim; i++ {
		v.Data[i] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
	}
	return v
}

func (self *NeuralNetwork) SaveModel(path string) {

}

func (self *NeuralNetwork) LoadModel(path string) {

}

func (algo *NeuralNetwork) Init(params map[string]string) {
	algo.Params.LearningRate, _ = strconv.ParseFloat(params["learning-rate"], 64)
	algo.Params.LearningRateDiscount, _ = strconv.ParseFloat(params["learning-rate-discount"], 64)
	algo.Params.Regularization, _ = strconv.ParseFloat(params["regularization"], 64)
	algo.Params.w1, _ = strconv.ParseFloat(params["w1"], 64)

	steps, _ := strconv.ParseInt(params["steps"], 10, 32)
	hidden, _ := strconv.ParseInt(params["hidden"], 10, 64)
	verbose, _ := strconv.ParseInt(params["verbose"], 10, 32)
	dropout, _ := strconv.ParseInt(params["dropout"], 10, 32)

	algo.Params.Steps = int(steps)
	algo.Params.Hidden = int64(hidden)
	algo.Params.Verbose = int(verbose)
	algo.Params.Dropout = int(dropout)

	global, _ := strconv.ParseInt(params["global"], 10, 64)
	test_dataset := core.NewDataSet()

	if algo.Params.Verbose == 1 {
		err := test_dataset.Load(params["validation_path"], global)
		if err != nil {
			test_dataset = nil
		}
		algo.TestSet = test_dataset
	}
}

func (algo *NeuralNetwork) Train(dataset *core.DataSet) {
	algo.Model = TwoLayerWeights{}
	algo.Model.L1 = core.NewMatrix()
	algo.Model.L2 = core.NewMatrix()

	for i := int64(0); i < algo.Params.Hidden; i++ {
		algo.Model.L1.Data[i] = core.NewVector()
	}

	initalized := make(map[int64]int)
	max_label := 0
	for _, sample := range dataset.Samples {
		if max_label < sample.Label {
			max_label = sample.Label
		}
		for _, f := range sample.Features {
			_, ok := initalized[f.Id]
			if !ok {
				for i := int64(0); i < algo.Params.Hidden; i++ {
					algo.Model.L1.SetValue(i, f.Id, (rand.Float64()-0.5)/math.Sqrt(float64(algo.Params.Hidden)))
				}
				initalized[f.Id] = 1
			}
		}
	}
	algo.MaxLabel = int64(max_label)

	for i := int64(0); i <= algo.Params.Hidden; i++ {
		for j := int64(0); j <= algo.MaxLabel; j++ {
			algo.Model.L2.SetValue(i, j, (rand.NormFloat64() / math.Sqrt(float64(algo.MaxLabel)+1.0)))
		}
	}

	hidden_dropout := make(map[int64]int)
	for hi := int64(0); hi <= algo.Params.Hidden; hi++ {
		hidden_dropout[hi] = 0 // No dropout by default
	}

	for step := 0; step < algo.Params.Steps; step++ {
		if algo.Params.Verbose <= 0 {
			fmt.Printf(".")
		}
		total := len(dataset.Samples)
		counter := 0
		for _, sample := range dataset.Samples {
			if algo.Params.Dropout == 1 {
				for hi := int64(0); hi <= algo.Params.Hidden; hi++ {
					hidden_dropout[hi] = rand.Intn(2)
				}
			}

			y := core.NewVector()
			z := core.NewVector()
			e := core.NewVector()
			class_weight := 1.0
			if sample.Label == 1 {
				class_weight = algo.Params.w1
			}
			delta_hidden := core.NewVector()

			for i := int64(0); i < algo.Params.Hidden; i++ {
				if hidden_dropout[i] == 0 {
					sum := float64(0)
					wi := algo.Model.L1.Data[i]
					for _, f := range sample.Features {
						sum += f.Value * wi.GetValue(f.Id)
					}
					y.Data[i] = util.Sigmoid(sum)
				} else {
					y.Data[i] = 0.0
				}
			}
			y.Data[algo.Params.Hidden] = 1.0
			for i := int64(0); i <= algo.MaxLabel; i++ {
				sum := float64(0)
				for j := int64(0); j <= algo.Params.Hidden; j++ {
					sum += y.GetValue(j) * algo.Model.L2.GetValue(j, i)
				}
				z.SetValue(i, sum)
			}
			z = z.SoftMaxNorm()
			e.SetValue(int64(sample.Label), 1.0)
			e.AddVector(z, -1.0)

			for i := int64(0); i <= algo.Params.Hidden; i++ {
				if hidden_dropout[i] == 0 {
					delta := float64(0)
					for j := int64(0); j <= algo.MaxLabel; j++ {
						wij := algo.Model.L2.GetValue(i, j)
						sig_ij := e.GetValue(j) * (1 - z.GetValue(j)) * z.GetValue(j)
						delta += sig_ij * wij
						wij += class_weight * algo.Params.LearningRate * (y.GetValue(i)*sig_ij - algo.Params.Regularization*wij)
						algo.Model.L2.SetValue(i, j, wij)
					}
					delta_hidden.SetValue(i, delta)
				} else {
					delta_hidden.SetValue(i, 0.0)
				}
			}

			done := make(chan int)
			for i := int64(0); i < algo.Params.Hidden; i++ {
				go algo.update_weights_to_one_hidden_neuron(i, sample, class_weight, y, delta_hidden, done)
			}
			for i:= int64(0); i < algo.Params.Hidden; i++ {
				<- done
			}

			counter++
			if algo.Params.Verbose > 0 && counter%2000 == 0 {
				fmt.Printf("Epoch %d %f%%\n", step+1, float64(counter)/float64(total)*100)
			}
		}

		if algo.Params.Verbose > 0 && algo.TestSet != nil {
			algo.Evaluate(algo.TestSet)
		}
		algo.Params.LearningRate *= algo.Params.LearningRateDiscount
	}
	if algo.Params.Dropout == 1 {
		algo.Model.L2.Scale(0.5)
	}
	fmt.Println()
}

func (algo *NeuralNetwork) update_weights_to_one_hidden_neuron(i int64, sample *core.Sample, class_weight float64, y *core.Vector, delta_hidden *core.Vector, done chan int){
	wi := algo.Model.L1.Data[i]
	for _, f := range sample.Features {
		wji := wi.GetValue(f.Id)
		wji += class_weight * algo.Params.LearningRate * (delta_hidden.GetValue(i)*f.Value*y.GetValue(i)*(1-y.GetValue(i)) - algo.Params.Regularization*wji)
		wi.SetValue(f.Id, wji)
	}
	done <- 1
}

func (algo *NeuralNetwork) PredictMultiClass(sample *core.Sample) *core.ArrayVector {
	y := core.NewVector()
	z := core.NewArrayVector()
	for i := int64(0); i < algo.Params.Hidden; i++ {
		sum := float64(0)
		for _, f := range sample.Features {
			sum += f.Value * algo.Model.L1.Data[i].GetValue(f.Id)
		}
		y.Data[i] = util.Sigmoid(sum)
	}
	y.Data[algo.Params.Hidden] = 1
	for i := 0; i <= int(algo.MaxLabel); i++ {
		sum := float64(0)
		for j := int64(0); j <= algo.Params.Hidden; j++ {
			sum += y.GetValue(j) * algo.Model.L2.GetValue(j, int64(i))
		}
		z.SetValue(i, sum)
	}
	z = z.SoftMaxNorm()
	return z
}

func (algo *NeuralNetwork) Predict(sample *core.Sample) float64 {
	z := algo.PredictMultiClass(sample)
	return z.GetValue(1)
}

func (algo *NeuralNetwork) Evaluate(dataset *core.DataSet) {
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
