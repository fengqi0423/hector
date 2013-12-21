package hector

import (
	"sort"
	"math"
)

type Pred struct {
	Prediction float64
}

type LabelPrediction struct {
	Pred
	Label int
}

func (lp *LabelPrediction) DoubleValue() float64 {
	return float64(lp.Label)
}

type RealPrediction struct { // Real valued
	Pred
	Value float64
}

func (rp *LabelPrediction) DoubleValue() float64 {
	return rp.Value
}

type By func(p1, p2 *LabelPrediction) bool

type labelPredictionSorter struct {
	predictions []*LabelPrediction
	by      By
}

func (s *labelPredictionSorter) Len() int {
	return len(s.predictions)
}

func (s *labelPredictionSorter) Swap(i, j int) {
	s.predictions[i], s.predictions[j] = s.predictions[j], s.predictions[i]
}

func (s *labelPredictionSorter) Less(i, j int) bool {
	return s.by(s.predictions[i], s.predictions[j])
}

func (by By) Sort(predictions []*LabelPrediction) {
	sorter := &labelPredictionSorter{
		predictions: predictions,
		by:      by,
	}
	sort.Sort(sorter)
}

func AUC(predictions0 []*LabelPrediction) float64 {
	predictions := []*LabelPrediction{}
	for _, pred := range predictions0{
		predictions = append(predictions, pred)
	}
	prediction := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}
	
	By(prediction).Sort(predictions)
	
	pn := 0.0
	nn := float64(len(predictions))
	ret := 0.0
	count := nn
	for i, lp := range predictions{
		if lp.Label > 0 {
			pn += 1.0
			nn -= 1.0
			ret += float64(count) - float64(i)
		}
	}
	ret2 := pn * (pn + 1) / 2.0;
	if pn * nn == 0.0{
		return 0.5
	}
	return (ret - ret2) / (pn * nn)
}

func RMSE(predictions []*Pred) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (pred.DoubleValue() - pred.Prediction) * (pred.DoubleValue() - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}

func ErrorRate(predictions []*Pred) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		if (pred.DoubleValue() - 0.5) * (pred.Prediction - 0.5) < 0 {
			ret += 1.0
		}
		n += 1.0
	}
	return ret / n
}
