package hector

import (
    "math"
)

type CovFunc func(Vector, Vector)float64

func CovMatrix(X []*Sample, cov_func CovFunc) (*Matrix) {
    l := int64(len(X))
    ret := NewMatrix()
    for i := int64(0); i < l; i++ {
        for j := i; j < l; j++ {
            c := cov_func(X[i], X[j])
            ret.SetValue(i, j, c)
            ret.SetValue(j, i, c)
        }
    }
    return ret
}

func CovVector(X []*Sample, y *Sample, cov_func CovFunc) (*Vector) {
    l := int64(len(X))
    ret := NewVector()
    for i := int64(0); i < l; i++ {
        ret.SetValue(i, cov_func(X[i], y))
    }
    return ret
}

/* 
 Squared error covariance function
 ARD = auto relevance detection, and here indicates there is a scaling/radius factor per dimension
 */
type CovSEARD struct {
    Radiuses Vector // dim -> radius
    Amp float64
}

func (cov_func *CovSEARD) Init(radiuses Vector, amp float64) {
    cov_func.Radiuses = radiuses
    cov_func.Amp = amp
}

func (cov_func *CovSEARD) Cov(x1 Vector, x2 Vector) float64 {
    ret := 0.0
    tmp := 0.0
    for key, r := range cov_func.Radiuses.data {
        v1, _ := x1.GetValue(key)
        v2, _ := x2.GetValue(key)
        tmp = (v1-v2)/r
        ret += tmp * tmp
    }
    ret = cov_func.Amp * math.Exp(-ret)
    return ret
}
