package hector

import (
    "strconv"
    )

/*
    Given matrix m and vector v, compute inv(m)*v.
    Based on Gibbs and MacKay 1997, and Mark N. Gibbs's PhD dissertation 
  
    Details:
    A - positive seminidefinite matrix
    u - a vector
    theta - positive number
    C = A + I*theta
    Returns inv(C)*u - So you need the diagonal noise term for covariance matrix in a sense. 
    However, this algorithm is numerically stable, the noise term can be very small and the inversion can still be calculated...
*/
func ApproximateInversion(A *Matrix, u *Vector, theta float64, dim int64) *Vector {
    max_itr := 500
    tol := 0.01

    C := NewMatrix()
    for key, val := range A.data {
        C.data[key] = val.Copy()
    }

    // Add theta to diagonal elements
    for i := int64(0); i < dim; i++ {
        _, ok := C.data[i]
        if !ok {
            C.data[i] = NewVector()
        }
        C.data[i].data[i] = C.data[i].data[i] + theta
    }

    var Q_l float64
    var Q_u float64
    var dQ float64
    u_norm := u.Dot(u)/2

    // Lower bound
    y_l := NewVector()
    g_l := u.Copy()
    h_l := u.Copy()
    lambda_l := float64(0)
    gamma_l := float64(0)
    var tmp_f1 float64
    var tmp_f2 float64
    var tmp_v1 *Vector
    tmp_f1 = g_l.Dot(g_l)
    tmp_v1 = C.MultiplyVector(h_l)

    // Upper bound
    y_u := NewVector()
    g_u := u.Copy()
    h_u := u.Copy()
    lambda_u := float64(0) 
    gamma_u := float64(0)
    var tmp_f3 float64
    var tmp_f4 float64
    var tmp_v3 *Vector
    var tmp_v4 *Vector
    tmp_v3 = g_u.MultiplyMatrix(A)
    tmp_v4 = C.MultiplyVector(h_u)
    tmp_f3 = tmp_v1.Dot(g_u)

    for i := 0; i < max_itr; i++ {
        // Lower bound
        lambda_l = tmp_f1 / h_l.Dot(tmp_v1)
        y_l.AddVector(h_l, lambda_l) //y_l next
        Q_l = y_l.Dot(u) - 0.5*(y_l.MultiplyMatrix(C)).Dot(y_l)

        // Upper bound
        lambda_u = tmp_f3/tmp_v3.Dot(tmp_v4)
        y_u.AddVector(h_u, lambda_u) //y_u next
        Q_u = (y_u.MultiplyMatrix(A)).Dot(u) - 0.5*((y_u.MultiplyMatrix(C)).MultiplyMatrix(A)).Dot(y_u)

        dQ = (u_norm-Q_u)/theta - Q_l
        if dQ < tol{
            break
        }

        // Lower bound var updates
        g_l.AddVector(tmp_v1, -lambda_l) //g_l next
        tmp_f2 = g_l.Dot(g_l)
        gamma_l = tmp_f2/tmp_f1
        for key, val := range h_l.data {
            h_l.SetValue(key, val * gamma_l)
        }
        h_l.AddVector(g_l, 1) //h_l next
        tmp_f1 = tmp_f2 //tmp_f1 next
        tmp_v1 = C.MultiplyVector(h_l) //tmp_v1 next

        // Upper bound var updates
        g_u.AddVector(tmp_v4, -lambda_u) //g_u next
        tmp_v3 = g_u.MultiplyMatrix(A) //tmp_v3 next
        tmp_f4 = tmp_v3.Dot(g_u)
        gamma_u = tmp_f4/tmp_f3
        for key, val := range h_u.data {
            h_u.SetValue(key, val * gamma_u)
        }
        h_u.AddVector(g_u, 1) //h_u next
        tmp_v4 = C.MultiplyVector(h_u) //tmp_v4 next
        tmp_f3 = tmp_f4 // tmp_f3 next
    }

    return y_l
}

type GaussianProcessParameters struct {
    Dim int64
}

type GaussianProcess struct {
    Params GaussianProcessParameters
    CovarianceFunc CovFunc
    CovMatrix = Matrix
}

func (self *NeuralNetwork) SaveModel(path string){

}

func (self *NeuralNetwork) LoadModel(path string){
    
}

func (algo *GaussianProcess) Init(params map[string]string) {
    /*
    dim, _ := strconv.ParseInt(params["dim"], 10, 64)

    algo.Params = GaussianProcessParameters{}
    algo.Params.Dim = dim // Pass in dim as a param.. and require feature space to be continous.

    radius := 0.2
    camp := 40.0
    cf := CovSEARD{}
    radiuses := NewVector()
    for i := int64(0); i < dim; i++ {
        radiuses.SetValue(i, radius)
    }
    cf.Init(radiuses, camp)

    algo.CovarianceFunc = cf
    */
}

func (algo *GaussianProcess) Train(dataset * DataSet) {
    /*
    algo.DataSet = dataset
    */
}

func (algo *GaussianProcess) Predict(sample *Sample) float64 {
    /*
    z := algo.PredictMultiClass(sample)
    return z.GetValue(1)
    */
    return 0.5
}

