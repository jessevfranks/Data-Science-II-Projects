package scalation
package modeling

import scalation.mathstat.{VectorD, MatrixD, Plot}
import scalation.modeling.ActivationFun._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplePerceptron2` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple neural network (Perceptron).
 *
 *      grad g = -x.ᵀ * (y - ŷ) * ƒ    where pred ŷ = f(x * b)
 *
 *  Computations done at the scalar level: X -> y.  R^2 = .886
 *  > runMain scalation.modeling.simplePerceptron2
 */
@main def simplePerceptron2 (): Unit =

    // 9 data points:         One    x1    x2    y1   y2
    val xy = MatrixD ((9, 5), 1.0,  0.1,  0.1,  0.5, 0.25,       // dataset
                              1.0,  0.1,  0.5,  0.3, 0.49,
                              1.0,  0.1,  1.0,  0.2, 0.64,

                              1.0,  0.5,  0.1,  0.8, 0.04,
                              1.0,  0.5,  0.5,  0.5, 0.25,
                              1.0,  0.5,  1.0,  0.3, 0.49,

                              1.0,  1.0,  0.1,  1.0, 0.0,
                              1.0,  1.0,  0.5,  0.8, 0.04,
                              1.0,  1.0,  1.0,  0.5, 0.25)

    val (x, y) = (xy(?, 0 until 3), xy(?, 3))                    // input matrix, output/response vector

    val sst = (y - y.mean).normSq                               // sum of squares total

    val b = VectorD (0.1, 0.2, 0.1)                             // initial weights/parameters (random in practice)

    val η = 2.5                                                 // learning rate (to be tuned)
    var u, ŷ, ε, ƒ, δ: Double = 0.0
    var g: VectorD = null
    val yp = new VectorD (y.dim)                                // save each prediction in yp

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")
        var sse = 0.0
        for i <- x.indices do
            val (xi, yi) = (x(i), y(i))                         // randomize i for Pure Stochastic Gradient Descent (PSGD)

            // forward prop: input -> output
            u = xi ∙ b                                          // pre-activation scalar via dot (∙) product
            ŷ = f_sigmoid.f (u)                                 // prediction scalar
            ε = yi - ŷ                                          // error scalar

            // backward prop: output -> input
            ƒ = ŷ * (1.0 - ŷ)                                   // derivative (f') for sigmoid
            δ = ε * ƒ                                           // delta correction scalar
            g = xi * -δ                                         // gradient vector

            // parameter update
            b -= g * η                                          // update parameter vector

            yp(i) = ŷ                                           // save i-th prediction
            sse  += ε * ε                                       // sum of squared errors
        end for
        val r2 = 1.0 - sse / sst                                // R^2

        println (s"""
        u   = $u
        ŷ   = $ŷ
        ε   = $ε
        ƒ   = $ƒ
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)

    end for
    new Plot (null, y, yp, "IGD for Perceptron y", lines = true)

end simplePerceptron2

