package project1

import scalation.mathstat.*
import scalation.modeling.Regression

object Project1Utils {
  def run_linear_regressions(x: MatrixD, y: VectorD): Unit = {
    val linearRegression = new Regression(x, y)

    // run in-sample split (default)
    linearRegression.trainNtest()()
    println(linearRegression.summary())

    // run a train-test-split of 80-20 (default
    linearRegression.validate()
    println(linearRegression.summary())

    // run 5-fold cross validation
    linearRegression.crossValidate()
    println(linearRegression.summary())
  }

}
