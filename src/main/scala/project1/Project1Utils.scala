package project1

import scalation.mathstat.*
import scalation.modeling.{LassoRegression, Regression, RidgeRegression, SymRidgeRegression}

object Project1Utils {
  
  // 1
  def runLinearRegressions(x: MatrixD, y: VectorD): Unit = {
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

  // 2a
  def runRidgeRegressions(x: MatrixD, y: VectorD): Unit = {
    val ridgeRegression = new RidgeRegression(x, y)

    // run in-sample split (default)
    ridgeRegression.trainNtest()()
    println(ridgeRegression.summary())

    // run a train-test-split of 80-20 (default
    ridgeRegression.validate()
    println(ridgeRegression.summary())
  }

  // 2b
  def runLassoRegressions(x: MatrixD, y: VectorD): Unit = {
    val lassoRegression = new LassoRegression(x, y)

    // run in-sample split (default)
    lassoRegression.trainNtest()()
    println(lassoRegression.summary())

    // run a train-test-split of 80-20 (default
    lassoRegression.validate()
    println(lassoRegression.summary())
  }
  
  // 4
  def runSymRidgeRegression(x: MatrixD, y: VectorD): Unit = {
    val symRidgeRegression = SymRidgeRegression.quadratic(x, y)

    // run in-sample split (default)
    symRidgeRegression.trainNtest()()
    println(symRidgeRegression.summary())

    // run a train-test-split of 80-20 (default
    symRidgeRegression.validate()
    println(symRidgeRegression.summary())
  }
}
