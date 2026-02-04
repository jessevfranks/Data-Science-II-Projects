package project1

import scalation.sq
import scalation.mathstat.*
import scalation.modeling.{qk, LassoRegression, Regression, RidgeRegression, SymRidgeRegression, TranRegression}
import scala.math.sqrt

class Project1Utils(x: MatrixD, y: VectorD, fname: Array[String]) {
  
  // 1
  def runLinearRegressions(): Unit = {
    val linearRegression = new Regression(x, y, fname)

    // run in-sample split (default)
    println("Linear Regression: In-Sample")
    linearRegression.trainNtest()()
    println(linearRegression.summary())

    // run a train-test-split of 80-20 (default)
    println("Linear Regression: TT Split of 80/20")
    linearRegression.validate()
    println(linearRegression.summary())

    // run 5-fold cross validation
    println("Linear Regression: 5-fold cross validation")
    linearRegression.crossValidate()
    println(linearRegression.summary())
  }

  // 2a
  def runRidgeRegressions(): Unit = {
    val ridgeRegression = new RidgeRegression(x, y, fname)

    // run in-sample split (default)
    println("Ridge Regression: In-Sample")
    ridgeRegression.trainNtest()()
    println(ridgeRegression.summary())

    // run a train-test-split of 80-20 (default)
    println("Ridge Regression: TT Split of 80/20")
    ridgeRegression.validate()
    println(ridgeRegression.summary())
  }

  // 2b
  def runLassoRegressions(): Unit = {
    val lassoRegression = new LassoRegression(x, y, fname)

    // run in-sample split (default)
    println("Lasso Regression: In-Sample")
    lassoRegression.trainNtest()()
    println(lassoRegression.summary())

    // run a train-test-split of 80-20 (default)
    println("Lasso Regression: TT Split of 80/20")
    lassoRegression.validate()
    println(lassoRegression.summary())
  }

  // 3
  def runTransformedRegressions(): Unit = {
    
    println("Transformed Regression: Sqrt")
    val sqrtReg = new TranRegression(x, y, fname, tran = sqrt, itran = sq)
    sqrtReg.trainNtest()()
    println(sqrtReg.summary())

    // defaults to log1p
    println("Transformed Regression: Log1p")
    val logReg = new TranRegression(x, y, fname)
    logReg.trainNtest()()
    println(logReg.summary())

    // defaults to Box-Cox when apply() is used
    println("Transformed Regression: Box-Cox")
    val boxCoxReg = TranRegression.apply(x, y, fname)
    boxCoxReg.trainNtest()()
    println(boxCoxReg.summary())

    
    println("Transformed Regression: Yeo-Johnson")
    val yjReg = TranRegression.app_yj(x, y, fname)
    yjReg.trainNtest()()
    println(yjReg.summary())
  }
  
  // 4
  def runSymRidgeRegression(): Unit = {
    val symRidgeRegression = SymRidgeRegression.quadratic(x, y, fname)

    // run in-sample split (default)
    println("SymRidge Regression: In-Sample")
    symRidgeRegression.trainNtest()()
    println(symRidgeRegression.summary())

    // run a train-test-split of 80-20 (default)
    println("SymRidge Regression: TT Split of 80/20")
    symRidgeRegression.validate()
    println(symRidgeRegression.summary())
  }

  //--- Feature Selection methods ---
  // Note - these may need to be changed to allow us to analyze
  //        differences/see how models would evolve using the techniques
  def runForwardSelect(): Unit = {
    val reg = new Regression(x,y,fname)
    val (cols, rSq) = reg.forwardSelAll()
    val order = getImportance(cols, rSq, reg)
    println(s"Columns ranked in order of importance from forward select: ${order.mkString("Array(", ", ", ")")}")
  }

  def runBackwardsElimination(): Unit = {
    val reg = new Regression(x,y,fname)
    val (cols, rSq) = reg.backwardElimAll(first=0)
    val order = getImportance(cols, rSq, reg)
    println(s"Columns ranked in order of importance from backward select: ${order.mkString("Array(", ", ", ")")}")
  }

  def runStepwiseSelect(): Unit = {
    val reg = new Regression(x, y, fname)
    val (cols, rSq) = reg.stepwiseSelAll()
    val order = getImportance(cols, rSq, reg)
    println(s"Columns ranked in order of importance from stepwise select: ${order.mkString("Array(", ", ", ")")}")
  }

  private def getImportance(cols: Iterable[Int], rSq: MatrixD, regression: Regression): Array[Int] = {
    println(f"cols: ${cols.size}")
    println(s"rSq = $rSq")
    val importance = regression.importance(cols.toArray, rSq)

    // retain only the column numbers for readability
    importance.map(_._1)
  }



}
