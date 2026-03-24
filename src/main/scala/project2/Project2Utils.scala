package project2

import scalation.mathstat.*
import scalation.modeling.neuralnet.{NeuralNet_2L, NeuralNet_3L}
import scalation.modeling.Regression
import scalation.modeling.neuralnet.Optimizer
import scalation.modeling.qk

class Project2Utils(x: MatrixD, y: VectorD, fname: Array[String]) {

  // Convert VectorD -> MatrixD for NeuralNet_3L 
  private val yMat: MatrixD = MatrixD.fromVector(y) // y requires matrixD here, and MatrixD.fromVector(y) is the correct conversion method.
  // Why convert types? = Regression (Project 1) is a single-output model. It only ever predicts one column, so a VectorD is sufficient and more natural.
  // The core math for NNets operates on matrices (weights are MatrixD, predictions are MatrixD, deltas are MatrixD)
  //     So the y parameter must also be MatrixD to keep everything dimensionally consistent during forward/backward propagation.
  // The .perceptron factory method is the bridge. When u only have one output (which is the common regression case), 
  //    ScalaTion provides .perceptron specifically to accept a VectorD and handle the conversion to MatrixD internally.

  // 1. 2L Neural Network must use .perceptron factory since y is a single VectorD
  def runNeuralNet2L(): Unit = {
    val mod = NeuralNet_2L.perceptron(x, y, fname)

    println("NeuralNet_2L: In-Sample")
    mod.trainNtest()()
    println(mod.summary2())

    println("NeuralNet_2L: TT Split 80/20")
    mod.validate()() // NEEDS 2 be called with extra ()
  }

  // 2. 3L Neural Network must use .rescale factory with MatrixD response
  def runNeuralNet3L(): Unit = {
    val mod = NeuralNet_3L.rescale(x, yMat, fname)

    println("NeuralNet_3L: In-Sample")
    mod.trainNtest()()
    println(mod.summary2())

    println("NeuralNet_3L: TT Split 80/20")
    mod.validate()()
  }

  // Hyper-parameter tuning helpers
  def setEta(eta: Double): Unit = Optimizer.hp("eta") = eta
  def setBatchSize(bSize: Int): Unit = Optimizer.hp("bSize") = bSize.toDouble

  // (extra not sure if we need)Feature Selection (uses Regression since NN forwardSelAll needs MatrixD y)
  def runForwardSelect(): Unit = {
    val reg = new Regression(x, y, fname)
    val (cols, _) = reg.forwardSelAll() // ignore rSq val for all three. must pass _ to tell scala to ignore
    println(s"Forward Select column order: ${cols.mkString(", ")}")
  }

  def runBackwardsElimination(): Unit = {
    val reg = new Regression(x, y, fname)
    val (cols, _) = reg.backwardElimAll(first = 0)
    println(s"Backward Elim column order: ${cols.mkString(", ")}")
  }

  def runStepwiseSelect(): Unit = {
    val reg = new Regression(x, y, fname)
    val (cols, _) = reg.stepwiseSelAll()
    println(s"Stepwise Select column order: ${cols.mkString(", ")}")
  }
}