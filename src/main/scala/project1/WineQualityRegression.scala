package project1

import scalation.theory._

object WineQualityRegression {
  val filePath = "data/project1/wine-quality-red.csv"
  val dataName = "WineQualityRegression"
  val data: Dataset = Dataset(dataName, filePath, 12, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 11)

  @main def run(): Unit = {
    val utils = new Project1Utils(data.x, data.y, data.fname)
//    utils.runLinearRegressions()
//    utils.runRidgeRegressions()
//    utils.runLassoRegressions()
//    utils.runTransformedRegressions()
//    utils.runSymRidgeRegression()
//    utils.runForwardSelect()
//    utils.runBackwardsElimination()
    utils.runStepwiseSelect()
  }
}

