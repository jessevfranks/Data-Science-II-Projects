package project1

//import scalation.mathstat._
import scalation.theory._

object HousePriceRegression {
  val filePath = "project1/house_price_regression_dataset.csv"
  val dataName = "HousePriceRegression"
  val data: Dataset = Dataset(dataName, filePath, 8, Array(0, 1, 2, 3, 4, 5, 6), 7)

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
