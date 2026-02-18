package project1

//import scalation.mathstat._
import scalation.theory._

object HousePriceRegression {
  // previously used raw dataset, but now use clean after running feature selection algorithms on the raw dataset
  val filePath = "project1/house_price_regression_dataset_cleaned.csv"

  val dataName = "HousePriceRegression"
  val data: Dataset = Dataset(dataName, filePath, 7, Array(0, 1, 2, 3, 4, 5), 6)

  @main def runHousePriceRegressions(): Unit = {
    val utils = new Project1Utils(data.x, data.y, data.fname)
    //utils.runLinearRegressions()
    //utils.runRidgeRegressions()
    //utils.runLassoRegressions()
    //utils.runTransformedRegressions()
    utils.runSymRidgeRegression()
//    utils.runForwardSelect()
//    utils.runBackwardsElimination()
//    utils.runStepwiseSelect()
  }
}
