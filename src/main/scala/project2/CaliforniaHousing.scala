package project2

import scalation.theory.*

object CaliforniaHousing {
  // California Housing: 20,640 rows x 10 features
  // Download: https://www.kaggle.com/datasets/camnugent/california-housing-prices
  // Place cleaned CSV at: data/project2/california_housing_cleaned.csv
  // Target column (last): median_house_value
  val filePath = "project2/california_housing_cleaned.csv"
  val dataName = "CaliforniaHousing"
  val data: Dataset = Dataset(dataName, filePath, 10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8), 9)

  @main def runCaliforniaHousing(): Unit = {
    val utils = new Project2Utils(data.x, data.y, data.fname)

    utils.runNeuralNet2L()
    utils.runNeuralNet3L()

    // Bonus: Feature selection
    // utils.runForwardSelect()
    // utils.runBackwardsElimination()
    // utils.runStepwiseSelect()
  }
}