package project2

import scalation.theory.*

object AutoMPG {
  // Auto MPG: 398 rows x 7 features
  // Download: https://archive.ics.uci.edu/dataset/9/auto+mpg
  // Place cleaned CSV at: data/project2/auto_mpg_cleaned.csv
  // Columns: cylinders, displacement, horsepower, weight, acceleration, model_year, origin -> mpg
  val filePath = "project2/auto_mpg_cleaned.csv"
  val dataName = "AutoMPG"
  val data: Dataset = Dataset(dataName, filePath, 7, Array(0, 1, 2, 3, 4, 5), 6)

  @main def runAutoMPG(): Unit = {
    val utils = new Project2Utils(data.x, data.y, data.fname)

    utils.runNeuralNet2L()
    utils.runNeuralNet3L()

    // Bonus: Feature selection
    // utils.runForwardSelect()
    // utils.runBackwardsElimination()
    // utils.runStepwiseSelect()
  }
}