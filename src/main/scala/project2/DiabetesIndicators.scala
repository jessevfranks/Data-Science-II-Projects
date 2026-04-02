package project2

import scalation.theory.*

object DiabetesIndicators {

  val filePath = "project2/diabetes.csv"
  val dataName = "DiabetesIndicators"
  val data: Dataset = Dataset(dataName, filePath, 10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8), 9)

  @main def runDiabetesIndicator(): Unit = {
    val utils = new Project2Utils(data.x, data.y, data.fname)

    utils.runNeuralNet2L()
    utils.runNeuralNet3L()

    // Bonus: Feature selection
     utils.runForwardSelect() // Order: 0, 8, 2, 1, 7, 3, 6, 5, 4
     utils.runBackwardsElimination() // Order: 0, 8, 1, 6, 3, 7, 5, 4
     utils.runStepwiseSelect() // Order: 0, 8, 2, 1, 7, 3, 6, 5 (left 4 out)
  }
}


