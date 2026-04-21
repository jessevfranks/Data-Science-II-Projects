package finalProject

import project1.Project1Utils
import scalation.theory.*

object FeatureSelection {
  val filePath = "finalProject/UCI_Credit_Card_Cleaned.csv"

  val dataName = "UCI Credit Card"
  val data: Dataset = Dataset(
    dataName,
    filePath,
    24,
    Array(0, 1, 2, 3, 4, 5, 6, 7, 8 ,0, 10, 11,12,13,14,15,16,17,18,19,20,21,23),
    22
  )

  @main def runCCFeatureSelection(): Unit = {
    val utils = new Project1Utils(data.x, data.y, data.fname)

    //utils.runForwardSelect()
    //utils.runBackwardsElimination()
    utils.runStepwiseSelect()
  }
}
