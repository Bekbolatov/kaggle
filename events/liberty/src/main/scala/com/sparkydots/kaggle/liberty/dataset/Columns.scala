package com.sparkydots.kaggle.liberty.dataset

object Columns extends Serializable {

  val id = "Id"
  val label = "Hazard"

  val all = Seq("Id", "Hazard",
    "T1_V1", "T1_V2", "T1_V3", "T1_V4", "T1_V5", "T1_V6", "T1_V7", "T1_V8", "T1_V9", "T1_V10", "T1_V11", "T1_V12", "T1_V13", "T1_V14", "T1_V15", "T1_V16", "T1_V17",
    "T2_V1", "T2_V2", "T2_V3", "T2_V4", "T2_V5", "T2_V6", "T2_V7", "T2_V8", "T2_V9", "T2_V10", "T2_V11", "T2_V12", "T2_V13", "T2_V14", "T2_V15")

  val all_nolabel = all.filter(_ != label)

  val predictors = Seq(
    "T1_V1", "T1_V2", "T1_V3", "T1_V4", "T1_V5", "T1_V6", "T1_V7", "T1_V8", "T1_V9", "T1_V10", "T1_V11", "T1_V12", "T1_V13", "T1_V14", "T1_V15", "T1_V16", "T1_V17",
    "T2_V1", "T2_V2", "T2_V3", "T2_V4", "T2_V5", "T2_V6", "T2_V7", "T2_V8", "T2_V9", "T2_V10", "T2_V11", "T2_V12", "T2_V13", "T2_V14", "T2_V15")

  // By possible smallest type that can hold the value and also can be used as a Spark DataFrame column type
  val intValues = Seq("Id", "Hazard",
    "T1_V1", "T1_V2", "T1_V3", "T1_V10", "T1_V13", "T1_V14",
    "T2_V1", "T2_V2", "T2_V4", "T2_V6", "T2_V7", "T2_V8", "T2_V9", "T2_V10", "T2_V14", "T2_V15")

  val letterValues = Seq(
    "T1_V4", "T1_V5", "T1_V6", "T1_V7", "T1_V8", "T1_V9", "T1_V11", "T1_V12", "T1_V15", "T1_V16", "T1_V17",
    "T2_V3", "T2_V5", "T2_V11", "T2_V12", "T2_V13")

  val intValues_nolabel = intValues.filter(_ != label)

  val letterValues_nolabel = letterValues.filter(_ != label)

  val intPredictors = intValues.filter(c => c != label && c != id)

  val letterPredictors = letterValues.filter(c => c != label && c != id)

}
