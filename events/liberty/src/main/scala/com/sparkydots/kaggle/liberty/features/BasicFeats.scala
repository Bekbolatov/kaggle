package com.sparkydots.kaggle.liberty.features

import com.sparkydots.kaggle.liberty.dataset.Columns
import com.sparkydots.spark.dataframe.RichDataFrame
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class BasicFeats(known: DataFrame, lb:DataFrame) {

  //generate maps for categorical feats
  // potentially there is some scale

  // for now only numericals

  /**
   * input dataset contains 'Id' (=id), 'Hazard' (=label), 'T1_V1', 'T1_V2', ... (=predictors)
   * @param dataset
   * @return
   */
  def transform(dataset: DataFrame, num: Int): DataFrame = {
    import dataset.sqlContext.implicits._

    val onlyNumericals = dataset.select(RichDataFrame.cols(Columns.label +: Columns.intPredictors): _*)

    onlyNumericals.map { row =>
      val values = (1 until num).map(i => row.getInt(i).toDouble).toArray
      LabeledPoint(row.getInt(0).toDouble, Vectors.dense(values))
    }.toDF()

  }



}
