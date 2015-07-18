package com.sparkydots.kaggle.avito.functions

import com.sparkydots.kaggle.avito.functions.UdfFunctions._
import org.apache.spark.sql.DataFrame
import com.sparkydots.kaggle.avito.functions.Functions._
import org.apache.spark.sql.functions._

object DFFunctions extends Serializable {

  /**
   * preds = DataFrame (pred: prob CTR (Double), isClick: actual isClick (Int))
   * out = DataFrame error
   */
  val calcError = (preds: DataFrame) => preds.select(error(preds("pred"), preds("isclick")).as("e")).agg(avg("e"))

}
