package com.sparkydots.kaggle.avito.functions

import com.sparkydots.kaggle.avito.functions.UdfFunctions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object DFFunctions extends Serializable {

  /**
   * preds = DataFrame (pred: prob CTR (Double), isClick/label: actual isClick (Double))
   * out = DataFrame error
   */
  val df_calcError = (preds: DataFrame) => preds.select(udf_error(preds(preds.columns(0)), preds(preds.columns(1))).as("e")).agg(avg("e")).collect()(0).getDouble(0)

}
