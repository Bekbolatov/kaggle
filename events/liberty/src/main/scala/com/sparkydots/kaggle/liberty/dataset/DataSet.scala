package com.sparkydots.kaggle.liberty.dataset

import com.sparkydots.spark.dataframe.{ReadWrite, RichDataFrame}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col


object DataSet  extends Serializable {

  val localPath: String = "/Users/rbekbolatov/data/kaggle/liberty/scratch"
  val location: String = "data/kaggle/liberty"
  val processedDir: String = "apple"

  def getDatasets(rw: ReadWrite) = (rw.load("typedKnown"), rw.load("typedLb"))

  def getDatasetsFromOrig(rw: ReadWrite) = {

    val rawKnown = rw.loadCSV("train")
    val rawLb = rw.loadCSV("test")

    val typedKnown = reduceTypes(rawKnown)
    val typedLb = reduceTypes(rawLb)

    rw.save(typedKnown, "typedKnown")
    rw.save(typedLb, "typedLb")

    (typedKnown, typedLb)
  }

  def reduceTypes(df: DataFrame): DataFrame = {
    if (df.columns.contains(Columns.label)) {
      RichDataFrame.castColumns(df, Columns.intValues, "int")
    } else {
      RichDataFrame.castColumns(df, Columns.intValues_nolabel, "int").
        withColumn(Columns.label, col(Columns.id)).
        select(RichDataFrame.cols(Columns.all): _*)
    }
  }

}
