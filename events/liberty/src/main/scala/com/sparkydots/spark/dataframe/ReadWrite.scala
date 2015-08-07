package com.sparkydots.spark.dataframe


import java.io.FileWriter

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

class ReadWrite(@transient sqlContext: SQLContext, location: String, processedDir: String, localPath: String) extends Serializable with Logging {
  /**
   * @param filename only the name portion (no .extension)
   * @param header  "true", "false"
   * @param delimiter  ",", "\t"
   * @param extension "csv", "tsv"
   * @return
   */
  def loadCSV(filename: String, header: String = "true", delimiter: String = ",", extension: String = "csv") = {
    val path = s"$location/${filename}.$extension"
    logInfo(s"Reading csv file from: ")
    sqlContext.read.
      format("com.databricks.spark.csv").
      options(Map("header" -> header, "delimiter" -> delimiter)).
      load(path)
  }

  def load(filename: String) = sqlContext.read.load(s"$location/$processedDir/$filename.parquet")
  def save(df: DataFrame, filename: String) = df.write.save(s"$location/$processedDir/$filename.parquet")

  def writeFFM(filename: String, data: RDD[(Double, Array[(Int, Int, Double)])]) = {
    val sub = new FileWriter(s"$localPath/$filename", false)
    val points = data.map { case (label, features) =>
      label.toInt.toString + " " + features.map { case (field, feature, value) => f"$field:$feature:$value%1.1f" }
    }.collect()
    points.foreach(p => sub.write(p + "\n"))
    sub.close()
  }

  def writeToFile(line: String, filename: String, append: Boolean = true) = {
    val sub = new FileWriter(s"$localPath/$filename", append)
    sub.write(line)
    sub.close()
  }

  /**
   *
   * @param header  "Id,Hazard"
   * @param data     "id": Int, "pred": Double
   * @param filename   "mysub1.csv"
   * @param append
   */
  def writeLibertySubmissionToFile(header: String, data: DataFrame, filename: String, append: Boolean = true) = {
    val localData = data.orderBy("id").map(r => (r.getLong(0).toInt, r.getDouble(1))).collect()
    val sub = new FileWriter(s"$localPath/$filename", append)
    sub.write(header + "\n")
    localData.foreach { case (label, pred) =>
      sub.write(f"$label,$pred%1.10f\n")
    }
    sub.close()
  }

}

