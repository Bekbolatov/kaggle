package com.sparkydots.spark.dataframe


import java.io.FileWriter

import org.apache.spark.Logging
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

  def writeToFile(line: String, filename: String, append: Boolean = true) = {
    val sub = new FileWriter(s"$localPath/$filename", append)
    sub.write(line)
    sub.close()
  }

}

