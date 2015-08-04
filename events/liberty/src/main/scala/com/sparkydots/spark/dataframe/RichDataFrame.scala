package com.sparkydots.spark.dataframe

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}


object RichDataFrame extends Serializable {

  def cols(columnNames: Seq[String]): Seq[Column] = columnNames.map(col)

  def castColumn(df: DataFrame, targetColumnName: String, newType: String): DataFrame =
    replaceColumn(df, targetColumnName, col(targetColumnName).cast(newType))

  def castColumns(df: DataFrame, targetColumnNames: Seq[String], newType: String): DataFrame =
    replaceColumns(df, targetColumnNames.map(targetColumnName => (targetColumnName, col(targetColumnName).cast(newType))))

  def replaceColumn(df: DataFrame, targetColumnName: String, newColumn: Column): DataFrame =
    replaceColumns(df, Seq((targetColumnName, newColumn)))

  def replaceColumns(df: DataFrame, targetColumnNamesAndNewColumns: Seq[(String, Column)]): DataFrame = {
    val targetColumnNames = targetColumnNamesAndNewColumns.toMap
    df.select(df.columns.map { columnName =>
      targetColumnNames.get(columnName).map { newColumn =>
        newColumn.as(columnName)
      }.getOrElse(col(columnName))
    }: _*)
  }

}
