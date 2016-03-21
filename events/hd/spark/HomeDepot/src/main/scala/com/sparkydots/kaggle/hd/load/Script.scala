package com.sparkydots.kaggle.hd.load

import org.apache.spark.sql.SQLContext

object Script extends Serializable {

  def script(sqlContext: SQLContext) = {
    implicit val sqlimp = sqlContext
    import com.sparkydots.kaggle.hd.load._

    val bq = Loader.loadQueries()
    val aa = Loader.cleanRawText(bq)
  }

}
