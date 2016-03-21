package com.sparkydots.kaggle.hd.load

import org.apache.spark.sql.SQLContext

object Script extends Serializable {

  // Create HDFS DIR
  // sudo su - hdfs
  // hdfs dfs -mkdir /user/ec2-user
  // hdfs dfs -chmod -R 777 /user/ec2-user

  // spark-shell --packages com.databricks:spark-csv_2.10:1.4.0 --jars homedepot_2.10-1.0.jar
  def script(sqlContext: SQLContext) = {

    // Logging
    import org.apache.log4j.{Level, Logger}
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    implicit val sqlimp = sqlContext
    import com.sparkydots.kaggle.hd.load._
    import com.sparkydots.kaggle.hd.features._

    val bq = Loader.loadQueries("rawclean.parquet")

    val features = bq.flatMap { case product =>
        Features.testFeatures(product)
    }
  }


  def script0(sqlContext: SQLContext) = {

    // Logging
    import org.apache.log4j.{Level, Logger}
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)



    // import org.apache.spark.sql.functions._
    // import sqlContext.implicits._

    implicit val sqlimp = sqlContext
    import com.sparkydots.kaggle.hd.load._

    val bq = Loader.loadQueries()
    val aa = Loader.cleanRawText(bq)

  }



}
