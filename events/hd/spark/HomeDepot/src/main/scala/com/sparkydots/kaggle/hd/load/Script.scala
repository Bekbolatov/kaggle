package com.sparkydots.kaggle.hd.load

import org.apache.spark.sql.{Row, SQLContext}

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
    import com.sparkydots.kaggle.hd.spell._

    val bq = Loader.loadQueries("rawclean2.parquet")

//    val bq = Loader.loadQueries("reorg.parquet")
//    val bqq = Loader.cleanRawText(bq)
//    Loader.saveQueries(bqq, "rawclean2.parquet")

    import sqlContext.implicits._

    val features = bq.flatMap { case product =>
      PromiscuousSpelling(product).matches()
    }

    val features_df = features.cache().toDF("id", "title", "descr", "attr", "brand")
    features_df.write.save("s3n://sparkydotsdata/kaggle/hd/orig/matches.parquet")


    val fdf = features_df.select($"id", $"title._1", $"title._2", $"descr._1", $"descr._2", $"attr", $"brand._1", $"brand._2")

    val frdd = features_df.select($"id", $"title._1", $"title._2", $"descr._1", $"descr._2", $"attr", $"brand._1", $"brand._2").rdd
    frdd.coalesce(1, true).saveAsTextFile("s3n://sparkydotsdata/kaggle/hd/features/matched_strings.csv")

    val fs = sqlContext.read.load("s3n://sparkydotsdata/kaggle/hd/orig/matches.parquet")


      val aa = fs.map { case r:Row =>
        (r.getInt(0), r.getAs[Seq[Row]](1).map { case Row(k: Seq[Row], v: Seq[Row]) => (k.map { case Row(kk) => kk}, v.map { case Row(kk) => kk}) })
      }

//      [112899,[WrappedArray(purell),WrappedArray()],[WrappedArray(),WrappedArray()],WrappedArray(),[WrappedArray(),WrappedArray()]]

    //    val a = features.sample(false, 1.0/ 50000, 101)
//    a.cache.count()
//    val a_df = a.toDF("id", "title", "descr", "attr", "brand")
//    a_df.write.save("s3n://sparkydotsdata/kaggle/hd/orig/example_save.parquet")





//    val bq = Loader.loadQueries("rawclean.parquet")

//    val features = bq.flatMap { case product =>
//      PromiscuousSpelling(product).matches()
//        //Features.testFeatures(product)
//    }
//
//    features.cache().count()


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
