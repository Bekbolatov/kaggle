package com.sparkydots.kaggle.hd.load

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{DoubleType, IntegerType}

// implicit val sqlimp = sqlContext
// import org.apache.spark.sql.functions._
// import sqlContext.implicits._

object Loader {

  case class OrigTrain(id: Int, product_uid: Int, product_title: String, search_term: String, relevance: Double)
  case class OrigTest(id: Int, product_uid: Int, product_title: String, search_term: String)
  case class OrigDescr(product_uid: Int, product_description: String)
  case class OrigAttr(product_uid: Int, name: String, value: String)

  case class Queries(uid: Int, title: String, desc: String, attrs: Seq[(String, String)], queries: Seq[(Int, String)])

  def load(filename: String, base: String = s"s3n://sparkydotsdata/kaggle/hd/orig")(implicit sqlContext: SQLContext) =
    sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(s"$base/$filename")
  //.select($"id".cast(IntegerType), $"product_uid".cast(IntegerType), $"product_title", $"search_term", $"relevance".cast(DoubleType)).as[OrigTrain]

  def loadHD(base: String = s"s3n://sparkydotsdata/kaggle/hd/orig")(implicit sqlContext: SQLContext) = {
    import sqlContext.implicits._

    val trainFile = load("train.csv").as[OrigTrain]
    val testFile = load("test.csv").as[OrigTest]
    val descrFile = load("product_descriptions.csv").as[OrigDescr]
    val attrFile = load("attributes.csv").as[OrigAttr]

    (trainFile, testFile, descrFile, attrFile)
  }



  def doit()(implicit sqlContext: SQLContext) = {

    val (train_df, test_df, descr_df, attr_df) = loadHD()

    // train
    // (id, product_uid, product_title, search_term, relevance)
    // test
    // (id, product_uid, product_title, search_term)
    // descr
    // (product_uid, product_description)
    // attr
    // (product_uid, name, value)

    // we want:
    // (uid, title, description, [attr1->attr1val, ...], [id->search_term, id->search_term,...])

    import org.apache.spark.sql.functions._
    import sqlContext.implicits._

    val train2_df = train_df.map { case OrigTrain(id, uid, title, query, score) => OrigTest(id, uid, title, query) }

    val base_queries = train2_df.union(test_df).rdd.map { case OrigTest(id, uid, title, query) =>
      (uid, (title, Seq((id, query))))
    }.reduceByKey { case ((t1: String, q1: Seq[(Int, String)]), (t2: String, q2: Seq[(Int, String)])) =>
      (t1, q1 ++ q2)
    }

    val descriptions = descr_df.rdd.map { case OrigDescr(product_uid, product_description) =>  (product_uid, product_description) }

    val base_queries2 = base_queries.leftOuterJoin(descriptions).map { case (id: Int, ((title: String, queries: Seq[(Int, String)]) , product_description: Option[String])) =>
      (id, (title, product_description.getOrElse(""), queries))
    }

    val attributes = attr_df.rdd.map { case OrigAttr(product_uid, name, value) =>
      (product_uid, Seq((name, value)))
    }.reduceByKey { case (a1: Seq[(String, String)], a2: Seq[(String, String)]) =>
      a1 ++ a2
    }

    val base_queries3 = base_queries2.leftOuterJoin(attributes).map { case (id: Int, ((title: String, product_description: String, queries: Seq[(Int, String)]) , attrs: Option[Seq[(String, String)]])) =>
      Queries(id, title, product_description, attrs.getOrElse(Seq()), queries)
    }

    base_queries3

  }
}



