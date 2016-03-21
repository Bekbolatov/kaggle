package com.sparkydots.kaggle.hd.load

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row

case class OrigTrain(id: Int, product_uid: Int, product_title: String, search_term: String, relevance: Double)

case class OrigTest(id: Int, product_uid: Int, product_title: String, search_term: String)

case class OrigDescr(product_uid: Int, product_description: String)

case class OrigAttr(product_uid: Int, name: String, value: String)

case class Product(uid: Int, title: String, desc: String, attrs: Seq[(String, String)], queries: Seq[(Int, String)])

object Loader {

  val BASE: String = s"s3n://sparkydotsdata/kaggle/hd/orig"

  def load(filename: String, base: String = s"s3n://sparkydotsdata/kaggle/hd/orig")(implicit sqlContext: SQLContext) =
    sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(s"$base/$filename")

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

    val descriptions = descr_df.rdd.map { case OrigDescr(product_uid, product_description) => (product_uid, product_description) }

    val base_queries2 = base_queries.leftOuterJoin(descriptions).map { case (id: Int, ((title: String, queries: Seq[(Int, String)]), product_description: Option[String])) =>
      (id, (title, product_description.getOrElse(""), queries))
    }

    val attributes = attr_df.rdd.map { case OrigAttr(product_uid, name, value) =>
      (product_uid, Seq((name, value)))
    }.reduceByKey { case (a1: Seq[(String, String)], a2: Seq[(String, String)]) =>
      a1 ++ a2
    }

    val base_queries3 = base_queries2.leftOuterJoin(attributes).map { case (id: Int, ((title: String, product_description: String, queries: Seq[(Int, String)]), attrs: Option[Seq[(String, String)]])) =>
      Product(id, title, product_description, attrs.getOrElse(Seq()), queries)
    }

    base_queries3

  }


  def saveQueries(bq: RDD[Product], filename: String = "reorg.parquet")(implicit sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val bqdf = bq.map(q => Product.unapply(q).get).toDF("uid", "title", "descr", "attrs", "qs")
    bqdf.write.save(s"$BASE/$filename")
  }

  def loadQueries(filename: String = "reorg.parquet")(implicit sqlContext: SQLContext) = {
    val bqdf = sqlContext.read.load(s"$BASE/$filename")
      .rdd
      .map {
      case r: Row =>
        Product(
          r.getInt(0),
          r.getString(1),
          r.getString(2),
          r.getAs[Seq[Row]](3).map { case Row(k: String, v: String) => (k, v) },
          r.getAs[Seq[Row]](4).map { case Row(k: Int, v: String) => (k, v) }
        )
    }
    bqdf
  }

  // result in "rawclean.parquet"
  def cleanRawText(bq: RDD[Product])(implicit sqlContext: SQLContext): RDD[Product] = {
    // org.apache.spark.sql.DataFrame =
    // [uid: int, title: string, descr: string, attrs: array<struct<_1:string,_2:string>>, qs: array<struct<_1:int,_2:string>>]
    val bqc = bq.map { case Product(uid: Int, title: String, descr: String, attrs: Seq[(String, String)], qs: Seq[(Int, String)]) =>
      val attr_clean = attrs.map { case (k: String, v: String) => (CleanText.clean(k), CleanText.clean(v)) }
      val qs_clean = qs.map { case (id: Int, q: String) => (id, CleanText.clean(q)) }
      Product(uid, CleanText.clean(title), CleanText.clean(descr), attr_clean, qs_clean)
    }

    bqc
  }

}



