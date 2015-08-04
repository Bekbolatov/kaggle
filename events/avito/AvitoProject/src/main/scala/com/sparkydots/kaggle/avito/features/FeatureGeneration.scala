package com.sparkydots.kaggle.avito.features

import com.sparkydots.kaggle.avito.features.FeatureEncoding._
import com.sparkydots.kaggle.avito.functions.Functions._
import com.sparkydots.kaggle.avito.load.LoadSave
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import scala.util.Try


class FeatureGeneration(sqlContext: SQLContext, wordsDictFile: String = "onlyWords20000", wordDictFileNei: Option[String]) extends Serializable {
  /**
   *
  "isClick",
    "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
    "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
    "position", "histctr",
    "category", "params", "price", "title", "adImpCount", "adClickCount",

   "searchId", "adId", "userId",  <<<>> NEW
   "neiPrice", "neiTitle", "neiParams", "neiCat", <<<>> NEW

    "searchLocLevel", "searchLocPar", "searchCatLevel", "searchCatPar", "adCatLevel", "adCatPar"
   */

/*
0	isClick
1	os
2	uafam
3	visitCount
4	phoneCount
5	impCount
6	clickCount
7	searchTime
8	searchQuery
9	searchLoc
10	searchCat
11	searchParams
12	loggedIn
13	position
14	histctr
15	category
16	params
17	price
18	title
19	adImpCount
20	adClickCount
21	searchId
22	adId
23	userId
24	neiPrice
25	neiTitle
26	neiParams
27	neiCat
28	searchLocLevel
29	searchLocPar
30	searchCatLevel
31	searchCatPar
32	adCatLevel
33	adCatPar
 */

    def featurize(data: DataFrame, sqlContext: SQLContext): DataFrame = {
      import sqlContext.implicits._

      val wordsDict = sqlContext.sparkContext.broadcast(LoadSave.loadDF(sqlContext, wordsDictFile).map({
        case Row(word: String, wordId: Int) => word -> wordId
      }).collect.toMap)

      val wordsDict2 = wordDictFileNei.map { loc =>
        sqlContext.sparkContext.broadcast(LoadSave.loadDF(sqlContext, wordsDictFile).map({
          case Row(word: String, wordId: Int) => word -> wordId
        }).collect.toMap)
      }

      val paramsDict = sqlContext.sparkContext.broadcast(LoadSave.loadDF(sqlContext, "params1000").map({
        case Row(param: Int, paramId: Int) => param -> paramId
      }).collect.toMap)

      val featurized = data.map { r =>
        val isClick = r.getInt(0).toDouble
        val os = Try(r.getInt(1)).getOrElse(-1)
        val uafam = Try(r.getInt(2)).getOrElse(-1)

        val visitCount = Try(r.getLong(3).toInt).getOrElse(0)
        val phoneCount = Try(r.getLong(4).toInt).getOrElse(0)
        val impCount = Try(r.getLong(5).toInt).getOrElse(0)
        val clickCount = Try(r.getLong(6).toInt).getOrElse(0)

        val searchTime = Try(r.getInt(7)).getOrElse(0)
        val searchQuery = Try(r.getString(8).toLowerCase).getOrElse("")
        val searchLoc = Try(r.getInt(9)).getOrElse(-1)
        val searchCat = Try(r.getInt(10)).getOrElse(-1)
        val searchParams = Try({
          val l = r.getSeq[Int](11)
          if (l == null) {
            Seq.empty
          } else {
            l
          }
        }).getOrElse(Seq.empty)

        val loggedIn = Try(r.getInt(12)).getOrElse(-1)
        val position = r.getInt(13)
        val histctr = r.getDouble(14)
        val category = Try(r.getInt(15)).getOrElse(-1)
        val params = Try({
          val l = r.getSeq[Int](16)
          if (l == null) {
            Seq.empty
          } else {
            l
          }
        }).getOrElse(Seq.empty)
        val price = Try(r.getDouble(17)).getOrElse(-1.0)
        val title = Try(r.getString(18).toLowerCase).getOrElse("")

        val adImpCount = Try(r.getLong(19).toInt).getOrElse(0)
        val adClickCount = Try(r.getLong(20).toInt).getOrElse(0)
        //searchId, adId, userId --skip

        // nei
        val neiPrice = Try(r.getDouble(24)).getOrElse(-1.0)
        val neiTitle = Try(r.getString(25).toLowerCase).getOrElse("")
        val neiCategory = Try(r.getInt(26)).getOrElse(-1)
        val neiParams = Try({
          val l = r.getSeq[Int](27)
          if (l == null) {
            Seq.empty
          } else {
            l
          }
        }).getOrElse(Seq.empty)

        val searchLocLevel = Try(r.getInt(28)).getOrElse(-1)
        val searchLocPar = Try(r.getInt(29)).getOrElse(-1)
        val searchCatLevel = Try(r.getInt(30)).getOrElse(-1)
        val searchCatPar = Try(r.getInt(31)).getOrElse(-1)
        val adCatLevel = Try(r.getInt(32)).getOrElse(-1)
        val adCatPar = Try(r.getInt(33)).getOrElse(-1)

        val queryTitlePos1 = Try(r.getDouble(34)).getOrElse(0.0) //48.11245
        val queryTitlePos = if (queryTitlePos1 != 0.0) queryTitlePos1 else 48.11245
        val queryTitleNeg1 = Try(r.getDouble(35)).getOrElse(0.0) //2210.1516
        val queryTitleNeg = if (queryTitleNeg1 != 0.0) queryTitleNeg1 else 2210.1516

        val cleanQueryLoc = trueLoc(searchLoc)
        val cleanQueryCat = trueCat(searchCat)
        val cleanAdCat = trueCat(category)

        val ctr = if (impCount > 50) clickCount * 1.0 / impCount else histctr
        val adCtr = if (adImpCount > 10000) adClickCount * 1.0 / adImpCount else 0.007450876279364931

        val titleWordIds = splitString(title).flatMap(p => wordsDict.value.get(stemString(p)))
        val queryWordsIds = splitString(searchQuery).flatMap(p => wordsDict.value.get(stemString(p)))
        val paramIds = params.flatMap(p => paramsDict.value.get(p))
        val searchParamIds = searchParams.flatMap(p => paramsDict.value.get(p))

        val neiTitleWordIds = splitString(neiTitle).flatMap(p => wordsDict.value.get(stemString(p)))

        val queryTitleMatch = titleWordIds.toSet.intersect(queryWordsIds.toSet).toSeq
        val queryNeiTitleMatch = neiTitleWordIds.toSet.intersect(queryWordsIds.toSet).toSeq

        val neiTitleWordIds2 = if (wordDictFileNei.nonEmpty)
          splitString(neiTitle).flatMap(p => wordsDict2.get.value.get(stemString(p)))
        else
          Seq()

        val totalCrossQueryTitle = titleWordIds.size * queryWordsIds.size

        // val asd = udf[Int, String, String] (  (x, y) => {val a = splitStringWithCutoff( x, 2 ).size ; val b = splitStringWithCutoff(y, 2).size; val c = a*b; if (c > 0) c else 1  } )
        //rawTrain.filter("queryTitlePos != 0.0").select( (col( "queryTitlePos")/ asd(col("searchQuery"), col("title"))).as("huya")   ).agg(avg("huya")).show
        // rawTrain.filter("queryTitleNeg != 0.0").select( (col( "queryTitleNeg")/ asd(col("searchQuery"), col("title"))).as("huya")   ).agg(avg("huya")).show
        //rawTrain.filter("isClick = 0").select(col("searchQuery"), col("title"),col( "queryTitlePos"), col("queryTitleNeg"), (col( "queryTitlePos")/ asd(col("searchQuery"), col("title"))).as("huya")   ).agg(avg("huya")).show
        //rawTrain.filter("isClick = 1").select(col("searchQuery"), col("title"),col( "queryTitlePos"), col("queryTitleNeg"), (col( "queryTitlePos")/ asd(col("searchQuery"), col("title"))).as("huya")   ).agg(avg("huya")).show
        //rawTrain.filter("isClick = 0").select(col("searchQuery"), col("title"),col( "queryTitlePos"), col("queryTitleNeg"), (col( "queryTitleNeg")/ asd(col("searchQuery"), col("title"))).as("huya")   ).agg(avg("huya")).show
        //rawTrain.filter("isClick = 1").select(col("searchQuery"), col("title"),col( "queryTitlePos"), col("queryTitleNeg"), (col( "queryTitleNeg")/ asd(col("searchQuery"), col("title"))).as("huya")   ).agg(avg("huya")).show
        val smallFeaturesIndices =
          booleanFeature(loggedIn > 0) ++
          booleanFeature(clickCount < 1) ++
          booleanFeature(phoneCount > 1) ++
          booleanFeature(searchParams.isEmpty) ++
          booleanFeature(params.isEmpty) ++
          booleanFeature(length(searchQuery) < 1) ++
          booleanFeature(visitCount > 10) ++
          booleanFeature(impCount > 1000) ++
          booleanFeature(position < 3) ++
          booleanFeature(position > 5) ++
          booleanFeature(price > 200000.0) ++
          booleanFeature(price > 100 && price < 10000.0) ++
          booleanFeature(price <= 0.0) ++
          booleanFeature(price < neiPrice && neiPrice > 0 && price > 0) ++
          booleanFeature(price*5 < neiPrice && price > 0) ++
          booleanFeature(neiPrice <= 0) ++
          booleanFeature(queryNeiTitleMatch.size >= queryTitleMatch.size && queryNeiTitleMatch.nonEmpty) ++
          booleanFeature(searchCatPar ==  adCatPar) ++
          intFeature(hourOfDay(searchTime), 24) ++
          intFeature(dayOfWeek(searchTime), 7) ++
          intFeature(dayOfWeek(searchTime)*24 + hourOfDay(searchTime), 24*7) ++
          intFeature(searchLocLevel - 1, 3) ++
          intFeature(searchCatLevel - 1, 3) ++
          intFeature(searchLocPar + 1, 86) ++
          intFeature(searchCatPar - 2, 11) ++
          intFeature( math.max(math.min(os - 1, 50), 0), 51) ++
          intFeature( math.max(math.min(uafam - 1, 88), 0), 89) ++
          intFeature(trueCat(searchCat), trueCatSize) ++
          intFeature(trueCat(category), trueCatSize) ++
          intFeature(adCatLevel - 1, 3) ++
          intFeature(adCatPar + 1, 13)

        val (numSmallFeatures, smallFeatures) = smallFeaturesIndices.foldLeft(0, Seq[(Int, Double)]()) {
          case ((offset, cumFeats), (blocksize, feats)) =>
            (offset + blocksize, cumFeats ++ feats.map(f => (f + offset, 1.0)))
        }

        println(s" numSmallFeatures = $numSmallFeatures")

        val whichFeaturesIndices = smallFeaturesIndices ++
          intFeature(trueLoc(searchLoc), trueLocSize) ++
          indicatorFeatures(titleWordIds, wordsDict.value.size) ++
          indicatorFeatures(queryWordsIds, wordsDict.value.size) ++
          indicatorFeatures(queryTitleMatch, wordsDict.value.size) ++
          (if (wordDictFileNei.nonEmpty) indicatorFeatures(neiTitleWordIds2, wordsDict2.get.value.size) else Seq()) ++
          indicatorFeatures(paramIds.toSet.intersect(searchParamIds.toSet).toSeq, paramsDict.value.size) ++
          indicatorFeatures(paramIds, paramsDict.value.size) ++
          indicatorFeatures(searchParamIds, paramsDict.value.size)

        val (categoricalOffset, categoricalFeatures) = whichFeaturesIndices.foldLeft(0, Seq[(Int, Double)]()) {
          case ((offset, cumFeats), (blocksize, feats)) =>
            (offset + blocksize, cumFeats ++ feats.map(f => (f + offset, 1.0)))
        }

        val continuousFeatures =
          Seq((categoricalOffset + 1, math.min(searchParams.toSet.intersect(params.toSet).size.toDouble / 3, 1.0))) ++
          Seq((categoricalOffset + 2, math.min(length(searchQuery).toDouble / 38, 1.0))) ++
          Seq((categoricalOffset + 3, ctr)) ++
          Seq((categoricalOffset + 4, adCtr)) ++
          Seq((categoricalOffset + 5, histctr)) ++
  //        Seq((categoricalOffset + 6, queryTitlePos/ (if (totalCrossQueryTitle > 0) 1.0*totalCrossQueryTitle else 1.0) )) ++
  //        Seq((categoricalOffset + 6, (7*queryTitlePos - queryTitleNeg)/ (if (totalCrossQueryTitle > 0) 1.0*totalCrossQueryTitle else 1.0) )) ++
          Seq((categoricalOffset + 6, if (queryTitlePos1 != 0) ((40*queryTitlePos - queryTitleNeg)/ (if (totalCrossQueryTitle > 0) 1.0*totalCrossQueryTitle else 1.0)) else 0)) ++
//          Seq((categoricalOffset + 6, if (queryTitlePos1 != 0) (queryTitlePos1 / (if (totalCrossQueryTitle > 0) 1.0*totalCrossQueryTitle else 1.0)) else 18.11245)) ++
          Seq((categoricalOffset + 7, math.min(queryTitleMatch.size.toDouble / 4, 1.0))) ++
          Seq((categoricalOffset + 8, math.min(queryNeiTitleMatch.size.toDouble / 4, 1.0)))

        val features = categoricalFeatures ++ continuousFeatures

        LabeledPoint(isClick, Vectors.sparse(categoricalOffset + 9, dedupeFeatures(features)))
      }.toDF()

      featurized
    }

    def otherInteractions(singleFeatures: Seq[(Int, Double)],
                          otherFeatures: Seq[(Int, Double)],
                          offset: Int): Seq[(Int, Double)] = {
      var i = -1
      var j = -1
      singleFeatures.flatMap { f1 =>
        j = j + 1
        otherFeatures.flatMap { f2 =>
          i = i + 1
          if ( f1._1 != f2._1 && f1._2 != 0.0 && f2._2 != 0.0) {
            Some((j * otherFeatures.size + i, f1._2 * f2._2))
          } else {
            None
          }
        }
      }
    }

}
