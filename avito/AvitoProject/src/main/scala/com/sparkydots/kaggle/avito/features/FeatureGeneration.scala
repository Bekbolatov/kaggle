package com.sparkydots.kaggle.avito.features

import com.sparkydots.kaggle.avito.features.FeatureEncoding._
import com.sparkydots.kaggle.avito.functions.Functions._
import com.sparkydots.kaggle.avito.load.LoadSave
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import scala.util.Try


class FeatureGeneration(sqlContext: SQLContext, wordsDictFile: String = "onlyWords20000") extends Serializable {
  /**
   *
  "isClick",
    "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
    "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
    "position", "histctr",
    "category", "params", "price", "title", "adImpCount", "adClickCount"
    "searchLocLevel", "searchLocPar", "searchCatLevel", "searchCatPar", "adCatLevel", "adCatPar"
   */
  def featurize(data: DataFrame, sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._

    val wordsDict = sqlContext.sparkContext.broadcast(LoadSave.loadDF(sqlContext, wordsDictFile).map({
      case Row(word: String, wordId: Int) => word -> wordId
    }).collect.toMap)

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

      val searchLocLevel = Try(r.getInt(21)).getOrElse(-1)
      val searchLocPar = Try(r.getInt(22)).getOrElse(-1)
      val searchCatLevel = Try(r.getInt(23)).getOrElse(-1)
      val searchCatPar = Try(r.getInt(24)).getOrElse(-1)
      val adCatLevel = Try(r.getInt(25)).getOrElse(-1)
      val adCatPar = Try(r.getInt(26)).getOrElse(-1)

      val cleanQueryLoc = trueLoc(searchLoc)
      val cleanQueryCat = trueCat(searchCat)
      val cleanAdCat = trueCat(category)

      val ctr = if (impCount > 50) clickCount * 1.0 / impCount else histctr
      val adCtr = if (adImpCount > 10000) adClickCount * 1.0 / adImpCount else 0.007450876279364931

      val titleWordIds = splitString(title).flatMap(p => wordsDict.value.get(stemString(p)))
      val queryWordsIds = splitString(searchQuery).flatMap(p => wordsDict.value.get(stemString(p)))

      val smallFeaturesIndices =
        booleanFeature(loggedIn > 0) ++
        booleanFeature(phoneCount > 1) ++
        booleanFeature(searchParams.isEmpty) ++
        booleanFeature(params.isEmpty) ++
        booleanFeature(length(searchQuery) < 1) ++
        booleanFeature(visitCount > 10) ++
        booleanFeature(impCount > 1000) ++
        booleanFeature(position < 3) ++
        booleanFeature(position > 5) ++
        booleanFeature(price > 100000.0) ++
        booleanFeature(price > 100 && price < 10000.0) ++
        booleanFeature(price <= 0.0) ++
        booleanFeature(searchCatPar ==  adCatPar) ++
        intFeature(hourOfDay(searchTime), 24) ++
        intFeature(dayOfWeek(searchTime), 7) ++
        intFeature(dayOfWeek(searchTime)*24 + hourOfDay(searchTime), 24*7) ++
        intFeature(trueCat(searchCat), trueCatSize) ++
        intFeature(trueCat(category), trueCatSize) ++
        intFeature(searchLocLevel - 1, 3) ++
        intFeature(searchLocPar + 1, 86) ++
        intFeature(searchCatLevel - 1, 3) ++
        intFeature(searchCatPar - 2, 11) ++
        intFeature(adCatLevel - 1, 3) ++
        intFeature(adCatPar + 1, 13)

      val (numSmallFeatures, smallFeatures) = smallFeaturesIndices.foldLeft(0, Seq[(Int, Double)]()) {
        case ((offset, cumFeats), (blocksize, feats)) =>
          (offset + blocksize, cumFeats ++ feats.map(f => (f + offset, 1.0)))
      }

      val whichFeaturesIndices = smallFeaturesIndices ++
        intFeature(trueLoc(searchLoc), trueLocSize) ++
        indicatorFeatures(titleWordIds, wordsDict.value.size) ++
        indicatorFeatures(queryWordsIds, wordsDict.value.size) ++
        indicatorFeatures(params.flatMap(p => paramsDict.value.get(p)), paramsDict.value.size) ++
        indicatorFeatures(searchParams.flatMap(p => paramsDict.value.get(p)), paramsDict.value.size)

      val (categoricalOffset, categoricalFeatures) = whichFeaturesIndices.foldLeft(0, Seq[(Int, Double)]()) {
        case ((offset, cumFeats), (blocksize, feats)) =>
          (offset + blocksize, cumFeats ++ feats.map(f => (f + offset, 1.0)))
      }

      val continuousFeatures =
        Seq((categoricalOffset + 1, searchParams.toSet.intersect(params.toSet).size.toDouble)) ++
        Seq((categoricalOffset + 2, length(searchQuery).toDouble)) ++
        (if (os < 0) Seq((categoricalOffset + 3, 1.0)) else Seq[(Int, Double)]()) ++
        Seq((categoricalOffset + 4, ctr)) ++
        Seq((categoricalOffset + 5, adCtr)) ++
        Seq((categoricalOffset + 6, histctr)) ++
        Seq((categoricalOffset + 7, titleWordIds.toSet.intersect(queryWordsIds.toSet).size.toDouble))

      val features = categoricalFeatures ++ continuousFeatures

      LabeledPoint(isClick, Vectors.sparse(categoricalOffset + 8, dedupeFeatures(features)))
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
