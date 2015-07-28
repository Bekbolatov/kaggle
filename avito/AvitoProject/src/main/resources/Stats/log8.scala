


import com.sparkydots.kaggle.avito.functions.Functions._
import com.sparkydots.kaggle.avito.load.LoadSave
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.Row


val queryAds = rawTrain.filter("isClick = 1").select("searchQuery","adId").cache()

val termAds = queryAds.flatMap(r => (splitStringWithCutoff(r.getString(0), 2).map(x => (stemString(x), r.getInt(1))))).cache

val wordsDict = sqlContext.sparkContext.broadcast(LoadSave.loadDF(sqlContext, "words50").map({
  case Row(word: String, wordId: Int) => word -> wordId
}).collect.toMap)


//termAds.map { case (word, ad) => (word + ad, 1) }.reduceByKey(_ + _).sortBy(- _._2).take(10).foreach(println)



val ratings = termAds.
  map { case (word, ad) => (word + ad, (1, word, ad)) }.
  reduceByKey( (x, y) => (x._1 + y._1, x._2, x._3)).
  flatMap { case (_, (cnt, word, ad)) =>
  wordsDict.value.get(word).map { wordId =>
    Rating(wordId, ad, cnt)
  }
}

termAdCounts.sortBy(- _._3).take(10).foreach(println)



(8056,6115044,354)
(7866,13558903,222)
(8051,13558903,145)
(8050,31151820,125)
(7858,31151820,125)
(7996,31565576,121)
(7844,19417025,108)
(7690,6115044,105)
(7992,6892762,104)
(7860,7453083,95)

scala> termAds.map { case (word, ad) => (word + ad, 1) }.reduceByKey(_ + _).sortBy(- _._2).take(10).foreach(println)
(велосип6115044,354)
(автокрес13558903,222)
(детск13558903,145)
(свадебн31151820,125)
(плат31151820,125)
(ноутб31565576,121)
(playstati19417025,108)
(велосипе6115044,105)
(телевиз6892762,104)
(macbo7453083,95)




val termAdCounts = termAds.map { case (word, ad) => (word + ad, 1) }.
  reduceByKey(_ + _).
  map( )



val rank = 10
val numIterations = 20
val model = ALS.train(ratings, rank, numIterations, 0.01)

// Evaluate the model on rating data
val usersProducts = ratings.map { case Rating(user, product, rate) =>
  (user, product)
}
val predictions =
  model.predict(usersProducts).map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }
val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
  ((user, product), rate)
}.join(predictions)
val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)


