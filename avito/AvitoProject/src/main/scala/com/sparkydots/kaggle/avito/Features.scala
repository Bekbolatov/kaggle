package com.sparkydots.kaggle.avito

import org.apache.spark.sql.SQLContext

import scala.util.hashing.MurmurHash3
import scala.math._

object Features {

  def getUserHistory(sqlContext: SQLContext) = {

    val userInfo = sqlContext.sql("select id, uaOsId, uaDeviceId, uaFamilyId from user").rdd
    val visits = sqlContext.sql("select userId, adId, eventTime from visit").rdd
    val phoneRequests = sqlContext.sql("select userId, adId, eventTime from phoneRequest").rdd
    val searches = sqlContext.sql("select userId, eventTime, userLogged, searchQuery, locationId, categoryId, params from searchInfo").rdd
    val impressions = sqlContext.sql("select i.userId, i.eventTime, s.histctr, s.isClick from searchStream s join searchInfo i on (i.id = s.searchId) where s.type = 3").rdd


    sqlContext.sql("select count(distinct searchId, adId) from searchStream").show()

  }

  def hash(feature: String, numBuckets: Int) : Int = {
    return (abs(MurmurHash3.stringHash(feature)) % numBuckets)
  }



  def calcError(sqlContext: SQLContext) = {


    sqlContext.sql("select count(1), avg(errf(s.histctr, s.isClick)) from searchStream s join searchInfo i on (i.id = s.searchId) where s.type = 3 and i.eventTime > 1900800").show()

    sqlContext.sql(
      """
        |select i.user, avg(errf(s.histctr, s.isClick)
        |from searchStream s
        |join searchInfo i on (i.id = s.searchId)
        |where s.type = 3 and i.eventTime > 1900800
        |group by
      """.stripMargin).show()

  }

}
