package com.sparkydots.kaggle.avito

import org.apache.spark.SparkContext

/*
spark-shell --jars AvitoProject-assembly-1.0.jar
val sqlctx = com.sparkydots.kaggle.avito.ReadData.ingest(sc)
 */

object ReadData {

  def ingest(sc: SparkContext) = {
    import UdfFunctions._

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    sqlContext.udf.register("strLen", (s: String) => s.length())
    //import sqlContext.implicits._
    def loadDF(filename: String) = sqlContext.load("com.databricks.spark.csv",
      Map("header" -> "true", "delimiter" -> "\t", "path" -> s"s3n://sparkydotsdata/kaggle/avito/${filename}.tsv"))

    // Category
    val _dfCategory = loadDF("Category")
    val dfCategory = _dfCategory.
      withColumn("id", toInt(_dfCategory("CategoryID"))).
      withColumn("level", toInt(_dfCategory("Level"))).
      withColumn("parentCategoryId", toInt(_dfCategory("ParentCategoryID"))).
      select("id", "level", "parentCategoryId").
      cache()

    dfCategory.registerTempTable("category")

    // Location
    val _dfLocation = loadDF("Location")
    val dfLocation = _dfLocation.
      withColumn("id", toInt(_dfLocation("LocationID"))).
      withColumn("level", toInt(_dfLocation("Level"))).
      withColumn("regionId", toIntOrMinus(_dfLocation("RegionID"))).
      withColumn("cityId", toIntOrMinus(_dfLocation("CityID"))).
      select("id", "level", "regionId", "cityId").
      cache()

    dfLocation.registerTempTable("location")

    //User: 0.1G
    val _dfUser = loadDF("UserInfo")
    val dfUser = _dfUser.
      withColumn("id", toInt(_dfUser("UserID"))).
      withColumn("uaId", toInt(_dfUser("UserAgentID"))).
      withColumn("uaOsId", toInt(_dfUser("UserAgentOSID"))).
      withColumn("uaDeviceId", toInt(_dfUser("UserDeviceID"))).
      withColumn("uaFamilyId", toInt(_dfUser("UserAgentFamilyID"))).
      select("id", "uaId", "uaOsId", "uaDeviceId", "uaFamilyId").
      cache()

    dfUser.registerTempTable("user")

    //Ad: 5G
    val _dfAd = loadDF("AdsInfo")
    val dfAd = _dfAd.
      withColumn("id", toInt(_dfAd("AdID"))).
      withColumn("locationId", toIntOrMinus(_dfAd("LocationID"))).
      withColumn("categoryId", toIntOrMinus(_dfAd("CategoryID"))).
      withColumn("params", parseParams(_dfAd("Params"))).
      withColumn("price", toDoubleOrMinus(_dfAd("Price"))).
      withColumn("titleLength", length(_dfAd("Title"))). // skipping this column, since can't process much - maybe later can look at only length
      withColumn("isContext", toInt(_dfAd("IsContext"))).
      select("id", "locationId", "categoryId", "params", "price", "isContext").
      cache()

    dfAd.registerTempTable("ad")

    // Visit : 12G (3.2G)
    val _dfVisit = loadDF("VisitsStream")
    val dfVisit = _dfVisit.
      withColumn("userId", toInt(_dfVisit("UserID"))).
      withColumn("ipId", toInt(_dfVisit("IPID"))).
      withColumn("adId", toInt(_dfVisit("AdID"))).
      withColumn("eventTime", parseTime(_dfVisit("ViewDate"))).
      select("userId", "ipId", "adId", "eventTime").
      cache()

    dfVisit.registerTempTable("visit")

    // Phone Request: 0.6G
    val _dfPhone = loadDF("PhoneRequestsStream")
    val dfPhone = _dfPhone.
      withColumn("userId", toInt(_dfPhone("UserID"))).
      withColumn("ipId", toInt(_dfPhone("IPID"))).
      withColumn("adId", toInt(_dfPhone("AdID"))).
      withColumn("eventTime", parseTime(_dfPhone("PhoneRequestDate"))).
      select("userId", "ipId", "adId", "eventTime").
      cache()

    dfPhone.registerTempTable("phoneRequest")

    // SearchInfo : 8.8G
    // SearchID	SearchDate	IPID	UserID	IsUserLoggedOn	SearchQuery	LocationID	CategoryID	SearchParams
    // 4	2015-05-10 18:11:01.0	898705	3573776	0		3960	22	{83:'Обувь', 175:'Женская одежда', 88:'38'}
    val _dfSearchInfo = loadDF("SearchInfo")
    val dfSearchInfo = _dfSearchInfo.
      withColumn("id", toInt(_dfSearchInfo("SearchID"))).
      withColumn("eventTime", parseTime(_dfSearchInfo("SearchDate"))).
      withColumn("ipId", toInt(_dfSearchInfo("IPID"))).
      withColumn("userId", toInt(_dfSearchInfo("UserID"))).
      withColumn("userLogged", toInt(_dfSearchInfo("IsUserLoggedOn"))).
      withColumn("searchQuery", toLower(_dfSearchInfo("SearchQuery"))).
      withColumn("locationId", toIntOrMinus(_dfSearchInfo("LocationID"))).
      withColumn("categoryId", toIntOrMinus(_dfSearchInfo("CategoryID"))).
      withColumn("params", parseParams(_dfSearchInfo("SearchParams"))).
      select("id", "eventTime", "ipId", "userId", "userLogged", "searchQuery", "locationId", "categoryId", "params").
      cache()

    dfSearchInfo.registerTempTable("searchInfo")

    // SearchStream : 10G  -- trainSearchStream.tsv  (5.9G)
    //SearchID	AdID	Position	ObjectType	HistCTR	IsClick
    // 2	11441863	1	3	0.001804	0
    //3	36256251	2	2
    val _dfSearchStream = loadDF("trainSearchStream")
    val dfSearchStream = _dfSearchStream.
      withColumn("searchId", toInt(_dfSearchStream("SearchID"))).
      withColumn("adId", toInt(_dfSearchStream("AdID"))).
      withColumn("position", toInt(_dfSearchStream("Position"))).
      withColumn("type", toInt(_dfSearchStream("ObjectType"))). //1, 2, 3
      withColumn("histctr", toDoubleOrMinus(_dfSearchStream("HistCTR"))). // only for ObjectType 3
      withColumn("isClick", toIntOrMinus(_dfSearchStream("IsClick"))). // now: -1, 0, 1
      select("searchId", "adId", "position", "type", "histctr", "isClick").
      cache()

    dfSearchStream.registerTempTable("searchStream")


    /*

    val sqlctx = sqlContext
    // Some queries
    sqlctx.sql(
      """
      select v.userId, v.ipId, v.adId, v.eventTime, r.eventTime,
      (r.eventTime -  v.eventTime)*1.0/60 as betweenTime
      from visit v join phoneRequest r
      on (r.adId = v.adId and r.userId = v.userId and r.ipId = v.ipId and r.eventTime > v.eventTime and r.eventTime < v.eventTime + 60*15)
      limit 30
                   """).show


    //  About 5% of visits request phone
    // number of visits that were followed by a phone request within 15 minutes: 14,943,390 (if no distinct 15,602,276)
    sqlctx.sql(
      """
        select count(distinct  v.userId, v.ipId, v.adId, v.eventTime)
        from visit v join phoneRequest r
        on (r.adId = v.adId and r.userId = v.userId and r.ipId = v.ipId and r.eventTime > v.eventTime and r.eventTime < v.eventTime + 60*15)
        limit 30
                   """).show // number of all visits: 286,821,375
    sqlctx.sql(
      """
          select count(1)
          from visit
                   """).show // distinct IP addresses for a given userID: userID collisions and also different internet connections

    val userIps = sqlctx.sql(
        """
      select v.userId, count(distinct v.ipId) as cnt
      from visit v
      group by v.userId
      order by cnt desc
                                 """) // distinct IP addresses for a given userID: userID collisions and also different internet connections
    val userVisits = sqlctx.sql(
        """
        select v.userId, count(v.ipId) as cnt
        from visit v
        group by v.userId
        order by cnt desc
                                    """)
                                    */

    sqlContext
  }
}
