package com.sparkydots.kaggle.avito

import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level

/*
spark-shell --jars AvitoProject-assembly-1.0.jar
val (sqlContext, dfCategory, dfLocation, dfUser, dfAd, dfVisit, dfPhone, dfSearchInfo, dfSearchStream, dfSearchStreamToFind) = com.sparkydots.kaggle.avito.ReadData.ingest(sc)
 */

object ReadData {

  def ingest(sc: SparkContext) = {
    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    import Functions._
    import UdfFunctions._

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    sqlContext.udf.register("strLen", (s: String) => s.length())
    sqlContext.udf.register("errf", _error)
    //val df2 = sqlctx.sql("select id, errf(histctr, 0) as err from searchStreamToFind where histctr >= 0.0 ")

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
      withColumn("title", toLower(_dfAd("Title"))).
      withColumn("isContext", toInt(_dfAd("IsContext"))).
      select("id", "locationId", "categoryId", "params", "price", "title", "isContext").
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

    // SearchStream : 0.5G  -- testSearchStream.tsv
    //ID	SearchID	AdID	Position	ObjectType	HistCTR
    //1	1	10915336	1	3	0.004999
    //2	1	12258424	6	1
    val _dfSearchStreamToFind = loadDF("testSearchStream")
    val dfSearchStreamToFind = _dfSearchStreamToFind.
      withColumn("id", toInt(_dfSearchStreamToFind("ID"))).
      withColumn("searchId", toInt(_dfSearchStreamToFind("SearchID"))).
      withColumn("adId", toInt(_dfSearchStreamToFind("AdID"))).
      withColumn("position", toInt(_dfSearchStreamToFind("Position"))).
      withColumn("type", toInt(_dfSearchStreamToFind("ObjectType"))). //1, 2, 3
      withColumn("histctr", toDoubleOrMinus(_dfSearchStreamToFind("HistCTR"))). // only for ObjectType 3
      select("id", "searchId", "adId", "position", "type", "histctr").
      cache()

    dfSearchStreamToFind.registerTempTable("searchStreamToFind")


    // force reads
    sqlContext.sql("select count(1) from ad").show
    sqlContext.sql("select count(1) from location").show
    sqlContext.sql("select count(1) from category").show
    sqlContext.sql("select count(1) from user").show
    sqlContext.sql("select count(1) from visit").show
    sqlContext.sql("select count(1) from phoneRequest").show
    sqlContext.sql("select count(1) from searchInfo").show
    sqlContext.sql("select count(1) from searchStream").show
    sqlContext.sql("select count(1) from searchStreamToFind").show



    /*

    val sqlctx = sqlContext
    // Some queries
    sqlctx.sql(

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




val (sqlContext, dfCategory, dfLocation, dfUser, dfAd, dfVisit, dfPhone, dfSearchInfo, dfSearchStream, dfSearchStreamToFind) = com.sparkydots.kaggle.avito.ReadData.ingest(sc)
sqlContext.sql("select count(1) from ad").show
sqlContext.sql("select count(1) from location").show
sqlContext.sql("select count(1) from category").show
sqlContext.sql("select count(1) from user").show
sqlContext.sql("select count(1) from visit").show
sqlContext.sql("select count(1) from phoneRequest").show
sqlContext.sql("select count(1) from searchInfo").show
sqlContext.sql("select count(1) from searchStream").show
sqlContext.sql("select count(1) from searchStreamToFind").show

//there is a searchInfo for each search item
sqlContext.sql("select count(1) from searchStreamToFind s left outer join searchInfo i on (s.searchId = i.id) where i.id is not null").show()
sqlContext.sql("select count(distinct i.userId, s.eventTime) from searchStreamToFind s left outer join searchInfo i on (s.searchId = i.id) where s.type = 3 and i.id is not null").show()
sqlContext.sql("select i.userId, count(1) as cnt from searchStreamToFind s left outer join searchInfo i on (s.searchId = i.id) where s.type = 3 and i.id is not null group by i.userId order by cnt desc limit 10").show()

sqlContext.sql("select count(distinct i.userId) from searchStream s left outer join searchInfo i on (s.searchId = i.id)").show()

sqlContext.sql("select i.id, count(distinct s.position) as cnt from searchStreamToFind s left outer join searchInfo i on (s.searchId = i.id) where s.type = 3 group by i.id order by cnt desc limit 20").show(20)

sqlContext.sql("select *  from searchStreamToFind s left outer join searchInfo i on (s.searchId = i.id)  order by i.userId limit 50").show(50)

sqlContext.sql("select min(i.eventTime), max(i.eventTime) from searchStreamToFind s left outer join searchInfo i on (s.searchId = i.id) where s.type = 3 and i.id is not null").show()

    val inters = sqlContext.sql("""
      select s.searchId, s.adId, s.position, s.type, s.histctr, s.isClick,
      i.id, i.eventTime, i.ipId, i.userId, i.userLogged, i.searchQuery, i.locationId, i.categoryId, i.params,
      a.id, a.locationId, a.categoryId, a.params, a.price, a.title, a.isContext
      from searchStream s
      left outer join searchInfo i on (s.searchId = i.id)
      left outer join ad a on (s.adId = a.id)
      """).cache()

    sqlContext.sql(
      """
        | select count(1)
        | from searchStream s left outer join
      """.stripMargin





    // User Ids: touched in test set and the ones not touched yet
    // We need to remove all users that are involved in final test/submission set to avoid data contamination
    val dfTouchedUsers = sqlContext.sql(
      """
        |select distinct i.userId as id
        |from searchStreamToFind s
        |left outer join searchInfo i on (s.searchId = i.id)
        |where i.id is not null
      """.stripMargin).cache()

    dfTouchedUsers.registerTempTable("touchedUsers")

    val dfUntouchedUsers = sqlContext.sql(
      """
        |select distinct i.userId as id
        |from searchStream s
        |left outer join searchInfo i on (s.searchId = i.id)
        |left outer join touchedUsers tu on (tu.id = i.userId)
        |where i.id is not null and tu.id is null
      """.stripMargin).cache()

    dfUntouchedUsers.registerTempTable("untouchedUsers")



                                    */

    (sqlContext, dfCategory, dfLocation, dfUser, dfAd, dfVisit, dfPhone, dfSearchInfo, dfSearchStream, dfSearchStreamToFind)
  }
}
