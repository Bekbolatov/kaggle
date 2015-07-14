package com.sparkydots.kaggle.avito

//libraryDependencies += "com.github.nscala-time" %% "nscala-time" % "1.8.0"
//http://stackoverflow.com/questions/3614380/whats-the-standard-way-to-work-with-dates-and-times-in-scala-should-i-use-java
//http://stackoverflow.com/questions/16996549/how-to-convert-string-to-date-time-in-scala
//import com.github.nscala_time.time.Imports._
//DateTime.parse("2014-07-06")

//  spark-shell --packages "com.databricks:spark-csv_2.10:1.0.3,com.github.nscala-time:nscala-time_2.10:2.0.0"
object Ingest {

  def ingest(sc: SparkContext) {


    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._
    val toInt = udf[Int, String]( _.toInt)
    val toIntOrMinus = udf[Int, String]( (s:String) => Try(s.toInt).getOrElse(-1))

    // temporary
    val parseTime = udf((tt: String) => {
      val format = new java.text.SimpleDateFormat("yyyy-MM-dd hh:mm:ss.0")
      format.parse(tt).getTime/1000
      //val Array(dt, tm) = tt.split(" ")
      //val Array(year, month, day) = dt.split("-")
      //val Array(hour, minute, second) = tm.split("\\.")(0).split(":")
    })
    sqlContext.udf.register("strLen", (s: String) => s.length())


    //  Data Loc
    val dataLoc = "s3n://sparkydotsdata/kaggle/avito/"
    def loadDF(filename: String) = sqlContext.load("com.databricks.spark.csv", Map("header" -> "true", "delimiter" -> "\t", "path" -> s"${dataLoc}${filename}.tsv"))

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

    // Visit
    val _dfVisit = loadDF("VisitsStream")
    val dfVisit = _dfVisit.
    withColumn("userId", toInt(_dfVisit("UserID"))).
    withColumn("ipId", toInt(_dfVisit("IPID"))).
    withColumn("adId", toInt(_dfVisit("AdID"))).
    withColumn("eventTime", parseTime(_dfVisit("ViewDate"))).
    select("userId", "ipId", "adId", "eventTime").
    cache()

    dfVisit.registerTempTable("visit")

    // Phone Requests
    val _dfPhone = loadDF("PhoneRequestsStream")
    val dfPhone = _dfPhone.
    withColumn("userId", toInt(_dfPhone("UserID"))).
    withColumn("ipId", toInt(_dfPhone("IPID"))).
    withColumn("adId", toInt(_dfPhone("AdID"))).
    withColumn("eventTime", parseTime(_dfPhone("PhoneRequestDate"))).
    select("userId", "ipId", "adId", "eventTime").
    cache()

    dfPhone.registerTempTable("phoneRequest")

    // Some queries
    sqlContext.sql("""
      select v.userId, v.ipId, v.adId, v.eventTime, r.eventTime,
      (r.eventTime -  v.eventTime)*1.0/60 as betweenTime
      from visit v join phoneRequest r
      on (r.adId = v.adId and r.userId = v.userId and r.ipId = v.ipId and r.eventTime > v.eventTime and r.eventTime < v.eventTime + 60*15)
      limit 30
      """).show


      //  About 5% of visits request phone
      // number of visits that were followed by a phone request within 15 minutes: 14,943,390 (if no distinct 15,602,276)
      sqlContext.sql("""
        select count(distinct  v.userId, v.ipId, v.adId, v.eventTime)
        from visit v join phoneRequest r
        on (r.adId = v.adId and r.userId = v.userId and r.ipId = v.ipId and r.eventTime > v.eventTime and r.eventTime < v.eventTime + 60*15)
        limit 30
        """).show

        // number of all visits: 286,821,375
        sqlContext.sql("""
          select count(1)
          from visit
          """).show


    // distinct IP addresses for a given userID: userID collisions and also different internet connections
    val userIps = sqlContext.sql("""
      select v.userId, count(distinct v.ipId) as cnt
      from visit v
      group by v.userId
      order by cnt desc
      """)

      // distinct IP addresses for a given userID: userID collisions and also different internet connections
      val userVisits = sqlContext.sql("""
        select v.userId, count(v.ipId) as cnt
        from visit v
        group by v.userId
        order by cnt desc
        """)

        /*
        userVisits.filter("cnt > 9900").show
        
        userId  cnt
1203081 91155  => also in phonerequests: 87128 dfPhone.filter("userId = 1203081").count
1987990 81749
4263912 81210
769640  37953
1113291 37345
4285777 31443
2030364 30271
2595408 22975
2596770 22518
4325557 20128
2765584 14800
1678977 13535
2020992 12540
302342  12138
1020711 11940
332472  11938
2346153 11867
2057229 10596
1509209 9948
*/


  }
}
