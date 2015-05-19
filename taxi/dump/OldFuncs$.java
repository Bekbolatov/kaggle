package com.sparkydots.kaggle.taxi;

object OldFuncs {

  def travelTime(n: Int): Double = (n - 1) * 15
  def logTravelTime(n: Int): Double = math.log(travelTime(n) + 1)

  def readFile(sqlContext: SQLContext, fileName: String): DataFrame = sqlContext.csvFile(s"/user/ds/data/kaggle/taxi/$fileName.csv")

  def means(data: RDD[((String, Long), Double)]): Map[(String, Long), Double] = {
    data.groupByKey.mapValues { mcs =>
      val n = mcs.size
      val sum = mcs.sum
      sum/n
    }.collect.toMap
  }

  def tripData(row: Row): (String, Long, Boolean, Double) = {
    val callType = row(1).toString
    val timeHour = (row(5).toString.toLong % 86400) / 24
    val missing = row(7).toString == "TRUE"
    val travel = if (row(8).toString.length < 5) {
      0
    } else {
      travelTime(row(8).toString.drop(2).dropRight(2).split("\\],\\[").map(xy => xy.split(",").map(_.toDouble)).length)
    }
    (callType, timeHour, missing, travel)
  }
  def readData(sc: SparkContext, fileName: String): RDD[(String, Long, Boolean, Double)] = {
    val sqlContext = new SQLContext(sc)
    val df = readFile(sqlContext, fileName)
    df.map(tripData)
  }

  def tripDataFeatures(row: Row): Option[LabeledPoint] = {

    val b = new ArrayBuffer[Double]()

    //1.TRIP_ID: (String) It contains an unique identifier for each trip;
    // --skipped--

    //2.CALL_TYPE: (char) It identifies the way used to demand this service. It may contain one of three possible values:
    //‘A’ if this trip was dispatched from the central;
    //‘B’ if this trip was demanded directly to a taxi driver on a specific stand;
    //‘C’ otherwise (i.e. a trip demanded on a random street).
//      b ++= (row(1) match {
//        case "A" => Array(1.0, 0.0, 0.0)
//        case "B" => Array(0.0, 1.0, 0.0)
//        case "C" => Array(0.0, 0.0, 1.0)
//        case "_" => Array(0.0, 0.0, 0.0)
//      })

    //3.ORIGIN_CALL: (integer) It contains an unique identifier for each phone number which was used to demand, at least, one service. It identifies the trip’s customer if CALL_TYPE=’A’. Otherwise, it assumes a NULL value;
    // --skipped--

    //4.ORIGIN_STAND: (integer): It contains an unique identifier for the taxi stand.
    //It identifies the starting point of the trip if CALL_TYPE=’B’. Otherwise, it assumes a NULL value;
    // --skipped--

    //5.TAXI_ID: (integer): It contains an unique identifier for the taxi driver that performed each trip;
    // --skipped--

    //6.TIMESTAMP: (integer) Julian Timestamp (in seconds). It identifies the trip’s start;
//      b += (row(5).toString.toLong % 120)
    b += (row(5).toString.toLong % 86400) / 24

    //7.DAYTYPE: (char) It identifies the daytype of the trip’s start. It assumes one of three possible values:
    //‘B’ if this trip started on a holiday or any other special day (i.e. extending holidays, floating holidays, etc.);
    //‘C’ if the trip started on a day before a type-B day;
    //‘A’ otherwise (i.e. a normal day, workday or weekend).
//      b ++= (row(6) match {
//        case "A" => Array(1.0, 0.0, 0.0)
//        case "B" => Array(0.0, 1.0, 0.0)
//        case "C" => Array(0.0, 0.0, 1.0)
//        case "_" => Array(0.0, 0.0, 0.0)
//      })

    //8.MISSING_DATA: (Boolean) It is FALSE when the GPS data stream is complete and TRUE whenever one (or more) locations are missing

    //9.POLYLINE: (String): It contains a list of GPS coordinates (i.e. WGS84 format) mapped as a string. The beginning and the end of the string are identified with brackets (i.e. [ and ], respectively). Each pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]. This list contains one pair of coordinates for each 15 seconds of trip. The last list item corresponds to the trip’s destination while the first one represents its start;

    val pathSegments = if (row(8).toString.length < 5) {
      None
    } else {
      Some(row(8).toString.drop(2).dropRight(2).split("\\],\\[").map(xy => xy.split(",").map(_.toDouble)))
    }

    pathSegments.map { segments =>

      val label = travelTime(segments.length)

      LabeledPoint(label, Vectors.dense(b.toArray))

    }
  }



}
