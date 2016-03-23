package com.sparkydots.kaggle.hd.features

import org.scalatest.{Matchers, FlatSpec}
import scala.io.Source

class FeaturesTest extends FlatSpec with Matchers {


  "Features" should "decode" in {
    val pattern = raw"\[([0-9]+),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\)\]".r
    //[102853,WrappedArray(popcorn, paint),WrappedArray(),WrappedArray(paint),WrappedArray(),WrappedArray(),WrappedArray(),WrappedArray()]
    val testit = "[102853,WrappedArray(popcorn, paint),WrappedArray(),WrappedArray(paint),WrappedArray(),WrappedArray(),WrappedArray(),WrappedArray()]"

    testit match {
      case pattern(id, a1, a2, a3, a4, a5, a6, a7) =>
        println(id)
        println(a1)
        println(a2)
    }


    import java.io._
    val file = new File("/Users/rbekbolatov/data/kaggle/homedepot/matched_strings_clean.csv")
    val bw = new BufferedWriter(new FileWriter(file))

    for (line <- Source.fromFile("/Users/rbekbolatov/data/kaggle/homedepot/matched_strings.csv").getLines()) {
      line match {
        case pattern(id, a1, a2, a3, a4, a5, a6, a7) =>
          bw.write(s"""$id,"$a1","$a2","$a3","$a4","$a5","$a6","$a7"""")
          bw.newLine()
      }
    }

    bw.close()
  }
}
