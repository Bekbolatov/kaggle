package com.sparkydots.util.io

import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets

object FileIO {

  def write(lines: Seq[String], path: String = "/Users/renatb/data/kaggle/taxi_trip/pathsForTripA.csv") {
    val content = lines.mkString("\n").getBytes(StandardCharsets.UTF_8)
    Files.write(Paths.get(path), content)
  }
}
