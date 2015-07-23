name := "AvitoProject"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.3.1" % "provided",
  "com.github.fommil.netlib" % "all" % "1.1.2" % "provided",
  //"com.github.nscala-time" %% "nscala-time" % "1.8.0",
  "com.databricks" %% "spark-csv" % "1.0.3",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)
