name := "AvitoProject"

version := "1.0"

scalaVersion := "2.10.4"

val sparkVersion = "1.4.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
//  "com.github.fommil.netlib" % "all" % "1.1.2" % "provided",
  "com.databricks" %% "spark-csv" % "1.0.3",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)
