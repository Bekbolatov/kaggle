name := "taxi"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.3.1" % "provided",
  "com.databricks" %% "spark-csv" % "1.0.3",
  "com.github.nscala-time" %% "nscala-time" % "2.0.0"
)
