import sbt.Keys.resolvers

name := "scala-transformers"

version := "0.1"

scalaVersion := "2.12.12"

idePackagePrefix := Some("lightautoml.scala.transformers")

resolvers ++= Seq(
  ("Confluent" at "http://packages.confluent.io/maven")
        .withAllowInsecureProtocol(true)
)

mainClass := Some("lightautoml.scala.transformers.test_udfs")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "org.apache.spark" %% "spark-mllib" % "3.2.0",
  "ml.dmlc" %% "xgboost4j" % "1.5.2",
  "ml.dmlc" %% "xgboost4j-spark" % "1.5.2"
)
