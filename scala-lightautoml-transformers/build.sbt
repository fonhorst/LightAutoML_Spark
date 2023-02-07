name := "spark-lightautoml"

version := "0.1.1"

scalaVersion := "2.12.15"

//idePackagePrefix := Some("org.apache.spark.ml.feature.lightautoml")

resolvers ++= Seq(
  ("Confluent" at "http://packages.confluent.io/maven")
        .withAllowInsecureProtocol(true)
)

val sparkVersion = "3.1.3"

libraryDependencies ++= Seq(
    "com.microsoft.azure" % "synapseml_2.12" % "0.9.5-35-e962330b-SNAPSHOT",
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "org.scalatest" %% "scalatest" % "3.2.14" % Test
)

// uncomment the following lines if you need to build a fat jar
//lazy val app = (project in file("."))
//assemblyMergeStrategy in assembly := {
//    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//    case x => MergeStrategy.first
//}
//assembly / assemblyJarName := "spark-lightautoml-assembly-fatjar-0.1.1.jar"
