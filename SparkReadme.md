To compile inner `spark-lightautoml` Scala package you have to complete following steps:

### 1. Install Java JDK
In this example we used Amazon Java 8 JDK (Corretto).

How to install Java 8: https://docs.aws.amazon.com/corretto/latest/corretto-8-ug/what-is-corretto-8.html

Here you can find instructions how to install it on different systems (Windows, macOS, Linux).

### 2. Install SBT
To install SBT please follow the instructions here: https://docs.scala-lang.org/getting-started/sbt-track/getting-started-with-scala-and-sbt-on-the-command-line.html#installation

### 3. How to compile
First of all, you have to move into the `spark-lightautoml` directory. It is located in `lightautoml/spark/transformers/spark-lightautoml`.

Then you can call several commands `sbt clean` (to remove old compiled files) and after `sbt package`.

Another way is to call an interactive shell via `sbt shell`. In the shel you should type command `clean` and then `package`.

In the repository a dockerfile is provided (`Scala.dockerfile`). Using this image you can compile code inside  the container without installed java and/or sbt on the host operating system.