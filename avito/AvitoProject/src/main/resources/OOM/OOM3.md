9node m3 cluster
SPARK_REPL_OPTS="-XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=712m -Xmx=8g" spark-shell --jars AvitoProject-assembly-1.0.jar

spark.serializer org.apache.spark.serializer.KryoSerializer



15/07/21 02:38:58 ERROR util.Utils: Uncaught exception in thread task-result-getter-2
java.lang.OutOfMemoryError: Java heap space
        at org.apache.spark.scheduler.DirectTaskResult$$anonfun$readExternal$1.apply$mcV$sp(TaskResult.scala:61)
        at org.apache.spark.util.Utils$.tryOrIOException(Utils.scala:1138)
        at org.apache.spark.scheduler.DirectTaskResult.readExternal(TaskResult.scala:58)
        at java.io.ObjectInputStream.readExternalData(ObjectInputStream.java:1837)
        at java.io.ObjectInputStream.readOrdinaryObject(ObjectInputStream.java:1796)
        at java.io.ObjectInputStream.readObject0(ObjectInputStream.java:1350)
        at java.io.ObjectInputStream.readObject(ObjectInputStream.java:370)
        at org.apache.spark.serializer.JavaDeserializationStream.readObject(JavaSerializer.scala:68)
        at org.apache.spark.serializer.JavaSerializerInstance.deserialize(JavaSerializer.scala:88)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2$$anonfun$run$1.apply$mcV$sp(TaskResultGetter.scala:75)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2$$anonfun$run$1.apply(TaskResultGetter.scala:51)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2$$anonfun$run$1.apply(TaskResultGetter.scala:51)
        at org.apache.spark.util.Utils$.logUncaughtExceptions(Utils.scala:1618)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2.run(TaskResultGetter.scala:50)
        at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
        at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
        at java.lang.Thread.run(Thread.java:745)
Exception in thread "task-result-getter-2" java.lang.OutOfMemoryError: Java heap space
        at org.apache.spark.scheduler.DirectTaskResult$$anonfun$readExternal$1.apply$mcV$sp(TaskResult.scala:61)
        at org.apache.spark.util.Utils$.tryOrIOException(Utils.scala:1138)
        at org.apache.spark.scheduler.DirectTaskResult.readExternal(TaskResult.scala:58)
        at java.io.ObjectInputStream.readExternalData(ObjectInputStream.java:1837)
        at java.io.ObjectInputStream.readOrdinaryObject(ObjectInputStream.java:1796)
        at java.io.ObjectInputStream.readObject0(ObjectInputStream.java:1350)
        at java.io.ObjectInputStream.readObject(ObjectInputStream.java:370)
        at org.apache.spark.serializer.JavaDeserializationStream.readObject(JavaSerializer.scala:68)
        at org.apache.spark.serializer.JavaSerializerInstance.deserialize(JavaSerializer.scala:88)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2$$anonfun$run$1.apply$mcV$sp(TaskResultGetter.scala:75)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2$$anonfun$run$1.apply(TaskResultGetter.scala:51)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2$$anonfun$run$1.apply(TaskResultGetter.scala:51)
        at org.apache.spark.util.Utils$.logUncaughtExceptions(Utils.scala:1618)
        at org.apache.spark.scheduler.TaskResultGetter$$anon$2.run(TaskResultGetter.scala:50)
        at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
        at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
        at java.lang.Thread.run(Thread.java:745)