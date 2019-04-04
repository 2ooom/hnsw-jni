# hnswlib low-overhead JVM wrapper
JNI bindings for [HNSW kNN library](https://github.com/nmslib/hnswlib) 
Generated using bytedeco [javacpp library](https://github.com/bytedeco/javacpp)
The intent is provide fast hnsw implmentation for JVM.

## Building
```
gradle init
./gradlew generateLib
```
Will generate JNI bindings in Java and C and compile them for the current plutform.
