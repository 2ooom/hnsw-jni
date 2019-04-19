# hnswlib low-overhead JVM wrapper
JNI bindings for [HNSW kNN library](https://github.com/nmslib/hnswlib) 
Generated using bytedeco [javacpp library](https://github.com/bytedeco/javacpp)
The intent is provide fast hnsw implmentation for JVM.

## Building
```
gradle init
./gradlew generateLib
```
Will generate JNI bindings in Java and C and compile them for the current platform.


## Usage
Here is example that creates 15k embeddings (size 100), saving them into the index on disk, reading the index and retrieving some distances.

```java

import java.io.File;
import java.io.IOException;

public class HelloHnsw {

    public static void main(String[] args) throws IOException {
        int dimension = 100;
        int nbItems = 15000;
        float seedValue = 0.5f;
        int M = 16;
        int efConstruction = 200;

        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension);

        index.initNewIndex(nbItems, M, efConstruction);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getVector(dimension, seedValue / (i + 1));
            index.addItem(vector, i);
        }

        String path = new File("./bin", "index.hnsw").toString();
        index.save(path);

        index.load(path);
        System.out.println("Loaded" + index.getNbItems() + " items");
        long[] ids = index.getIds();

        for (int i = 0; i < 20; i++) {
            long queryId = ids[i];
            float[] query = index.getItem(queryId);
            KnnResult[] results = index.knnQuery(query, 3);
            for (KnnResult result : results) {
                System.out.println(queryId + " -> " + result.getItem() + " distance: " + result.getDistance());
            }
        }
        index.unload();
    }


    private static float[] getVector(int dimension, float value) {
        float[] vector = new float[dimension];
        for(int i = 0; i < dimension; i++) {
            vector[i] = value;
        }
        return vector;
    }
}
```

To run same example from repo:

```
./gradlew run
```

To run tests:

```
./gradlew test
```