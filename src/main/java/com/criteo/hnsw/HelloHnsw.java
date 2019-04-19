package com.criteo.hnsw;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class HelloHnsw {
    public static Random r = new Random();
    public static void main(String[] args) throws IOException {
        int dimension = 100;
        int nbItems = 15000;
        int M = 16;
        int efConstruction = 200;

        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension);

        index.initNewIndex(nbItems, M, efConstruction);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
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

    public static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for(int i = 0; i < dimension; i++) {
            vector[i] = r.nextFloat();
        }
        return vector;
    }
}
