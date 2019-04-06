package com.criteo.hnsw;

import java.io.File;

public class HelloHnsw {
    public static void main(String[] args) {
        int dimension = 100;
        int nbItems = 12;
        float seedValue = 0.5f;
        int M = 16;
        int efConstruction = 200;


        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension);

        String path = new File(".", "index-12.hnsw").toString();

        index.initNewIndex(nbItems, M, efConstruction);
        for (int i = 0; i < nbItems; i++) {
            long id = i - 5;
            float[] vector = getVector(dimension, seedValue / (i + 1));
            index.addItem(vector, id);
            System.out.println("Set item: " + id);

        }

        System.out.println("Saving " + index.getNbItems() + " items");
        index.save(path);

        index.load(path);
        System.out.println("Loaded" + index.getNbItems() + " items");
        long[] ids = index.getIds();

        for (long id : ids) {
            System.out.println("Get item: " + id + "; First: " + index.getItem(id)[0]);
        }

        long queryId = -1;
        float[] query = index.getItem(queryId);
        KnnResult[] results = index.knnQuery(query, 3);

        for (KnnResult result : results) {
            System.out.println(queryId + " -> " + result.getItem() + " distance: " + result.getDistance());
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
