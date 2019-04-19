package com.criteo.hnsw;


import java.io.File;
import java.io.IOException;

public class HelloHnsw {
    // TOOD: Move to examples folder
    public static void main(String[] args) throws IOException {
        hnsw();
    }

    public static void hnsw() {
        int dimension = 100;
        int nbItems = 15000;
        float seedValue = 0.5f;
        int M = 16;
        int efConstruction = 200;


        HnswIndex index = HnswIndex.create(Metrics.Euclidean, dimension);

        String path = new File(".", "index-12.hnsw").toString();

        System.out.println("Starting to build index");
        index.initNewIndex(nbItems, M, efConstruction);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getVector(dimension, seedValue / (i + 1));
            index.addItem(vector, i);
            //System.out.println("Set item: " + i);
        }

        System.out.println("Saving " + index.getNbItems() + " items");
        index.save(path);

        index.load(path);
        System.out.println("Loaded" + index.getNbItems() + " items");
        long[] ids = index.getIds();

        for (long id : ids) {
            //System.out.println("Get item: " + id + "; First: " + index.getItem(id)[0]);
        }

        for (int i = 0; i < 20; i++) {
            long queryId = i;
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
