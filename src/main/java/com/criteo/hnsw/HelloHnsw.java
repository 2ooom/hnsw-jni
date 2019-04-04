package com.criteo.hnsw;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.SizeTPointer;

import java.io.File;

public class HelloHnsw {
    public static void main(String[] args) {
        int dimension = 100;
        int nbItems = 12;
        float seedValue = 0.5f;
        int M = 16;
        int efConstruction = 200;


        long index = HnswLib.createEuclidean(dimension);

        String indexPathStr = new File(".", "index-12.hnsw").toString();

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, 42);
        for (int i = 0; i < nbItems; i++) {
            long id = i - 5;
            HnswLib.addItem(index, getVector(dimension, seedValue / (i + 1)), id);
            System.out.println("Set item: " + id);

        }

        System.out.println("Saving " + HnswLib.getNItems(index) + " items");
        HnswLib.saveIndex(index, indexPathStr);

        HnswLib.loadIndex(index, indexPathStr, nbItems);
        System.out.println("Loaded" + HnswLib.getNItems(index) + " items");
        SizeTPointer ids = HnswLib.getIdsList(index);

        for (long i = 0; i < ids.limit(); i++){
            long id = ids.get(i);
            System.out.println("Get item: " + id);
            FloatPointer v = HnswLib.getItem(index, id);

            System.out.println(v.get(0));
        }
    }


    private static FloatPointer getVector(int dimension, float value) {
        FloatPointer vector = new FloatPointer(dimension);
        for(long i = 0; i < dimension; i++) {
            vector.put(i, value);
        }
        return vector;
    }
}
