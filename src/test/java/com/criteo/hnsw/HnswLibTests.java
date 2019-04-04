package com.criteo.hnsw;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import static org.junit.Assert.assertEquals;

public class HnswLibTests {
    private static float delta = 0.00001f;

    @Test
    public void createIndicesNonNormalizedOfEachTypeSaveAndLoad() throws IOException {
        //System.out.println(Paths.get(getClass().getProtectionDomain().getCodeSource().getLocation().toURI().getPath()).toString());
        int dimension = 100;
        long nbItems = 12;
        float seedValue = 0.5f;
        int M = 16;
        int efConstruction = 200;
        int randomSeed = 42;


        long[] indices = new long[]{
            HnswLib.createEuclidean(dimension),
            HnswLib.createDotProduct(dimension),
        };

        File dir = Files.createTempDirectory("HnswLib").toFile();
        String indexPathStr = new File(dir, "index.hnsw").toString();
        System.out.println(indexPathStr);

        for (long index : indices) {
            HnswLib.init_new_index(index, nbItems, M, efConstruction, randomSeed);
            assertEquals(0, HnswLib.getNItems(index));
            for (int i = 0; i < nbItems; i++) {
                long id = i - 5;
                HnswLib.addItem(index, getVector(dimension, seedValue / (i + 1)), id);
            }

            assertEquals(nbItems, HnswLib.getNItems(index));
            SizeTPointer ids = HnswLib.getIdsList(index);
            for (int i = 0; i < ids.limit(); i++) {
                long id = ids.get(i);
                float expectedValue = seedValue / (id+5 + 1);
                FloatPointer item = HnswLib.getItem(index, id);
                assertEquals(dimension, item.limit());
                for (int j = 0; j < item.limit(); j++) {
                    assertEquals(expectedValue, item.get(j), delta);
                }
            }
            HnswLib.saveIndex(index, indexPathStr);

            HnswLib.loadIndex(index, indexPathStr, nbItems);
            ids = HnswLib.getIdsList(index);
            for (int i = 0; i < ids.limit(); i++) {
                long id = ids.get(i);
                float expectedValue = seedValue / (id+5 + 1);
                FloatPointer item = HnswLib.getItem(index, id);
                assertEquals(dimension, item.limit());
                for (long j = 0; j < item.limit(); j++) {
                    assertEquals(expectedValue, item.get(j), delta);
                }
            }
        }
    }

    private FloatPointer getVector(int dimension, float value) {
        FloatPointer vector = new FloatPointer(dimension);
        for(long i = 0; i < dimension; i++) {
            vector.put(i, value);
        }
        return vector;
    }
}
