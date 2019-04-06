package com.criteo.hnsw;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class HnswLibTests {
    private static float delta = 1.0E-5f;
    private float seedValue = 1;
    private int dimension = 100;
    private Function<Integer, Float> getValueById = (id) -> id == 0? 0: seedValue / id;
    private Function<Integer, Float> getNormalizedValueById = i -> i == 0? 0.0f: 1/(float)Math.sqrt(dimension);
    private long nbItems = 12;
    private int M = 16;
    private int efConstruction = 200;
    private int randomSeed = 42;


    @Test
    public void create_indices_save_and_load() throws IOException {
        Map<Long, Function<Integer, Float>> indices = new HashMap<Long, Function<Integer, Float>>() {{
            put(HnswLib.createEuclidean(dimension), getValueById);
            put(HnswLib.createDotProduct(dimension), getValueById);
            put(HnswLib.createAngular(dimension), getNormalizedValueById);
        }};

        File dir = Files.createTempDirectory("HnswLib").toFile();
        String indexPathStr = new File(dir, "index.hnsw").toString();

        long nbItems = 123;

        for (Map.Entry<Long, Function<Integer, Float>> entry : indices.entrySet()) {
            long index = entry.getKey();
            Function<Integer, Float> getValueById = entry.getValue();

            HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
            assertEquals(0, HnswLib.getNbItems(index));
            populateIndex(index, getValueById, nbItems, dimension);

            assertEquals(nbItems, HnswLib.getNbItems(index));

            assertAllVectorsMatchExpected(index, dimension, getValueById);
            HnswLib.saveIndex(index, indexPathStr);

            HnswLib.loadIndex(index, indexPathStr);
            assertAllVectorsMatchExpected(index, dimension, getValueById);

            HnswLib.destroyIndex(index);
        }
    }

    @Test
    public void check_Euclidean_index_returns_correct_neighbours() {
        int k = (int)nbItems;
        long index = HnswLib.createEuclidean(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatPointer distances = new FloatPointer(k);
        SizeTPointer items = new SizeTPointer(k);
        FloatPointer query = HnswLib.getItem(index, 0);
        long nbResults = HnswLib.knnQuery(index, query, items, distances, k);

        for(int i = 0; i < nbItems; i++) {
            System.out.println("item: " + items.get(i) + "; distance: " + distances.get(i));
        }
        assertEquals(k, nbResults);

        // 1st result should be self
        assertEquals(0, items.get(0));
        assertEquals(0, distances.get(0), delta);

        for(int i = 1; i < k; i++) {
            long found = items.get(i);
            float distance = distances.get(i);
            assertEquals(k - i, found);
            assertEquals(dimension * (float)Math.pow(getValueById.apply(k - i), 2), distance, delta);
        }

        HnswLib.destroyIndex(index);
    }

    @Test
    public void check_no_error_happens_if_less_items_is_returned_than_k() {
        long biggerK = nbItems * 2;

        long index = HnswLib.createEuclidean(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatPointer distances = new FloatPointer(biggerK);
        SizeTPointer items = new SizeTPointer(biggerK);
        FloatPointer query = HnswLib.getItem(index, 0);
        long nbResults = HnswLib.knnQuery(index, query, items, distances, biggerK);

        for(int i = 0; i < nbItems; i++) {
            System.out.println("item: " + items.get(i) + "; distance: " + distances.get(i));
        }
        assertEquals(nbItems, nbResults);

        for(int i = 1; i < nbItems; i++) {
            long found = items.get(i);
            float distance = distances.get(i);
            assertEquals(nbItems - i, found);
            assertEquals(dimension * (float)Math.pow(getValueById.apply((int)nbItems - i), 2), distance, delta);
        }

        HnswLib.destroyIndex(index);
    }

    @Test @Ignore("Need to implement")
    public void check_Angular_index_returns_correct_neighbours() {
        int k = (int)nbItems;
        long index = HnswLib.createAngular(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);

        // TODO: Implement
        HnswLib.destroyIndex(index);
    }


    @Test @Ignore("Inner product of A and B is implemented in hnswlib (space_ip.h) as A*B = 1 - (A1*B1 + A2*B2 + …)")
    public void check_DotProduct_index_returns_correct_neighbours() {
        int k = (int)nbItems;
        long index = HnswLib.createDotProduct(dimension);

        HnswLib.initNewIndex(index, nbItems, M, efConstruction, randomSeed);
        populateIndex(index, getValueById, nbItems, dimension);

        FloatPointer distances = new FloatPointer(k);
        SizeTPointer items = new SizeTPointer(k);

        for(int j = 0; j < nbItems; j++) {
            FloatPointer query = HnswLib.getItem(index, j);
            HnswLib.knnQuery(index, query, items, distances, k);

            for (int i = 0; i < nbItems; i++) {
                System.out.println(j + " -> " + items.get(i) + ": " + distances.get(i));
            }
            assertEquals(k, distances.limit());
            assertEquals(k, items.limit());

            for (int i = 0; i < k; i++) {
                long found = items.get(i);
                float distance = distances.get(i);
                assertEquals(k - i, found);
                assertEquals(1 - dimension * getValueById.apply(k - i) * getValueById.apply(j), distance, delta);
            }
        }
        HnswLib.destroyIndex(index);
    }

    private void populateIndex(long index, Function<Integer, Float> getValueById, long nbItems, int dimension) {
        for (int i = 0; i < nbItems; i++) {
            HnswLib.addItem(index, getVector(dimension, getValueById.apply(i)), i);
        }
    }

    private void assertAllVectorsMatchExpected(long index, int size, Function<Integer, Float> getExpectedValue) {
        SizeTPointer ids = HnswLib.getIdsList(index);
        for (int i = 0; i < ids.limit(); i++) {
            int id = (int)ids.get(i);

            float expectedValue = getExpectedValue.apply(id);
            FloatPointer item = HnswLib.getItem(index, id);
            assertAllValuesEqual(item, size, expectedValue);
        }
    }

    private void assertAllValuesEqual(FloatPointer vector, int size, float expectedValue) {
        assertEquals(size, vector.limit());
        for (long j = 0; j < size; j++) {
            assertEquals(expectedValue, vector.get(j), delta);
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
