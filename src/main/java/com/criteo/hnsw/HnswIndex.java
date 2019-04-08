package com.criteo.hnsw;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.SizeTPointer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Random;

public class HnswIndex {
    private long pointer;

    public static HnswIndex create(String metric, int dimension) {
        long pointer;
        switch (metric) {
            case Metrics.Angular: pointer = HnswLib.createAngular(dimension); break;
            case Metrics.Euclidean: pointer = HnswLib.createEuclidean(dimension); break;
            case Metrics.DotProduct: pointer = HnswLib.createDotProduct(dimension); break;
            default: throw new UnsupportedOperationException();
        }

        return new HnswIndex(pointer);
    }

    private HnswIndex(long pointer) {
        this.pointer = pointer;
    }

    public long load(String path) {
        HnswLib.loadIndex(pointer, path);
        return getNbItems();
    }

    public void addItem(float[] vector, long id) {
        HnswLib.addItem(pointer, vector, id);
    }

    public void save(String path) {
        HnswLib.saveIndex(pointer, path);
    }

    public void initNewIndex(long maxElements, long M, long efConstruction) {
        initNewIndex(maxElements, M, efConstruction, new Random().nextInt());
    }

    public void initNewIndex(long maxElements, long M, long efConstruction, long randomSeed) {
        HnswLib.initNewIndex(pointer, maxElements, M, efConstruction, randomSeed);
    }

    public void setEf(long ef) {
        HnswLib.setEf(pointer, ef);
    }

    public void unload() {
        HnswLib.destroyIndex(pointer);
    }

    public long getNbItems() {
        return HnswLib.getNbItems(pointer);
    }

    public long[] getIds() {
        SizeTPointer ids = HnswLib.getIdsList(pointer);
        long[] idsArr = new long[(int)ids.limit()];
        for(int i = 0; i < idsArr.length; i ++) {
            idsArr[i] = ids.get(i);
        }
        return idsArr;
    }

    public float[] getItem(long id) {
        FloatPointer vector = HnswLib.getItem(pointer, id);
        float[] vectorArr = new float[(int)vector.limit()];
        for(int i = 0; i < vectorArr.length; i ++) {
            vectorArr[i] = vector.get(i);
        }
        return vectorArr;
    }

    public KnnResult[] knnQuery(float[] vector, long k) {
        SizeTPointer items = new SizeTPointer(k);
        FloatPointer distances = new FloatPointer(k);
        FloatPointer query = new FloatPointer(vector);
        int resultSize = (int)HnswLib.knnQuery(pointer, query, items, distances, k);

        KnnResult[] results = new KnnResult[resultSize];
        for (int i = 0; i < resultSize; i++) {
            results[i] = new KnnResult(items.get(i), distances.get(i));
        }
        return results;
    }
}
