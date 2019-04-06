package com.criteo.hnsw;

public class KnnResult {
    private long item;
    private float distance;

    public long getItem() {
        return item;
    }

    public float getDistance() {
        return distance;
    }

    public KnnResult(long item, float distance) {
        this.item = item;
        this.distance = distance;
    }
}

