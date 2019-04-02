package com.criteo.hnsw;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        value = @Platform(
                compiler = {"cpp11"},
                include = {"hnswjava.h"}
        ),
        target = "com.criteo.hnsw",
        global = "com.criteo.hnsw.HnswLib"
)
public class HnswLibConfig implements InfoMapper {
    public void map(InfoMap infoMap) {
    }
}