package org.bytedeco.pytorch.geometric.sampler;

import org.bytedeco.pytorch.Tensor;

import java.util.HashMap;
import java.util.Map;

public class HeteroAdj {
    // edgeType (e.g., "user__buys__item") -> Tensor(CSR format)
    public Map<String, Tensor> rowPtr = new HashMap<>();
    public Map<String, Tensor> colIndex = new HashMap<>();

    public void addEdgeType(String edgeType, Tensor rowPtr, Tensor colIndex) {
        this.rowPtr.put(edgeType, rowPtr);
        this.colIndex.put(edgeType, colIndex);
    }
}