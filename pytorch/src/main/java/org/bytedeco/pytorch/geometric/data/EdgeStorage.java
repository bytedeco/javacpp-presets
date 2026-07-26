package org.bytedeco.pytorch.geometric.data;


import org.bytedeco.pytorch.Tensor;

// 边存储
public class EdgeStorage extends Storage {
    public Tensor getEdgeIndex() { return (Tensor) get("edge_index"); }
    public void setEdgeIndex(Tensor edgeIndex) { put("edge_index", edgeIndex); }

    public int getNumEdges() {
        Tensor edgeIndex = getEdgeIndex();
        return edgeIndex != null ? (int) edgeIndex.size(1) : 0;
    }
}
