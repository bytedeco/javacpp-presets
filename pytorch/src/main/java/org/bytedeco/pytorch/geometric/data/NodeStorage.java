package org.bytedeco.pytorch.geometric.data;

import org.bytedeco.pytorch.Tensor;

// 节点存储
public class NodeStorage extends Storage {
    public Tensor getX() { return (Tensor) get("x"); }
    public void setX(Tensor x) { put("x", x); }

    public int getNumNodes() {
        Tensor x = getX();
        return x != null ? (int) x.size(0) : 0;
    }
}
