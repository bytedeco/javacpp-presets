package org.bytedeco.pytorch.geometric.data;

// 异质图关系键 (Triple: src -> rel -> dst)
public record EdgeRel(String src, String rel, String dst) {
    @Override
    public String toString() {
        return String.format("('%s', '%s', '%s')", src, rel, dst);
    }
}
