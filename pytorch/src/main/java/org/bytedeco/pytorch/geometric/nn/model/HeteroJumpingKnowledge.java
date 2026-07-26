package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Hetero Version: 对每种节点类型分别做 JK
 */
public class HeteroJumpingKnowledge extends Module {
    private Map<String, JumpingKnowledge> jks;

    public HeteroJumpingKnowledge(Set<String> nodeTypes, String mode, long channels, int numLayers) {
        jks = new HashMap<>();
        for(String nt : nodeTypes) {
            JumpingKnowledge jk = new JumpingKnowledge(mode, channels, numLayers);
            jks.put(nt, jk);
            register_module("jk_" + nt, jk);
        }
    }

    public Map<String, Tensor> forward(Map<String, List<Tensor>> xsDict) {
        Map<String, Tensor> out = new HashMap<>();
        for(String nt : xsDict.keySet()) {
            out.put(nt, jks.get(nt).forward(xsDict.get(nt)));
        }
        return out;
    }
}