/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/PLE.scala (CGC)
 *
 * CGC — Customized Gate Control layer (per level of PLE).
 * Task-specific experts + shared experts + task gates (+ shared gate except last level).
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CGC extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nTask;
    private final int nExpertSpecific;
    private final int nExpertShared;
    private final boolean hasSharedGate;
//    private final List<MLP> expertsSpecific = new ArrayList<>();
//    private final List<MLP> expertsShared = new ArrayList<>();
//    private final List<MLP> gatesSpecific = new ArrayList<>();
    private final ModuleListImpl expertsSpecific = new ModuleListImpl();
    private final ModuleListImpl expertsShared = new ModuleListImpl();
    private final ModuleListImpl gatesSpecific = new ModuleListImpl();
    private final MLP gateShared; // nullable when last level

    public CGC(
            int curLevel,
            int nLevel,
            int nTask,
            int nExpertSpecific,
            int nExpertShared,
            int inputDims,
            Map<String, Object> expertParams,
            String device) {
        super("CGC");
        this.nTask = nTask;
        this.nExpertSpecific = nExpertSpecific;
        this.nExpertShared = nExpertShared;
        this.hasSharedGate = curLevel < nLevel;

        @SuppressWarnings("unchecked")
        List<Long> dimsList = expertParams != null && expertParams.containsKey("dims")
                ? (List<Long>) expertParams.get("dims")
                : List.of(128L);
        long[] expertDims = toLongArray(dimsList);
        long expertLast = expertDims[expertDims.length - 1];
        String expertActivation = expertParams != null && expertParams.containsKey("activation")
                ? String.valueOf(expertParams.get("activation"))
                : "relu";
        float expertDropout = expertParams != null && expertParams.containsKey("dropout")
                ? ((Number) expertParams.get("dropout")).floatValue()
                : 0.0f;

        long nExpertPerTask = (long) nExpertSpecific + nExpertShared;
        long nExpertAll = (long) nExpertSpecific * nTask + nExpertShared;

        for (int i = 0; i < nTask * nExpertSpecific; i++) {
            MLP m = new MLP(inputDims, expertDims, expertLast, expertActivation, expertDropout,
                    false, false, false, device);
            register_module("expert_specific_" + i, m);
            expertsSpecific.insert(i,m);
        }

        for (int i = 0; i < nExpertShared; i++) {
            MLP m = new MLP(inputDims, expertDims, expertLast, expertActivation, expertDropout,
                    false, false, false, device);
            register_module("expert_shared_" + i, m);
            expertsShared.insert(i,m);
        }

        for (int i = 0; i < nTask; i++) {
            MLP m = new MLP(inputDims, new long[]{nExpertPerTask}, nExpertPerTask, "softmax", 0.0f,
                    false, false, false, device);
            register_module("gate_specific_" + i, m);
            gatesSpecific.insert(i,m);
        }

        if (hasSharedGate) {
            MLP m = new MLP(inputDims, new long[]{nExpertAll}, nExpertAll, "softmax", 0.0f,
                    false, false, false, device);
            register_module("gate_shared", m);
            this.gateShared = m;
        } else {
            this.gateShared = null;
        }
    }

    public CGC(
            int curLevel,
            int nLevel,
            int nTask,
            int nExpertSpecific,
            int nExpertShared,
            int inputDims,
            Map<String, Object> expertParams) {
        this(curLevel, nLevel, nTask, nExpertSpecific, nExpertShared, inputDims, expertParams, DeviceSupport.backend());
    }

    /** Forward: xList has one tensor per task + shared at the tail. */
    public List<Tensor> forward(List<Tensor> xList) {
        List<Tensor> expertSpecificOuts = new ArrayList<>();
        for (int i = 0; i < nTask * nExpertSpecific; i++) {
            int taskIdx = i / nExpertSpecific;
            expertSpecificOuts.add(expertsSpecific.get(i).forward(xList.get(taskIdx)).unsqueeze(1));
        }

        List<Tensor> expertSharedOuts = new ArrayList<>();
        Tensor sharedInput = xList.get(xList.size() - 1);
        for (int s = 0; s < nExpertShared; s++) {
            expertSharedOuts.add(expertsShared.get(s).forward(sharedInput).unsqueeze(1));
        }

        List<Tensor> gateSpecificOuts = new ArrayList<>();
        for (int g = 0; g < nTask; g++) {
            gateSpecificOuts.add(gatesSpecific.get(g).forward(xList.get(g)).unsqueeze(-1));
        }

        List<Tensor> cgcOuts = new ArrayList<>();
        for (int ti = 0; ti < nTask; ti++) {
            List<Tensor> allExpertsForTask = new ArrayList<>();
            for (int j = ti * nExpertSpecific; j < (ti + 1) * nExpertSpecific; j++) {
                allExpertsForTask.add(expertSpecificOuts.get(j));
            }
            allExpertsForTask.addAll(expertSharedOuts);

            TensorVector expertVec = new TensorVector(allExpertsForTask.toArray(new Tensor[0]));
            Tensor expertConcat = torch.cat(expertVec, 1L);
            Tensor expertWeight = torch.mul(gateSpecificOuts.get(ti), expertConcat);
            cgcOuts.add(expertWeight.sum(1L));
        }

        if (hasSharedGate && gateShared != null) {
            List<Tensor> allExpertOuts = new ArrayList<>();
            allExpertOuts.addAll(expertSpecificOuts);
            allExpertOuts.addAll(expertSharedOuts);
            TensorVector sharedVec = new TensorVector(allExpertOuts.toArray(new Tensor[0]));
            Tensor expertConcatShared = torch.cat(sharedVec, 1L);
            Tensor gateSharedOut = gateShared.forward(sharedInput).unsqueeze(-1);
            Tensor expertWeightShared = torch.mul(gateSharedOut, expertConcatShared);
            cgcOuts.add(expertWeightShared.sum(1L));
        }

        return cgcOuts;
    }

    private static long[] toLongArray(List<Long> list) {
        long[] arr = new long[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
