/*
 * GraphDrugEncoder — molecular graph encoder for drug SMILES-derived graphs.
 *
 * Replaces pure sequence drug towers in DTI models (DrugBAN / DeepDTA variants)
 * with a GCN stack over atom features + adjacency — the standard GraphDTA /
 * DrugBAN-graph formulation.
 *
 * References:
 *   - Nguyen et al., "GraphDTA", Bioinformatics 2021
 *   - Kipf & Welling, GCN (ICLR 2017) — via ranking.GraphConvolution
 *
 * Input:
 *   atomFeatures [N, F]
 *   adj          [N, N]  (preferably sym-normalized; see normalizeAdj)
 *   batchId      [N] long in [0,B) for mini-batch of graphs (optional)
 * Output: [B, outDim]  (B=1 if no batchId)
 */
package org.bytedeco.pytorch.recommend.models.pharma;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.models.ranking.GraphConvolution;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GraphDrugEncoder extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<GraphConvolution> layers = new ArrayList<>();
    private final LinearImpl outProj;
    private final ReLUImpl relu;
    private final DropoutImpl dropout;
    private final int outDim;
    private final int numLayers;

    public GraphDrugEncoder(int atomFeatDim, int hiddenDim, int outDim) {
        this(atomFeatDim, hiddenDim, outDim, 3, 0.1f, DeviceSupport.backend());
    }

    public GraphDrugEncoder(int atomFeatDim, int hiddenDim, int outDim, int numLayers,
                            float dropoutProb, String device) {
        super("GraphDrugEncoder");
        if (numLayers < 1) throw new IllegalArgumentException("numLayers >= 1");
        this.outDim = outDim;
        this.numLayers = numLayers;
        this.relu = new ReLUImpl();
        this.dropout = new DropoutImpl(dropoutProb);

        for (int i = 0; i < numLayers; i++) {
            int inDim = (i == 0) ? atomFeatDim : hiddenDim;
            GraphConvolution gc = new GraphConvolution(inDim, hiddenDim, device);
            register_module("gc_" + i, gc);
            layers.add(gc);
        }
        this.outProj = new LinearImpl(hiddenDim, outDim);
        register_module("out_proj", outProj);

        if (device != null && !"cpu".equals(device)) {
            outProj.to(new Device(device), false);
        }
    }

    private Tensor encodeNodes(Tensor atomFeatures, Tensor adj) {
        Tensor h = atomFeatures;
        for (int i = 0; i < numLayers; i++) {
            h = layers.get(i).forward(h, adj);
            if (i < numLayers - 1) {
                h = dropout.forward(relu.forward(h));
            } else {
                h = relu.forward(h);
            }
        }
        return h; // [N, H]
    }

    /** Single graph → [1, outDim]. */
    public Tensor forward(Tensor atomFeatures, Tensor adj) {
        Tensor h = encodeNodes(atomFeatures, adj);
        Tensor g = h.mean(0L).unsqueeze(0L);
        return outProj.forward(g);
    }

    /**
     * Mini-batch of graphs via node→graph ids.
     * @param batchId [N] long in [0, B)
     * @return [B, outDim]
     */
    public Tensor forward(Tensor atomFeatures, Tensor adj, Tensor batchId) {
        Tensor h = encodeNodes(atomFeatures, adj);
        Tensor bid = batchId.toType(ScalarType.Long);
        long B = bid.max().item_long() + 1L;
        List<Tensor> reps = new ArrayList<>((int) B);
        for (long b = 0; b < B; b++) {
            Tensor mask = bid.eq(new Scalar(b)).toType(ScalarType.Float); // [N]
            Tensor denom = mask.sum().clamp_min(new Scalar(1.0f));
            Tensor sum = h.mul(mask.unsqueeze(1L)).sum(0L); // [H]
            reps.add(sum.div(denom).unsqueeze(0L));
        }
        TensorVector vec = new TensorVector();
        for (Tensor r : reps) vec.push_back(r);
        return outProj.forward(torch.cat(vec, 0L));
    }

    /**
     * Sym-normalize dense adjacency: D^{-1/2} (A+I) D^{-1/2}.
     */
    public static Tensor normalizeAdj(Tensor adj) {
        long n = adj.size(0);
        Tensor eye = torch.eye(n, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        try {
            eye = eye.to(adj.device(), ScalarType.Float);
        } catch (Throwable ignored) {}
        Tensor aHat = adj.toType(ScalarType.Float).add(eye);
        Tensor deg = aHat.sum(1L).clamp_min(new Scalar(1e-6f));
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5f));
        return aHat.mul(degInvSqrt.unsqueeze(1L)).mul(degInvSqrt.unsqueeze(0L));
    }

    public int outDim() {
        return outDim;
    }
}
