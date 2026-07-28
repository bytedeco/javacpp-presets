package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.relu;

/**
 * GENeralized Graph Convolution (Li et al., GENConv / DeepGCN).
 *
 * <pre>
 *   m_{ij} = ReLU(x_j + W_e e_{ij}) + ε
 *   x'_i   = W · Aggr({m_{ij}})
 * </pre>
 * Aggr ∈ {softmax, powermean, sum, mean, max}. Softmax/PowerMean use learnable
 * temperature {@code t} / exponent {@code p}.
 */
public class GENConv extends MessagePassing {

    private final LinearImpl lin;
    private final LinearImpl linEdge;
    private final Parameter t;
    private final Parameter p;
    private final float eps;
    private final String aggrType;
    private final long inChannels;
    private final long outChannels;

    public GENConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, "softmax", 1.0f, false, 1.0f, false, null, 1e-7f, true);
    }

    public GENConv(long inChannels, long outChannels, String aggr, float tVal, boolean learnT,
                   float pVal, boolean learnP, Integer edgeDim, float eps, boolean hasBias) {
        // Base string aggr only used for sum/mean/max; softmax/powermean override aggregate()
        super(toBaseAggr(aggr));
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.aggrType = aggr == null ? "softmax" : aggr;
        this.eps = eps;

        LinearOptions linOpt = new LinearOptions(inChannels, outChannels);
        linOpt.bias().put(hasBias);
        this.lin = register_module("lin", new LinearImpl(linOpt));

        if (edgeDim != null && edgeDim > 0) {
            LinearOptions eOpt = new LinearOptions(edgeDim, inChannels);
            eOpt.bias().put(hasBias);
            this.linEdge = register_module("lin_edge", new LinearImpl(eOpt));
        } else {
            this.linEdge = null;
        }

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor tInit = torch.tensor(new float[]{tVal}, fOpt).clone();
        if (learnT) {
            tInit.requires_grad_(true);
        }
        this.t = new Parameter(tInit, learnT);
        if (learnT) {
            register_parameter("t", this.t);
        } else {
            register_buffer("t", this.t);
        }

        Tensor pInit = torch.tensor(new float[]{pVal}, fOpt).clone();
        if (learnP) {
            pInit.requires_grad_(true);
        }
        this.p = new Parameter(pInit, learnP);
        if (learnP) {
            register_parameter("p", this.p);
        } else {
            register_buffer("p", this.p);
        }
    }

    private static String toBaseAggr(String aggr) {
        if (aggr == null) {
            return "sum";
        }
        switch (aggr) {
            case "add":
            case "sum":
                return "sum";
            case "mean":
                return "mean";
            case "max":
                return "max";
            case "softmax":
            case "powermean":
                return "sum"; // overridden in aggregate()
            default:
                throw new UnsupportedOperationException(
                        "GENConv aggr must be softmax|powermean|sum|mean|max, got: " + aggr);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("x must be [N,F], edge_index [2,E]");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }
        Tensor out = propagate(edge_index, x, edge_attr);
        return lin.forward(out);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        Tensor msg = x_j;
        if (edge_attr != null) {
            Tensor ea = edge_attr;
            if (linEdge != null) {
                ea = linEdge.forward(ea);
            }
            if (ea.dim() == 1) {
                ea = ea.view(new long[]{-1, 1}).expand_as(msg);
            }
            if (ea.size(1) != msg.size(1)) {
                throw new IllegalArgumentException(
                        "edge feature dim " + ea.size(1) + " != node dim " + msg.size(1));
            }
            msg = msg.add(ea);
        }
        return relu(msg).add(new Scalar(eps));
    }

    @Override
    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
        switch (aggrType) {
            case "softmax": {
                // sum_j softmax_j(t · m_j) · m_j
                Tensor tVal = t;
                Tensor scaled = inputs.mul(tVal);
                // stable softmax via AggrUtils
                Tensor alpha = AggrUtils.scatter_softmax(scaled, index, dimSize);
                Tensor weighted = inputs.mul(alpha);
                return AggrUtils.scatter(weighted, index, dimSize, "sum");
            }
            case "powermean": {
                float pFloat = p.item_float();
                if (Math.abs(pFloat) < 1e-8f) {
                    // geometric mean
                    Tensor logInputs = inputs.clamp_min(new Scalar(1e-16)).log();
                    return AggrUtils.scatter(logInputs, index, dimSize, "mean").exp();
                }
                Tensor powInputs = inputs.clamp_min(new Scalar(1e-16)).pow(p);
                Tensor meanPow = AggrUtils.scatter(powInputs, index, dimSize, "mean");
                return meanPow.clamp_min(new Scalar(1e-16)).pow(new Scalar(1.0 / pFloat));
            }
            case "add":
            case "sum":
                return AggrUtils.scatter(inputs, index, dimSize, "sum");
            case "mean":
                return AggrUtils.scatter(inputs, index, dimSize, "mean");
            case "max":
                return AggrUtils.scatter(inputs, index, dimSize, "max");
            default:
                return super.aggregate(inputs, index, dimSize);
        }
    }

    public Parameter getT() {
        return t;
    }

    public Parameter getP() {
        return p;
    }

    public String getAggrType() {
        return aggrType;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
