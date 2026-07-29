package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * Crystal Graph Convolution (CGConv, Xie & Grossman / CGCNN).
 *
 * <pre>
 *   z_{ij} = [ x_i || x_j || e_{ij} ]
 *   m_{ij} = σ(W_s z_{ij}) ⊙ softplus(W_f z_{ij})
 *   x'_i   = x_i + aggr_j m_{ij}  (+ b)  [, BatchNorm]
 * </pre>
 * Requires {@link #needsX_i()} so both endpoints are lifted. Edge features optional.
 */
public class CGConv extends MessagePassing {

    private final LinearImpl linF;
    private final LinearImpl linS;
    private final BatchNorm1dImpl bn;
    private final Parameter bias;
    private final long channels;
    private final int edgeDim;
    private final long zDim; // 2*channels + edgeDim

    public CGConv(long channels) {
        this(channels, 0, "sum", false, true);
    }

    public CGConv(long channels, int edgeDim, String aggr, boolean batchNorm, boolean hasBias) {
        super(normalizeAggr(aggr));
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be > 0");
        }
        if (edgeDim < 0) {
            throw new IllegalArgumentException("edgeDim must be >= 0");
        }
        this.channels = channels;
        this.edgeDim = edgeDim;
        this.zDim = 2 * channels + edgeDim;

        LinearOptions fOpt = new LinearOptions(zDim, channels);
        fOpt.bias().put(true);
        LinearOptions sOpt = new LinearOptions(zDim, channels);
        sOpt.bias().put(true);
        this.linF = register_module("lin_f", new LinearImpl(fOpt));
        this.linS = register_module("lin_s", new LinearImpl(sOpt));

        if (batchNorm) {
            BatchNormOptions bnOptions = new BatchNormOptions(channels);
            this.bn = register_module("bn", new BatchNorm1dImpl(bnOptions));
        } else {
            this.bn = null;
        }

        if (hasBias) {
            Tensor b = torch.zeros(new long[]{channels},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    private static String normalizeAggr(String aggr) {
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
            default:
                throw new IllegalArgumentException(
                        "CGConv aggr must be sum/add/mean/max, got: " + aggr);
        }
    }

    @Override
    protected boolean needsX_i() {
        return true;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new IllegalArgumentException("x and edge_index must not be null");
        }
        if (x.dim() != 2 || x.size(1) != channels) {
            throw new IllegalArgumentException(
                    "x must be [N," + channels + "], got F=" + (x.dim() > 1 ? x.size(1) : -1));
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }
        if (edgeDim > 0) {
            if (edge_attr == null) {
                throw new IllegalArgumentException("edge_attr required when edgeDim=" + edgeDim);
            }
            if (edge_attr.dim() != 2 || edge_attr.size(1) != edgeDim) {
                throw new IllegalArgumentException(
                        "edge_attr must be [E," + edgeDim + "]");
            }
            if (edge_attr.size(0) != edge_index.size(1)) {
                throw new IllegalArgumentException(
                        "edge_attr.size(0) must equal num edges");
            }
        }

        Tensor aggr = propagate(edge_index, x, edge_attr);
        Tensor out = aggr;
        if (bias != null) {
            out = out.add(bias);
        }
        // Residual
        out = x.add(out);
        if (bn != null) {
            out = bn.forward(out);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        Tensor z;
        if (edge_attr != null && edgeDim > 0) {
            z = torch.cat(new TensorVector(x_i, x_j, edge_attr), -1);
        } else {
            z = torch.cat(new TensorVector(x_i, x_j), -1);
        }
        if (z.size(1) != zDim) {
            throw new IllegalStateException(
                    "CGConv z dim " + z.size(1) + " != expected " + zDim);
        }
        Tensor g = torch.sigmoid(linS.forward(z));
        Tensor f = torch.softplus(linF.forward(z));
        return g.mul(f);
    }

    public long getChannels() {
        return channels;
    }

    public int getEdgeDim() {
        return edgeDim;
    }
}
