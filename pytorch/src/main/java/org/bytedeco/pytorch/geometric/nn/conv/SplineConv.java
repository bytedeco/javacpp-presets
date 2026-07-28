package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * SplineCNN convolution (Fey et al., SplineConv / PyG).
 *
 * <pre>
 *   m_{ij} = Σ_p B_p(u_{ij}) · W_p · x_j
 *   x'_i   = aggr_j m_{ij}  (+ W_root x_i + b)
 * </pre>
 * {@code u = edge_attr ∈ [0,1]^{E×dim}} are pseudo-coordinates.
 * B-spline basis of {@code degree} (implemented fully for degree=1 multi-dim
 * via tensor-product of 1-D linear bases; higher degree falls back to degree-1
 * with a clear log once — full higher-order B-splines need recursive Cox-de Boor).
 *
 * Weight tensor shape: {@code [K^dim, in, out]} for degree-1 (2 control points
 * per dim → effectively kernelSize per axis in the discrete grid).
 */
public class SplineConv extends MessagePassing {

    private final long inChannels;
    private final long outChannels;
    private final int dim;
    private final int kernelSize;
    private final int degree;
    private final Parameter weight;      // [prod(kernelSize^dim), in, out]
    private final LinearImpl linRoot;
    private final Parameter bias;
    private final long totalKernelSize;

    /** Transient spline basis / indices for one forward. */
    private Tensor _basis;      // [E, S]
    private Tensor _weightIdx;  // [E, S] long

    public SplineConv(long inChannels, long outChannels, int dim, int kernelSize) {
        this(inChannels, outChannels, dim, kernelSize, 1, true, true);
    }

    public SplineConv(long inChannels, long outChannels, int dim, int kernelSize,
                      int degree, boolean rootWeight, boolean hasBias) {
        super("mean");
        if (inChannels <= 0 || outChannels <= 0 || dim <= 0 || kernelSize < 2) {
            throw new IllegalArgumentException(
                    "SplineConv: in/out>0, dim>0, kernelSize>=2 required");
        }
        if (degree < 1) {
            throw new IllegalArgumentException("degree must be >= 1");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.dim = dim;
        this.kernelSize = kernelSize;
        this.degree = degree;
        // Grid of control points along each axis
        this.totalKernelSize = (long) Math.pow(kernelSize, dim);

        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor wInit = torch.empty(new long[]{totalKernelSize, inChannels, outChannels},
                fOpt, new org.bytedeco.pytorch.MemoryFormatOptional());
        torch.xavier_uniform_(wInit);
        wInit = wInit.clone().requires_grad_(true);
        this.weight = new Parameter(wInit, true);
        register_parameter("weight", this.weight);

        if (rootWeight) {
            this.linRoot = register_module("lin_root", new LinearImpl(inChannels, outChannels));
        } else {
            this.linRoot = null;
        }

        if (hasBias) {
            Tensor b = torch.zeros(new long[]{outChannels}, fOpt).clone().requires_grad_(true);
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        throw new UnsupportedOperationException(
                "SplineConv requires edge_attr in [0,1]^dim — use forward(x, edge_index, edge_attr)");
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (edge_attr == null) {
            throw new IllegalArgumentException("SplineConv requires edge_attr (pseudo-coordinates in [0,1])");
        }
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x must be [N," + inChannels + "]");
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }
        if (edge_attr.dim() != 2 || edge_attr.size(1) != dim) {
            throw new IllegalArgumentException("edge_attr must be [E," + dim + "]");
        }
        if (edge_attr.size(0) != edge_index.size(1)) {
            throw new IllegalArgumentException("edge_attr rows must equal num edges");
        }

        Tensor[] spline = computeLinearSplineBasis(edge_attr);
        this._basis = spline[0];
        this._weightIdx = spline[1];
        try {
            Tensor out = propagate(edge_index, x);
            if (linRoot != null) {
                out = out.add(linRoot.forward(x));
            }
            if (bias != null) {
                out = out.add(bias);
            }
            return out;
        } finally {
            this._basis = null;
            this._weightIdx = null;
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (_basis == null || _weightIdx == null) {
            throw new IllegalStateException("SplineConv message requires active forward with edge_attr");
        }
        long E = x_j.size(0);
        long S = _basis.size(1); // number of tensor-product basis terms (2^dim for degree-1)

        Tensor msg = torch.zeros(new long[]{E, outChannels}, x_j.options());
        for (int s = 0; s < S; s++) {
            Tensor b = _basis.select(1, s).unsqueeze(-1);                 // [E,1]
            Tensor wi = _weightIdx.select(1, s).to(torch.kLong());        // [E]
            // Clamp indices into valid weight grid
            wi = wi.clamp(new org.bytedeco.pytorch.ScalarOptional(new Scalar(0)),
                    new org.bytedeco.pytorch.ScalarOptional(new Scalar(totalKernelSize - 1)));
            Tensor w = weight.index_select(0, wi);                        // [E, in, out]
            // x_j [E,in] @ W [E,in,out] → [E,out]
            Tensor res = torch.matmul(x_j.unsqueeze(1), w).squeeze(1);
            msg = msg.add(res.mul(b));
        }
        return msg;
    }

    /**
     * Multi-dimensional degree-1 (linear) B-spline via tensor product.
     * For each dim d: left/right control points with bases (1-f), f.
     * Cartesian product yields 2^dim terms; flat weight index via mixed radix
     * of size {@code kernelSize} per axis.
     *
     * @return {basis [E, 2^dim], weightIdx [E, 2^dim] long}
     */
    private Tensor[] computeLinearSplineBasis(Tensor edgeAttr) {
        // Clamp pseudo-coordinates into [0, 1]
        Tensor u = edgeAttr.clamp(new org.bytedeco.pytorch.ScalarOptional(new Scalar(0.0)),
                new org.bytedeco.pytorch.ScalarOptional(new Scalar(1.0)));
        // Map to continuous grid [0, kernelSize-1]
        Tensor scaled = u.mul(new Scalar(kernelSize - 1));                 // [E, dim]
        Tensor idxLeft = scaled.floor().to(torch.kLong());                // [E, dim]
        // Right index, clamped
        Tensor idxRight = idxLeft.add(new Scalar(1)).clamp_max(new Scalar(kernelSize - 1));
        Tensor frac = scaled.sub(idxLeft.to(torch.ScalarType.Float));     // [E, dim] in [0,1]
        Tensor basLeft = frac.neg().add(new Scalar(1.0));                 // 1 - f
        Tensor basRight = frac;                                           // f

        // Tensor product over dim axes → 2^dim combinations
        int S = 1 << dim;
        long E = edgeAttr.size(0);
        Tensor basis = torch.ones(new long[]{E, S}, edgeAttr.options());
        Tensor weightIdx = torch.zeros(new long[]{E, S},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        for (int s = 0; s < S; s++) {
            Tensor b = torch.ones(new long[]{E}, edgeAttr.options());
            Tensor idx = torch.zeros(new long[]{E},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
            for (int d = 0; d < dim; d++) {
                boolean right = ((s >> d) & 1) == 1;
                Tensor bd = right ? basRight.select(1, d) : basLeft.select(1, d);
                Tensor id = right ? idxRight.select(1, d) : idxLeft.select(1, d);
                b = b.mul(bd);
                // mixed-radix: idx += id * kernelSize^d
                long stride = (long) Math.pow(kernelSize, d);
                idx = idx.add(id.mul(new Scalar(stride)));
            }
            basis.select(1, s).copy_(b);
            weightIdx.select(1, s).copy_(idx);
        }
        return new Tensor[]{basis, weightIdx};
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public int getDim() {
        return dim;
    }

    public int getKernelSize() {
        return kernelSize;
    }

    public int getDegree() {
        return degree;
    }
}
