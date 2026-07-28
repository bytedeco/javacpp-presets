package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * Gaussian Mixture Model convolution / MoNet (Monti et al., CVPR 2017).
 *
 * <pre>
 *   w_k(u) = exp( -½ Σ_d ((u_d − μ_{k,d}) / σ_{k,d})² )
 *   m_{ij} = Σ_k w_k(u_{ij}) · (W x_j)_k
 *   x'_i   = Σ_j m_{ij}  (+ W_root x_i + b)
 * </pre>
 * Pseudo-coordinates {@code u = edge_attr ∈ R^{E×dim}} must be supplied.
 * Uses industrial {@link MessagePassing} (sum aggregation); no intermediate closes.
 */
public class GMMConv extends MessagePassing {

    private final LinearImpl lin;       // in → K * out
    private final LinearImpl linRoot;   // optional root
    private final Parameter mu;         // [K, dim]
    private final Parameter sigma;      // [K, dim] (positive via abs in message)
    private final Parameter bias;
    private final long inChannels;
    private final long outChannels;
    private final int dim;
    private final int kernelSize;       // K

    /** Transient edge pseudo-coordinates for the active forward. */
    private Tensor _pseudo;

    public GMMConv(long inChannels, long outChannels, int dim, int kernelSize) {
        this(inChannels, outChannels, dim, kernelSize, true, true);
    }

    public GMMConv(long inChannels, long outChannels, int dim, int kernelSize,
                   boolean rootWeight, boolean hasBias) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || dim <= 0 || kernelSize <= 0) {
            throw new IllegalArgumentException("GMMConv dims must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.dim = dim;
        this.kernelSize = kernelSize;

        TensorOptions fOpt = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float));

        LinearOptions linOpt = new LinearOptions(inChannels, (long) kernelSize * outChannels);
        this.lin = register_module("lin", new LinearImpl(linOpt));

        // Learnable Gaussian parameters — own storage, register for bookkeeping
        Tensor muInit = torch.rand(new long[]{kernelSize, dim}, fOpt).clone();
        torch.xavier_uniform_(muInit);
        muInit.requires_grad_(true);
        this.mu = new Parameter(muInit, true);
        register_parameter("mu", this.mu);

        Tensor sigmaInit = torch.ones(new long[]{kernelSize, dim}, fOpt)
                .mul(new Scalar(0.1)).clone();
        sigmaInit.requires_grad_(true);
        this.sigma = new Parameter(sigmaInit, true);
        register_parameter("sigma", this.sigma);

        if (rootWeight) {
            this.linRoot = register_module("lin_root", new LinearImpl(inChannels, outChannels));
        } else {
            this.linRoot = null;
        }

        if (hasBias) {
            Tensor b = torch.zeros(new long[]{outChannels}, fOpt).clone();
            b.requires_grad_(true);
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        throw new UnsupportedOperationException(
                "GMMConv requires pseudo-coordinates — use forward(x, edge_index, edge_attr)");
    }

    /**
     * @param x          [N, inChannels]
     * @param edge_index [2, E]
     * @param edge_attr  pseudo-coordinates [E, dim] (typically in [0,1])
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (edge_attr == null) {
            throw new IllegalArgumentException("GMMConv requires edge_attr (pseudo-coordinates)");
        }
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x must be [N," + inChannels + "], got F=" + (x.dim() > 1 ? x.size(1) : -1));
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }
        if (edge_attr.dim() != 2 || edge_attr.size(1) != dim) {
            throw new IllegalArgumentException("edge_attr must be [E," + dim + "]");
        }
        if (edge_attr.size(0) != edge_index.size(1)) {
            throw new IllegalArgumentException("edge_attr.size(0) must equal num edges");
        }

        long N = x.size(0);
        // Project once: [N, K*out] → [N, K, out]
        Tensor xLin = lin.forward(x).view(N, kernelSize, outChannels);

        this._pseudo = edge_attr;
        Tensor out;
        try {
            // Message uses _pseudo + multi-kernel x_j; base sum-aggregates
            out = propagate(edge_index, xLin);
        } finally {
            this._pseudo = null;
        }

        if (linRoot != null) {
            out = out.add(linRoot.forward(x));
        }
        if (bias != null) {
            out = out.add(bias);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: [E, K, out]  (lifted from xLin)
        // Gaussian weights from pseudo-coordinates (prefer transient; fall back to edge_attr)
        Tensor u = _pseudo != null ? _pseudo : edge_attr;
        if (u == null) {
            throw new IllegalStateException("GMMConv message requires pseudo-coordinates");
        }

        // σ > 0 for numerical stability
        Tensor sigmaPos = sigma.abs().add(new Scalar(1e-16));           // [K, dim]
        Tensor diff = u.unsqueeze(1).sub(mu.unsqueeze(0));              // [E, K, dim]
        // exp(-0.5 * Σ_d (diff/σ)²)
        Tensor gaussian = diff.pow(new Scalar(2))
                .div(sigmaPos.unsqueeze(0).pow(new Scalar(2)))
                .sum(-1)
                .mul(new Scalar(-0.5))
                .exp();                                                 // [E, K]

        // Σ_k w_k · (W x_j)_k
        return x_j.mul(gaussian.unsqueeze(-1)).sum(1);                  // [E, out]
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

    public Parameter getMu() {
        return mu;
    }

    public Parameter getSigma() {
        return sigma;
    }
}
