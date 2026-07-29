package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.MemoryFormatOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Fused Graph Attention Network convolution operating on CSR/CSC graph formats.
 *
 * <p>Unlike {@link GATConv} (edge_index MessagePassing), this layer takes pre-built
 * CSR {@code [rowptr, col]} and CSC {@code [row, colptr]} so aggregation can run
 * without re-sorting edges each forward. Attention coefficients use the standard
 * additive form:
 * <pre>
 *   e_ij = a_src^T (W h_j) + a_dst^T (W h_i)
 *   α_ij = softmax_i(LeakyReLU(e_ij))
 *   h'_i = Σ_j α_ij W h_j
 * </pre>
 *
 * <p>When {@code negativeSlope < 0}, LeakyReLU is skipped (useful for unit tests).
 */
public class FusedGATConv extends Module {

    public final LinearImpl lin;
    /** Attention source vector [1, heads, outChannels] — retained leaf handle. */
    public final Tensor attSrc;
    /** Attention destination vector [1, heads, outChannels] — retained leaf handle. */
    public final Tensor attDst;

    private final long heads;
    private final long outChannels;
    private final boolean concat;
    private final double negativeSlope;

    public FusedGATConv(long inChannels, long outChannels, long heads,
                        boolean concat, double negativeSlope) {
        super();
        if (inChannels <= 0 || outChannels <= 0 || heads < 1) {
            throw new IllegalArgumentException(
                    "invalid FusedGAT dimensions: in=" + inChannels
                            + " out=" + outChannels + " heads=" + heads);
        }
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.negativeSlope = negativeSlope;

        this.lin = register_module("lin", new LinearImpl(inChannels, heads * outChannels));

        // Keep original leaf handles (register_parameter returns a dangling ByRef).
        Tensor attSrcInit = torch.randn(new long[]{1, heads, outChannels});
        torch.xavier_uniform_(attSrcInit);
        this.attSrc = attSrcInit.clone();
        register_parameter("att_src", this.attSrc);

        Tensor attDstInit = torch.randn(new long[]{1, heads, outChannels});
        torch.xavier_uniform_(attDstInit);
        this.attDst = attDstInit.clone();
        register_parameter("att_dst", this.attDst);
    }

    /**
     * Forward on CSR/CSC.
     *
     * @param x    node features [N, inChannels]
     * @param csr  {@code [rowptr, col]} (CSR; currently unused for aggregation but kept
     *             for API compatibility / future SpMM fused path)
     * @param csc  {@code [row, colptr]} — row is source indices sorted by destination
     * @param perm edge permutation CSR→CSC (kept for API compatibility)
     * @return     [N, heads*out] if concat else [N, out]
     */
    public Tensor forward(Tensor x, Tensor[] csr, Tensor[] csc, Tensor perm) {
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        if (x.dim() != 2) {
            throw new IllegalArgumentException(
                    "node features x must be 2-D, got dim=" + x.dim());
        }
        long N = x.size(0);

        if (csr == null || csr.length != 2 || csc == null || csc.length != 2) {
            throw new IllegalArgumentException(
                    "CSR/CSC must be [rowptr, col] and [row, colptr]");
        }
        Tensor row = AggrUtils.asLongIndex(csc[0]);    // sources sorted by dst [E]
        Tensor colptr = AggrUtils.asLongIndex(csc[1]); // dst offsets [N+1]

        long E = row.size(0);
        if (E == 0) {
            long outDim = concat ? heads * outChannels : outChannels;
            return torch.zeros(new long[]{N, outDim}, x.options());
        }

        // 1. Linear projection → [N, H, C]
        Tensor xLin = lin.forward(x).view(N, heads, outChannels);

        // 2. Per-node attention contributions [N, H]
        Tensor alphaSrc = xLin.mul(attSrc).sum(-1);
        Tensor alphaDst = xLin.mul(attDst).sum(-1);

        // 3. Fused CSC aggregation
        Tensor out = aggregateFused(xLin, alphaSrc, alphaDst, row, colptr, N);

        // 4. Multi-head merge
        if (concat) {
            out = out.view(N, heads * outChannels);
        } else {
            out = out.mean(1);
        }
        return out;
    }

    /**
     * CSC-based attention aggregation.
     * <p>Builds targetIdx from colptr so it is aligned with the CSC-ordered {@code row}.
     */
    private Tensor aggregateFused(Tensor xLin, Tensor alphaSrc, Tensor alphaDst,
                                  Tensor row, Tensor colptr, long numNodes) {
        // targetIdx[e] = destination node of edge e under CSC ordering
        Tensor colDiff = colptr.diff(); // degree per destination [N]
        Tensor targetIdx = torch.arange(new Scalar(numNodes), colDiff.options())
                .repeat_interleave(colDiff);
        targetIdx = AggrUtils.asLongIndex(targetIdx);

        // e_ij = alpha_src[j] + alpha_dst[i]
        Tensor eIj = alphaSrc.index_select(0, row)
                .add(alphaDst.index_select(0, targetIdx));

        if (negativeSlope >= 0) {
            eIj = torch.leaky_relu(eIj, new Scalar(negativeSlope));
        }

        // Softmax over incoming edges of each destination
        Tensor alpha = AggrUtils.scatter_softmax(eIj, targetIdx, numNodes); // [E, H]

        // Weighted messages [E, H, C] → scatter-sum to destinations
        Tensor msg = xLin.index_select(0, row).mul(alpha.unsqueeze(-1));
        return AggrUtils.scatter(msg, targetIdx, numNodes, "sum");
    }

    /**
     * Convert dense edge_index [2, E] into CSR / CSC / perm.
     *
     * @return {@code Object[]{ Tensor[]{rowptr, col}, Tensor[]{row, colptr}, perm }}
     */
    public static Object[] toGraphFormat(Tensor edgeIndex, long numNodes) {
        if (edgeIndex == null) {
            throw new IllegalArgumentException("edge_index must not be null");
        }
        if (edgeIndex.dim() != 2 || edgeIndex.size(0) != 2) {
            throw new IllegalArgumentException(
                    "edge_index must be [2, E], got shape=" + java.util.Arrays.toString(edgeIndex.shape()));
        }
        long E = edgeIndex.size(1);
        TensorOptions longOpts = edgeIndex.options()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long));

        if (E == 0) {
            Tensor rowptr = torch.zeros(new long[]{numNodes + 1}, longOpts);
            Tensor col = torch.empty(new long[]{0}, longOpts, new MemoryFormatOptional());
            Tensor row = torch.empty(new long[]{0}, longOpts, new MemoryFormatOptional());
            Tensor colptr = torch.zeros(new long[]{numNodes + 1}, longOpts);
            Tensor perm = torch.empty(new long[]{0}, longOpts, new MemoryFormatOptional());
            return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
        }

        Tensor srcNodes = AggrUtils.asLongIndex(edgeIndex.select(0, 0));
        Tensor dstNodes = AggrUtils.asLongIndex(edgeIndex.select(0, 1));

        // CSR: sort by source
        Tensor sortedCSR = srcNodes.argsort();
        Tensor col = dstNodes.index_select(0, sortedCSR);
        Tensor srcCount = torch.bincount(srcNodes, new TensorOptional(), numNodes);
        Tensor rowptr = torch.cat(new TensorVector(
                torch.zeros(new long[]{1}, srcCount.options()),
                srcCount.cumsum(0)
        ), 0);

        // CSC: sort by destination
        Tensor sortedCSC = dstNodes.argsort();
        Tensor row = srcNodes.index_select(0, sortedCSC);
        Tensor dstCount = torch.bincount(dstNodes, new TensorOptional(), numNodes);
        Tensor colptr = torch.cat(new TensorVector(
                torch.zeros(new long[]{1}, dstCount.options()),
                dstCount.cumsum(0)
        ), 0);

        // perm maps CSR edge order → CSC edge order
        Tensor perm = sortedCSR.argsort().index_select(0, sortedCSC);

        return new Object[]{new Tensor[]{rowptr, col}, new Tensor[]{row, colptr}, perm};
    }

    public long getHeads() {
        return heads;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public boolean isConcat() {
        return concat;
    }

    public double getNegativeSlope() {
        return negativeSlope;
    }

    public LinearImpl getLin() {
        return lin;
    }

    public Tensor getAttSrc() {
        return attSrc;
    }

    public Tensor getAttDst() {
        return attDst;
    }
}
