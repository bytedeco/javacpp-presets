package samples.demo.layer;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.FusedGATConv;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;

import static org.junit.Assert.*;

/**
 * FusedGATConv tests:
 * 1. Shape checks (concat / mean multi-head)
 * 2. Numerical accuracy with fixed weights (LeakyReLU disabled)
 * 3. Empty-edge boundary
 */
public class FusedGATConvTest {
    static {
        torch.manual_seed(42L);
    }

    @Test
    public void testForwardShape() {
        long inChannels = 4;
        long outChannels = 2;
        long heads = 2;
        FusedGATConv gatConv = new FusedGATConv(inChannels, outChannels, heads, true, 0.2);

        Tensor x = torch.randn(new long[]{5, 4}, floatOpts());

        Tensor edgeIndex = torch.tensor(new long[]{
                0, 0, 1, 2, 3, 4,   // sources
                1, 2, 3, 0, 4, 1    // destinations
        }, longOpts()).view(2, 6);

        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 5);
        Tensor[] csr = (Tensor[]) graphFormat[0];
        Tensor[] csc = (Tensor[]) graphFormat[1];
        Tensor perm = (Tensor) graphFormat[2];

        Tensor output = gatConv.forward(x, csr, csc, perm);
        long[] outputShape = output.shape();
        assertEquals("output must be 2-D", 2, outputShape.length);
        assertEquals("output nodes match input", 5, outputShape[0]);
        assertEquals("concat=true → heads*outChannels", heads * outChannels, outputShape[1]);

        FusedGATConv gatConvNoConcat = new FusedGATConv(inChannels, outChannels, heads, false, 0.2);
        Tensor outputNoConcat = gatConvNoConcat.forward(x, csr, csc, perm);
        assertEquals("concat=false → outChannels", outChannels, outputNoConcat.shape()[1]);

        System.out.println("testForwardShape output (concat=false):");
        torch.print(outputNoConcat);
    }

    @Test
    public void testNumericalAccuracy() {
        // 1 head, concat=false, negativeSlope=-1 disables LeakyReLU
        long inChannels = 2;
        long outChannels = 1;
        long heads = 1;
        FusedGATConv gatConv = new FusedGATConv(inChannels, outChannels, heads, false, -1.0);

        try (NoGradGuard guard = new NoGradGuard()) {
            // W = [[1, 0]] → xLin = first feature column
            Tensor linWeight = torch.tensor(new float[]{1.0f, 0.0f}, floatOpts()).view(1, 2);
            gatConv.lin.weight().copy_(linWeight);
            gatConv.lin.bias().copy_(torch.zeros(new long[]{1}, floatOpts()));

            // attSrc = attDst = [[[1.0]]]
            gatConv.attSrc.copy_(torch.tensor(new float[]{1.0f}, floatOpts()).view(1, 1, 1));
            gatConv.attDst.copy_(torch.tensor(new float[]{1.0f}, floatOpts()).view(1, 1, 1));
        }

        // Nodes: xLin = [1, 2, 3]
        Tensor x = torch.tensor(new float[]{
                1.0f, 2.0f,
                2.0f, 3.0f,
                3.0f, 4.0f
        }, floatOpts()).view(3, 2);

        // Edges 0→1, 0→2
        Tensor edgeIndex = torch.tensor(new long[]{0, 0, 1, 2}, longOpts()).view(2, 2);

        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 3);
        Tensor[] csr = (Tensor[]) graphFormat[0];
        Tensor[] csc = (Tensor[]) graphFormat[1];
        Tensor perm = (Tensor) graphFormat[2];

        Tensor output = gatConv.forward(x, csr, csc, perm);

        // Manual:
        // e(0→1) = 1+2 = 3, e(0→2) = 1+3 = 4
        // α1 = exp(3)/(exp(3)+exp(4)) ≈ 0.2689414  (but softmax is per-destination!)
        // Wait — each destination has only ONE incoming edge here, so softmax → 1.0
        // Node1: only edge 0→1 → α=1 → out=1.0 * xLin[0] = 1.0
        // Node2: only edge 0→2 → α=1 → out=1.0 * xLin[0] = 1.0
        // Node0: no in-edges → 0
        //
        // NOTE: the previous test expected softmax across ALL edges (wrong for GAT).
        // Correct GAT softmax is per destination node.
        System.out.println("testNumericalAccuracy output:");
        torch.print(output);

        float[] out = toFloat1d(output);
        assertEquals("node0 (no in-edges)", 0.0f, out[0], 1e-5f);
        assertEquals("node1 single in-edge → msg=xLin[0]=1", 1.0f, out[1], 1e-5f);
        assertEquals("node2 single in-edge → msg=xLin[0]=1", 1.0f, out[2], 1e-5f);
    }

    @Test
    public void testNumericalAccuracyMultiInEdges() {
        // Same setup, but two edges into node 1 so softmax is non-trivial.
        // Edges: 0→1, 2→1  (both target node 1)
        FusedGATConv gatConv = new FusedGATConv(2, 1, 1, false, -1.0);

        try (NoGradGuard guard = new NoGradGuard()) {
            gatConv.lin.weight().copy_(torch.tensor(new float[]{1.0f, 0.0f}, floatOpts()).view(1, 2));
            gatConv.lin.bias().copy_(torch.zeros(new long[]{1}, floatOpts()));
            gatConv.attSrc.copy_(torch.tensor(new float[]{1.0f}, floatOpts()).view(1, 1, 1));
            gatConv.attDst.copy_(torch.tensor(new float[]{1.0f}, floatOpts()).view(1, 1, 1));
        }

        // xLin = [1, 2, 3]
        Tensor x = torch.tensor(new float[]{
                1.0f, 0.0f,
                2.0f, 0.0f,
                3.0f, 0.0f
        }, floatOpts()).view(3, 2);

        // 0→1, 2→1
        Tensor edgeIndex = torch.tensor(new long[]{0, 2, 1, 1}, longOpts()).view(2, 2);

        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 3);
        Tensor output = gatConv.forward(x,
                (Tensor[]) graphFormat[0],
                (Tensor[]) graphFormat[1],
                (Tensor) graphFormat[2]);

        // e(0→1) = alphaSrc[0]+alphaDst[1] = 1+2 = 3
        // e(2→1) = alphaSrc[2]+alphaDst[1] = 3+2 = 5
        // α0 = exp(3)/(exp(3)+exp(5)), α2 = exp(5)/(exp(3)+exp(5))
        // out[1] = α0 * xLin[0] + α2 * xLin[2] = α0*1 + α2*3
        double e3 = Math.exp(3.0);
        double e5 = Math.exp(5.0);
        double a0 = e3 / (e3 + e5);
        double a2 = e5 / (e3 + e5);
        float expected1 = (float) (a0 * 1.0 + a2 * 3.0);

        System.out.println("testNumericalAccuracyMultiInEdges output:");
        torch.print(output);

        float[] out = toFloat1d(output);
        assertEquals("node0", 0.0f, out[0], 1e-5f);
        assertEquals("node1 attention-weighted", expected1, out[1], 1e-4f);
        assertEquals("node2", 0.0f, out[2], 1e-5f);
    }

    @Test
    public void testEmptyEdges() {
        FusedGATConv gatConv = new FusedGATConv(2, 1, 1, true, 0.2);
        Tensor x = torch.randn(new long[]{3, 2}, floatOpts());
        Tensor edgeIndex = torch.empty(new long[]{2, 0}, longOpts(), new MemoryFormatOptional());

        Object[] graphFormat = FusedGATConv.toGraphFormat(edgeIndex, 3);
        Tensor output = gatConv.forward(x,
                (Tensor[]) graphFormat[0],
                (Tensor[]) graphFormat[1],
                (Tensor) graphFormat[2]);

        assertArrayEquals(new long[]{3, 1}, output.shape());
        float[] out = toFloat1d(output);
        for (float v : out) {
            assertEquals("empty graph → zeros", 0.0f, v, 1e-7f);
        }
    }

    // ---- helpers ----

    private TensorOptions floatOpts() {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
    }

    private TensorOptions longOpts() {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Long))
                .device(new DeviceOptional(new Device(torch.kCPU())));
    }

    private float[] toFloat1d(Tensor t) {
        Tensor flat = t.contiguous().to(new Device(torch.kCPU()), torch.ScalarType.Float).view(-1);
        long n = flat.numel();
        float[] out = new float[(int) n];
        // Prefer bulk copy if available; otherwise element-wise.
        for (int i = 0; i < n; i++) {
            out[i] = flat.select(0, i).item_float();
        }
        return out;
    }

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(FusedGATConvTest.class);
        for (Failure failure : result.getFailures()) {
            System.err.println("FAIL: " + failure.toString());
            if (failure.getException() != null) {
                failure.getException().printStackTrace();
            }
        }
        System.out.println("Tests run=" + result.getRunCount()
                + " failed=" + result.getFailureCount()
                + " ignored=" + result.getIgnoreCount());
        if (!result.wasSuccessful()) {
            throw new RuntimeException("FusedGATConvTest had " + result.getFailureCount() + " failures");
        }
        System.out.println("✅ FusedGATConvTest all tests passed");
    }
}
