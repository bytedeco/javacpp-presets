/*
 * Phase 0 smoke: DeviceSupport, TensorHelpers, Feature types, Recommend facade.
 */
package samples.recommend;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.Recommend;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.features.DenseFeature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;

import java.util.Arrays;
import java.util.List;

public class SmokeFoundation {

    public static void main(String[] args) {
        Recommend.loadNative();
        System.out.println("==========================================");
        System.out.println("  Recommend Phase-0 Foundation Smoke");
        System.out.println("==========================================");

        int pass = 0;
        int fail = 0;

        // DeviceSupport
        try {
            String backend = DeviceSupport.backend();
            System.out.println("[INFO] backend=" + backend
                    + " cuda=" + DeviceSupport.cudaAvailable()
                    + " mps=" + DeviceSupport.mpsAvailable());
            if (backend == null || backend.isEmpty()) {
                throw new IllegalStateException("empty backend");
            }
            System.out.println("[PASS] DeviceSupport.backend");
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] DeviceSupport.backend: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }

        // TensorHelpers construction
        try {
            Tensor z = TensorHelpers.zeros(2, 3);
            Tensor r = TensorHelpers.randn(4, 8);
            Tensor lt = TensorHelpers.longTensorDirect(new long[]{1L, 2L, 3L});
            float[] fa = TensorHelpers.toFloatArray(z);
            if (fa.length != 6) {
                throw new IllegalStateException("zeros numel expected 6 got " + fa.length);
            }
            if (lt.numel() != 3) {
                throw new IllegalStateException("longTensor numel expected 3");
            }
            System.out.println("[PASS] TensorHelpers zeros/randn/longTensor shapes="
                    + Arrays.toString(z.shape()) + " / " + Arrays.toString(r.shape()));
            pass++;
            z.close();
            r.close();
            lt.close();
        } catch (Throwable t) {
            System.out.println("[FAIL] TensorHelpers: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }

        // Features
        try {
            SparseFeature u = Features.sparse("user_id", 1000, 8);
            DenseFeature age = Features.dense("age", 1);
            SequenceFeature hist = Features.sequence("item_hist", 5000, 8, "mean");
            List<Feature> feats =
                    Arrays.asList(u, age, hist);
            long sparseDim = Features.calcSparseDim(feats);
            if (sparseDim != 8) {
                throw new IllegalStateException("calcSparseDim expected 8 got " + sparseDim);
            }
            if (!hist.isSequence()) {
                throw new IllegalStateException("SequenceFeature.isSequence should be true");
            }
            System.out.println("[PASS] Features sparse/dense/sequence sparseDim=" + sparseDim);
            pass++;
        } catch (Throwable t) {
            System.out.println("[FAIL] Features: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }

        // Recommend facade
        try {
            Tensor t = Recommend.randn(2, 2);
            Tensor s = Recommend.sigmoid(t);
            float v = Recommend.toFloat(s.mean());
            if (Float.isNaN(v)) {
                throw new IllegalStateException("sigmoid mean is NaN");
            }
            System.out.println("[PASS] Recommend facade version=" + Recommend.version()
                    + " sigmoid_mean=" + v);
            pass++;
            t.close();
            s.close();
        } catch (Throwable t) {
            System.out.println("[FAIL] Recommend facade: " + t.getMessage());
            t.printStackTrace();
            fail++;
        }

        System.out.println("------------------------------------------");
        System.out.println("Summary: PASS=" + pass + " FAIL=" + fail);
        if (fail > 0) {
            System.exit(1);
        }
    }
}
