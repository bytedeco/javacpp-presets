/*
 * DelayedFeedbackDemo — end-to-end e-commerce delayed-conversion demo.
 *
 * Combines:
 *   - ESCM2 entire-space CVR (Alibaba SIGIR'22 / ESMM lineage)
 *   - DelayedFeedbackHead (Chapelle KDD'14 DFM)
 *   - synthetic click / conversion / elapsed-hours batches
 *
 * Illustrates the industrial pattern: train CTCVR on the full impression space
 * while correcting right-censored conversions that arrive hours after click.
 *
 * Run:
 *   java org.bytedeco.pytorch.utils.recommend.benchmarks.DelayedFeedbackDemo
 */
package org.bytedeco.pytorch.recommend.benchmarks;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.models.ecommerce.ESCM2;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DelayedFeedbackDemo {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private DelayedFeedbackDemo() {}

    public static void main(String[] args) {
        DeviceSupport.setDevice(DeviceSupport.DeviceType.CPU);
        final String device = "cpu";
        final int B = 64;
        final int steps = 8;

        List<Feature> feats = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            feats.add(new SparseFeature("f" + i, 500, 8));
        }

        // ESCM2 with domain adapter + delayed feedback head enabled
        ESCM2 model = new ESCM2(feats, new long[]{64L, 32L}, 3, true, true, device);
        Optimizer opt = new Adam(model.parameters(), new AdamOptions(1e-3));

        System.out.println("DelayedFeedbackDemo: ESCM2 + DFM on synthetic funnel data");
        float first = Float.NaN, last = Float.NaN;
        for (int step = 0; step < steps; step++) {
            Map<String, Tensor> x = new LinkedHashMap<>();
            for (Feature f : feats) {
                x.put(f.name(), torch.randint(500, new long[]{B},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))));
            }
            Tensor domain = torch.randint(3, new long[]{B},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            // ~30% click, ~20% of clicks convert; elapsed hours in (0.1, 48]
            Tensor click = torch.randint(2, new long[]{B},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .mul(torch.rand(new long[]{B}).lt(new Scalar(0.3f)).toType(ScalarType.Float));
            // force some clicks for stable gradients
            click = torch.randint(2, new long[]{B},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            Tensor conv = click.mul(torch.randint(2, new long[]{B},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))));
            Tensor elapsed = torch.rand(new long[]{B}).mul(new Scalar(48f)).add(new Scalar(0.5f));

            opt.zero_grad();
            Tensor preds = model.forward(x, domain);
            Tensor h = model.backboneFeatures(x, domain);
            Tensor loss = model.computeLoss(preds, click, conv, h, elapsed, 0.2f);
            loss.backward();
            opt.step();

            float v = loss.item().toFloat();
            if (step == 0) first = v;
            last = v;
            Tensor pCtcvr = preds.select(1L, ESCM2.COL_CTCVR).mean();
            System.out.printf("  step %d loss=%.6f mean_ctcvr=%.4f%n",
                    step, v, pCtcvr.item().toFloat());
        }
        if (Float.isNaN(first) || Float.isInfinite(last)) {
            throw new IllegalStateException("non-finite loss");
        }
        System.out.printf("DelayedFeedbackDemo PASS  first=%.6f last=%.6f%n", first, last);
    }
}
