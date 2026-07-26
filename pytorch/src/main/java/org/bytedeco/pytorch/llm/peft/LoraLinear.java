/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.peft;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import static org.bytedeco.pytorch.global.torch.dropout;
import static org.bytedeco.pytorch.global.torch.mm;
import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * LoRA-augmented linear layer (live {@link Module}, not offline weight maps).
 *
 * <p>Forward:
 * <pre>
 *   y = base(x) + scaling * (dropout(x) @ A^T @ B^T)
 * </pre>
 * where {@code A} is {@code [r, in]}, {@code B} is {@code [out, r]}, matching
 * Hugging Face PEFT layout ({@code lora_A.weight}, {@code lora_B.weight}).
 *
 * <p>{@link #merge()} / {@link #unmerge()} bake or restore
 * {@code ΔW = B @ A * scaling} into the base weight for faster inference.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LoraLinear extends Module {
    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl base;
    private final LoraConfig config;
    private final long inFeatures;
    private final long outFeatures;
    private final double scaling;

    /** Registered as {@code lora_A}, shape {@code [r, in]}. */
    private Tensor loraA;
    /** Registered as {@code lora_B}, shape {@code [out, r]}. */
    private Tensor loraB;

    private boolean merged;
    /** Cached ΔW used by {@link #unmerge()} (detached clone). */
    private Tensor mergedDelta;

    /**
     * Wrap an existing {@link LinearImpl}. Base weights are optionally frozen.
     *
     * <p><b>JavaCPP ownership rules</b> (see memory scalartype-intern / register_parameter):
     * <ul>
     *   <li>{@code register_parameter} returns a {@code @ByRef} view that must NOT be
     *       stored in a Java field — it dangles after the native call returns and
     *       subsequent {@code numel()}/{@code t()} SIGSEGV.</li>
     *   <li>Keep the pre-register Tensor handle (after {@code clone()} so we own storage),
     *       call {@code register_parameter} only for Module bookkeeping / optim discovery.</li>
     *   <li>If {@code base} is already registered under another Module (e.g. CausalLM),
     *       use {@link #borrowBase(LinearImpl, LoraConfig)} to avoid double
     *       {@code register_module}.</li>
     * </ul>
     */
    public LoraLinear(LinearImpl base, LoraConfig config) {
        this(base, config, /*registerBase=*/true);
    }

    /**
     * Wrap {@code base} without {@code register_module("base", base)} — for bases that
     * already live in another Module tree (CausalLM / Qwen / Llama).
     */
    public static LoraLinear borrowBase(LinearImpl base, LoraConfig config) {
        return new LoraLinear(base, config, /*registerBase=*/false);
    }

    private LoraLinear(LinearImpl base, LoraConfig config, boolean registerBase) {
        super("LoraLinear");
        if (base == null) {
            throw new IllegalArgumentException("base must not be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        this.base = registerBase ? register_module("base", base) : base;
        this.config = config;
        this.inFeatures = base.weight().size(1);
        this.outFeatures = base.weight().size(0);
        this.scaling = config.scaling();
        this.merged = false;

        if (config.freezeBase()) {
            base.weight().requires_grad_(false);
            try {
                if (base.bias() != null && !base.bias().isNull() && base.bias().defined()) {
                    base.bias().requires_grad_(false);
                }
            } catch (Exception ignored) {}
        }

        // A ~ N(0, 1/sqrt(r)); B zero so initial ΔW = 0 (PEFT default).
        // clone() so Java owns storage; requires_grad_ before register.
        Tensor aInit = randn(config.r(), inFeatures)
                .div(new Scalar(Math.sqrt(config.r())))
                .contiguous()
                .clone();
        Tensor bInit = zeros(outFeatures, config.r()).contiguous().clone();
        aInit.requires_grad_(true);
        bInit.requires_grad_(true);
        // Register for Module.parameters() discovery — ignore ByRef return.
        register_parameter("lora_A", aInit, true);
        register_parameter("lora_B", bInit, true);
        this.loraA = aInit;
        this.loraB = bInit;
    }

    /** Build a fresh base linear then wrap it. */
    public LoraLinear(long inFeatures, long outFeatures, LoraConfig config) {
        this(new LinearImpl(inFeatures, outFeatures), config);
    }

    public LoraConfig config() {
        return config;
    }

    public LinearImpl base() {
        return base;
    }

    public Tensor loraA() {
        return loraA;
    }

    public Tensor loraB() {
        return loraB;
    }

    public boolean isMerged() {
        return merged;
    }

    public long inFeatures() {
        return inFeatures;
    }

    public long outFeatures() {
        return outFeatures;
    }

    public double scaling() {
        return scaling;
    }

    /**
     * {@code y = base(x) + scale * ((dropout(x) @ A^T) @ B^T)} when not merged;
     * plain {@code base(x)} when merged.
     */
    public Tensor forward(Tensor input) {
        Tensor result = base.forward(input);
        if (merged || config.inferenceMode()) {
            // When merged, base already contains ΔW. inferenceMode still applies
            // dropout=0 path below if not merged.
            if (merged) {
                return result;
            }
        }
        Tensor x = input;
        if (config.dropout() > 0.0 && is_training()) {
            x = dropout(x, config.dropout(), /*train=*/true);
        }
        // x: [*, in] ; A: [r, in] ; B: [out, r]
        // x @ A^T -> [*, r] ; that @ B^T -> [*, out]
        Tensor aT = loraA.t(); // [in, r]
        Tensor bT = loraB.t(); // [r, out]
        Tensor mid = matmulLast(x, aT);
        Tensor delta = matmulLast(mid, bT).mul(new Scalar(scaling));
        return result.add(delta);
    }

    /** Trainable LoRA parameters only (A, B). */
    public TensorVector loraParameters() {
        TensorVector v = new TensorVector();
        v.push_back(loraA);
        v.push_back(loraB);
        return v;
    }

    /**
     * Bake {@code ΔW = B @ A * scaling} into {@code base.weight} (in-place).
     * After merge, {@link #forward} skips the adapter path.
     */
    public void merge() {
        if (merged) {
            return;
        }
        Tensor deltaW = mm(loraB, loraA).mul(new Scalar(scaling)); // [out, in]
        mergedDelta = deltaW.clone().detach();
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            base.weight().add_(deltaW);
        }
        merged = true;
    }

    /** Undo {@link #merge()} if previously merged. */
    public void unmerge() {
        if (!merged) {
            return;
        }
        if (mergedDelta != null && mergedDelta.defined()) {
            try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
                base.weight().sub_(mergedDelta);
            }
            mergedDelta = null;
        }
        merged = false;
    }

    /** ΔW without mutating base: {@code B @ A * scaling}, shape {@code [out, in]}. */
    public Tensor deltaWeight() {
        return mm(loraB, loraA).mul(new Scalar(scaling));
    }

    /**
     * Batched/unbatched matmul on the last dim: treats leading dims as batch.
     * For 2D this is plain {@code mm}; for higher rank uses reshape.
     */
    private static Tensor matmulLast(Tensor a, Tensor b) {
        if (a.dim() == 2 && b.dim() == 2) {
            return mm(a, b);
        }
        // Flatten batch: [B..., in] @ [in, out] -> [B..., out]
        long[] aSizes = a.shape();
        long in = aSizes[aSizes.length - 1];
        long rest = 1;
        for (int i = 0; i < aSizes.length - 1; i++) {
            rest *= aSizes[i];
        }
        Tensor flat = a.reshape(rest, in);
        Tensor out2d = mm(flat, b);
        long out = b.size(1);
        long[] outShape = new long[aSizes.length];
        System.arraycopy(aSizes, 0, outShape, 0, aSizes.length - 1);
        outShape[outShape.length - 1] = out;
        return out2d.reshape(outShape);
    }
}
