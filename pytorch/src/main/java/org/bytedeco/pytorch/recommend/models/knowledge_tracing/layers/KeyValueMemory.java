/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Memory.scala
 *
 * Key-Value Memory for DKVMN and DeepIRT.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class KeyValueMemory extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numSlots;
    private final int keyDim;
    private final int valueDim;
    private final Tensor keyMemory;
    private final Tensor valueMemoryInit;

    public KeyValueMemory(int numSlots, int keyDim, int valueDim) {
        this(numSlots, keyDim, valueDim, DeviceSupport.backend());
    }

    public KeyValueMemory(int numSlots, int keyDim, int valueDim, String device) {
        super("KeyValueMemory");
        this.numSlots = numSlots;
        this.keyDim = keyDim;
        this.valueDim = valueDim;

        // Key memory matrix: (numSlots, keyDim)
        Tensor km = torch.randn(
                new long[]{numSlots, keyDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar((float) Math.sqrt(1.0 / keyDim)));
        this.keyMemory = km;
        register_parameter("key_memory", keyMemory);

        // Value memory init: (numSlots, valueDim)
        Tensor vm = torch.zeros(
                new long[]{numSlots, valueDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.valueMemoryInit = vm;
        register_parameter("value_memory_init", valueMemoryInit);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            this.to(dev, false);
        }
    }

    /**
     * Compute attention weights over memory slots.
     *
     * @param key (batch, seq, keyDim)
     * @return Attention weights (batch, seq, numSlots)
     */
    public Tensor computeAttention(Tensor key) {
        Tensor kmT = keyMemory.view(1L, 1L, (long) numSlots, (long) keyDim);
        Tensor key4d = key.unsqueeze(2); // (batch, seq, 1, keyDim)
        // (batch, seq, numSlots)
        Tensor scores = torch.matmul(key4d, kmT.transpose(2, 3)).squeeze(3);
        return scores.softmax(2);
    }

    /**
     * Read from memory using attention weights.
     *
     * @param contentKey content key (batch, seq, keyDim) — unused in formula, kept for API parity
     * @param attn       (batch, seq, numSlots)
     * @param valueMem   (batch, numSlots, valueDim)
     * @return (batch, seq, valueDim)
     */
    public Tensor read(Tensor contentKey, Tensor attn, Tensor valueMem) {
        Tensor attnEx = attn.unsqueeze(3); // (batch, seq, numSlots, 1)
        return valueMem.unsqueeze(0).mul(attnEx).sum(2); // (batch, seq, valueDim)
    }

    /**
     * Write to memory using attention weights.
     *
     * @param contentValue (batch, seq, valueDim)
     * @param attn         (batch, seq, numSlots)
     * @param currentMem   (batch, numSlots, valueDim)
     * @param eraseVector  (batch, seq, valueDim)
     * @return updated value memory (batch, numSlots, valueDim)
     */
    public Tensor write(Tensor contentValue, Tensor attn, Tensor currentMem, Tensor eraseVector) {
        Tensor attnMean = attn.mean(1).unsqueeze(2); // (batch, numSlots, 1)
        Tensor eraseGate = torch.sigmoid(eraseVector.mean(1)).unsqueeze(1); // (batch, 1, valueDim)
        Tensor erased = currentMem.mul(attnMean.mul(eraseGate).neg().add(new Scalar(1.0)));
        Tensor addValue = attnMean.mul(contentValue.mean(1).unsqueeze(1)); // (batch, numSlots, valueDim)
        return erased.add(addValue);
    }

    public Tensor getInitialMemory() {
        return valueMemoryInit;
    }

    public Tensor getKeyMemory() {
        return keyMemory;
    }
}
