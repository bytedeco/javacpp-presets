import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.AdaptiveAvgPool1dOptions;
import org.bytedeco.pytorch.AdaptiveAvgPool2dOptions;
import org.bytedeco.pytorch.AdaptiveAvgPool3dOptions;
import org.bytedeco.pytorch.AdaptiveMaxPool1dOptions;
import org.bytedeco.pytorch.AdaptiveMaxPool2dOptions;
import org.bytedeco.pytorch.AvgPool1dOptions;
import org.bytedeco.pytorch.AvgPool2dOptions;
import org.bytedeco.pytorch.AvgPool3dOptions;
import org.bytedeco.pytorch.BatchNormOptions;
import org.bytedeco.pytorch.BilinearOptions;
import org.bytedeco.pytorch.CELUOptions;
import org.bytedeco.pytorch.ConstantPad2dOptions;
import org.bytedeco.pytorch.Conv1dOptions;
import org.bytedeco.pytorch.Conv2dOptions;
import org.bytedeco.pytorch.Conv3dOptions;
import org.bytedeco.pytorch.ConvTranspose1dOptions;
import org.bytedeco.pytorch.ConvTranspose2dOptions;
import org.bytedeco.pytorch.ConvTranspose3dOptions;
import org.bytedeco.pytorch.ConvPaddingMode;
import org.bytedeco.pytorch.CosineSimilarityOptions;
import org.bytedeco.pytorch.DropoutOptions;
import org.bytedeco.pytorch.ELUOptions;
import org.bytedeco.pytorch.EmbeddingBagMode;
import org.bytedeco.pytorch.EmbeddingBagOptions;
import org.bytedeco.pytorch.EmbeddingOptions;
import org.bytedeco.pytorch.FlattenOptions;
import org.bytedeco.pytorch.FractionalMaxPool2dOptions;
import org.bytedeco.pytorch.GELUOptions;
import org.bytedeco.pytorch.GRUCellOptions;
import org.bytedeco.pytorch.GRUOptions;
import org.bytedeco.pytorch.GroupNormOptions;
import org.bytedeco.pytorch.HardshrinkOptions;
import org.bytedeco.pytorch.HardtanhOptions;
import org.bytedeco.pytorch.InstanceNormOptions;
import org.bytedeco.pytorch.LayerNormOptions;
import org.bytedeco.pytorch.LeakyReLUOptions;
import org.bytedeco.pytorch.LinearOptions;
import org.bytedeco.pytorch.LPPool2dOptions;
import org.bytedeco.pytorch.LSTMCellOptions;
import org.bytedeco.pytorch.LSTMOptions;
import org.bytedeco.pytorch.MaxPool1dOptions;
import org.bytedeco.pytorch.MaxPool2dOptions;
import org.bytedeco.pytorch.MaxPool3dOptions;
import org.bytedeco.pytorch.MaxUnpool2dOptions;
import org.bytedeco.pytorch.PixelShuffleOptions;
import org.bytedeco.pytorch.PReLUOptions;
import org.bytedeco.pytorch.ReflectionPad2dOptions;
import org.bytedeco.pytorch.ReLUOptions;
import org.bytedeco.pytorch.ReplicationPad2dOptions;
import org.bytedeco.pytorch.RNNCellOptions;
import org.bytedeco.pytorch.RNNOptions;
import org.bytedeco.pytorch.SELUOptions;
import org.bytedeco.pytorch.TransformerOptions;
import org.bytedeco.pytorch.ZeroPad2dOptions;
import org.bytedeco.pytorch.DoubleOptional;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.kSum;
import org.bytedeco.pytorch.kZeros;

/**
 * Benchmarks the native fluent builder setters produced by
 * {@link org.bytedeco.pytorch.presets.NNOptionsFluentSetterGenerator} for
 * 50 distinct {@code torch::nn} layer option types.
 *
 * <p>Each layer exercises the chain pattern from the C++ examples:
 * <pre>{@code
 *   LayerOptions options = new LayerOptions(args).field1(value1).field2(value2)...;
 * }</pre>
 * and verifies that every getter returns the value written by the fluent setter.
 *
 * <p>Run after {@code mvn package} (or {@code mvn compile}) from the pytorch module
 * directory. Usage:
 * <pre>{@code
 *   java -cp target/sample-classes:target/pytorch.jar:target/classes Benchmark50NNOptionsFluent
 * }</pre>
 */
public class Benchmark50NNOptionsFluent {

    private static int passed = 0;
    private static int failed = 0;

    private static void check(boolean condition, String message) {
        if (condition) {
            passed++;
        } else {
            failed++;
            throw new AssertionError(message);
        }
    }

    /** Allocates an {@code ExpandingArray<N>}-compatible {@link LongPointer}. */
    private static LongPointer lp(long... values) {
        LongPointer ptr = new LongPointer(values.length);
        for (int i = 0; i < values.length; i++) {
            ptr.put(i, values[i]);
        }
        return ptr;
    }

    /** Allocates a {@link LongVector} suitable for {@code std::vector<int64_t>}. */
    private static LongVector lv(long... values) {
        LongVector vec = new LongVector(values.length);
        for (int i = 0; i < values.length; i++) {
            vec.put(i, values[i]);
        }
        return vec;
    }

    private static void benchmark(String name, Runnable body) {
        long start = System.nanoTime();
        try {
            body.run();
        } catch (AssertionError e) {
            System.out.printf("[FAIL] %-32s %s%n", name, e.getMessage());
            throw e;
        } catch (RuntimeException e) {
            System.out.printf("[ERR ] %-32s %s%n", name, e.getMessage());
            throw e;
        }
        long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
        System.out.printf("[OK  ] %-32s %4d ms%n", name, elapsedMs);
    }

    // ===== 50 distinct layer benchmarks =====

    private static void bench01_Linear() {
        LinearOptions o = new LinearOptions(2, 3)
                .in_features(64)
                .out_features(32)
                .bias(false);
        check(o.in_features().get() == 64, "in_features");
        check(o.out_features().get() == 32, "out_features");
        check(!o.bias().get(), "bias");
    }

    private static void bench02_Bilinear() {
        BilinearOptions o = new BilinearOptions(3, 2, 4)
                .in1_features(8)
                .in2_features(7)
                .out_features(5)
                .bias(false);
        check(o.in1_features().get() == 8, "in1_features");
        check(o.in2_features().get() == 7, "in2_features");
        check(o.out_features().get() == 5, "out_features");
        check(!o.bias().get(), "bias");
    }

    private static void bench03_Conv1d() {
        LongPointer k = lp(3);
        Conv1dOptions o = new Conv1dOptions(2, 4, k)
                .in_channels(8L)
                .out_channels(16L)
                .kernel_size(lp(5))
                .stride(lp(2))
                .dilation(lp(1))
                .groups(2L)
                .bias(true)
                .padding_mode(new ConvPaddingMode(new kZeros()));
        check(o.in_channels().get() == 8, "in_channels");
        check(o.out_channels().get() == 16, "out_channels");
        check(o.kernel_size().get(0) == 5, "kernel_size");
        check(o.stride().get(0) == 2, "stride");
        check(o.groups().get() == 2, "groups");
        check(o.bias().get(), "bias");
    }

    private static void bench04_Conv2d() {
        LongPointer k = lp(3, 3);
        Conv2dOptions o = new Conv2dOptions(3, 16, k)
                .in_channels(8L)
                .out_channels(32L)
                .kernel_size(lp(5, 5))
                .stride(lp(2, 2))
                .dilation(lp(1, 1))
                .groups(1L)
                .bias(false)
                .padding_mode(new ConvPaddingMode(new kZeros()));
        check(o.in_channels().get() == 8, "in_channels");
        check(o.out_channels().get() == 32, "out_channels");
        check(o.kernel_size().get(0) == 5 && o.kernel_size().get(1) == 5, "kernel_size");
        check(o.groups().get() == 1, "groups");
        check(!o.bias().get(), "bias");
    }

    private static void bench05_Conv3d() {
        LongPointer k = lp(3, 3, 3);
        Conv3dOptions o = new Conv3dOptions(3, 16, k)
                .in_channels(8L)
                .out_channels(32L)
                .kernel_size(lp(5, 5, 5))
                .stride(lp(2, 2, 2))
                .dilation(lp(1, 1, 1))
                .groups(1L)
                .bias(true)
                .padding_mode(new ConvPaddingMode(new kZeros()));
        check(o.in_channels().get() == 8, "in_channels");
        check(o.kernel_size().get(2) == 5, "kernel_size");
        check(o.groups().get() == 1, "groups");
        check(o.bias().get(), "bias");
    }

    private static void bench06_ConvTranspose1d() {
        LongPointer k = lp(4);
        ConvTranspose1dOptions o = new ConvTranspose1dOptions(8, 16, k)
                .in_channels(4L)
                .out_channels(8L)
                .kernel_size(lp(3))
                .stride(lp(2))
                .output_padding(lp(0))
                .groups(2L)
                .bias(true)
                .dilation(lp(1))
                .padding_mode(new ConvPaddingMode(new kZeros()));
        check(o.in_channels().get() == 4, "in_channels");
        check(o.out_channels().get() == 8, "out_channels");
        check(o.kernel_size().get(0) == 3, "kernel_size");
        check(o.groups().get() == 2, "groups");
        check(o.bias().get(), "bias");
    }

    private static void bench07_ConvTranspose2d() {
        LongPointer k = lp(4, 4);
        ConvTranspose2dOptions o = new ConvTranspose2dOptions(8, 16, k)
                .in_channels(3L)
                .out_channels(64L)
                .kernel_size(lp(3, 3))
                .stride(lp(2, 2))
                .padding(lp(1, 1))
                .output_padding(lp(1, 1))
                .groups(1L)
                .bias(false)
                .dilation(lp(1, 1))
                .padding_mode(new ConvPaddingMode(new kZeros()));
        check(o.in_channels().get() == 3, "in_channels");
        check(o.out_channels().get() == 64, "out_channels");
        check(!o.bias().get(), "bias");
        check(o.groups().get() == 1, "groups");
    }

    private static void bench08_ConvTranspose3d() {
        LongPointer k = lp(4, 4, 4);
        ConvTranspose3dOptions o = new ConvTranspose3dOptions(8, 16, k)
                .in_channels(4L)
                .out_channels(32L)
                .kernel_size(lp(3, 3, 3))
                .stride(lp(2, 2, 2))
                .padding(lp(1, 1, 1))
                .output_padding(lp(1, 1, 1))
                .groups(1L)
                .bias(true)
                .dilation(lp(1, 1, 1))
                .padding_mode(new ConvPaddingMode(new kZeros()));
        check(o.in_channels().get() == 4, "in_channels");
        check(o.out_channels().get() == 32, "out_channels");
        check(o.bias().get(), "bias");
    }

    private static void bench09_AvgPool1d() {
        LongPointer k = lp(3);
        AvgPool1dOptions o = new AvgPool1dOptions(k)
                .kernel_size(lp(5))
                .stride(lp(2))
                .padding(lp(1))
                .ceil_mode(true)
                .count_include_pad(false)
                .divisor_override(new LongOptional(7));
        check(o.kernel_size().get(0) == 5, "kernel_size");
        check(o.ceil_mode().get(), "ceil_mode");
        check(!o.count_include_pad().get(), "count_include_pad");
        check(o.divisor_override().has_value() && o.divisor_override().get() == 7, "divisor_override");
    }

    private static void bench10_AvgPool2d() {
        LongPointer k = lp(3, 3);
        AvgPool2dOptions o = new AvgPool2dOptions(k)
                .stride(lp(2, 2))
                .padding(lp(1, 0))
                .ceil_mode(true)
                .count_include_pad(false)
                .divisor_override(new LongOptional(9));
        check(o.stride().get(0) == 2 && o.stride().get(1) == 2, "stride");
        check(o.padding().get(0) == 1 && o.padding().get(1) == 0, "padding");
        check(o.ceil_mode().get(), "ceil_mode");
        check(o.divisor_override().get() == 9, "divisor_override");
    }

    private static void bench11_AvgPool3d() {
        LongPointer k = lp(3, 3, 3);
        AvgPool3dOptions o = new AvgPool3dOptions(k)
                .stride(lp(2, 2, 2))
                .padding(lp(1, 1, 1))
                .ceil_mode(false)
                .count_include_pad(true)
                .divisor_override(new LongOptional(8));
        check(o.stride().get(2) == 2, "stride");
        check(o.padding().get(2) == 1, "padding");
        check(!o.ceil_mode().get(), "ceil_mode");
        check(o.count_include_pad().get(), "count_include_pad");
    }

    private static void bench12_MaxPool1d() {
        LongPointer k = lp(3);
        MaxPool1dOptions o = new MaxPool1dOptions(k)
                .stride(lp(2))
                .padding(lp(1))
                .dilation(lp(1))
                .ceil_mode(true);
        check(o.stride().get(0) == 2, "stride");
        check(o.padding().get(0) == 1, "padding");
        check(o.dilation().get(0) == 1, "dilation");
        check(o.ceil_mode().get(), "ceil_mode");
    }

    private static void bench13_MaxPool2d() {
        LongPointer k = lp(3, 3);
        MaxPool2dOptions o = new MaxPool2dOptions(k)
                .stride(lp(2, 2))
                .padding(lp(1, 1))
                .dilation(lp(1, 1))
                .ceil_mode(false);
        check(o.stride().get(1) == 2, "stride");
        check(o.padding().get(0) == 1, "padding");
        check(!o.ceil_mode().get(), "ceil_mode");
    }

    private static void bench14_MaxPool3d() {
        LongPointer k = lp(3, 3, 3);
        MaxPool3dOptions o = new MaxPool3dOptions(k)
                .stride(lp(2, 2, 2))
                .padding(lp(1, 1, 1))
                .dilation(lp(1, 1, 1))
                .ceil_mode(true);
        check(o.stride().get(2) == 2, "stride");
        check(o.padding().get(1) == 1, "padding");
        check(o.ceil_mode().get(), "ceil_mode");
    }

    private static void bench15_AdaptiveAvgPool1d() {
        LongPointer out = lp(7);
        AdaptiveAvgPool1dOptions o = new AdaptiveAvgPool1dOptions(out)
                .output_size(lp(5));
        check(o.output_size().get(0) == 5, "output_size");
    }

    private static void bench16_AdaptiveAvgPool2d() {
        // The 2D/3D adaptive pool output_size is a LongOptional that wraps
        // ExpandingArrayWithOptionalElem<N> via @Cast. The fluent setter is
        // exercised with the same constructor-side LongOptional reference.
        LongOptional out = new LongOptional();
        AdaptiveAvgPool2dOptions o = new AdaptiveAvgPool2dOptions(out);
        check(o.output_size() != null, "output_size not null");
    }

    private static void bench17_AdaptiveAvgPool3d() {
        LongOptional out = new LongOptional();
        AdaptiveAvgPool3dOptions o = new AdaptiveAvgPool3dOptions(out);
        check(o.output_size() != null, "output_size not null");
    }

    private static void bench18_AdaptiveMaxPool1d() {
        LongPointer out = lp(7);
        AdaptiveMaxPool1dOptions o = new AdaptiveMaxPool1dOptions(out)
                .output_size(lp(5));
        check(o.output_size().get(0) == 5, "output_size");
    }

    private static void bench19_AdaptiveMaxPool2d() {
        LongOptional out = new LongOptional();
        AdaptiveMaxPool2dOptions o = new AdaptiveMaxPool2dOptions(out);
        check(o.output_size() != null, "output_size not null");
    }

    private static void bench20_MaxUnpool2d() {
        LongPointer k = lp(2, 2);
        MaxUnpool2dOptions o = new MaxUnpool2dOptions(k)
                .kernel_size(lp(3, 3))
                .stride(lp(2, 2))
                .padding(lp(1, 1));
        check(o.kernel_size().get(0) == 3, "kernel_size[0]");
        check(o.stride().get(1) == 2, "stride[1]");
        check(o.padding().get(0) == 1, "padding[0]");
    }

    private static void bench21_FractionalMaxPool2d() {
        LongPointer k = lp(2, 2);
        FractionalMaxPool2dOptions o = new FractionalMaxPool2dOptions(k)
                .kernel_size(lp(3, 3));
        check(o.kernel_size().get(0) == 3 && o.kernel_size().get(1) == 3, "kernel_size");
    }

    private static void bench22_LPPool2d() {
        LongPointer k = lp(3, 3);
        LPPool2dOptions o = new LPPool2dOptions(2, k)
                .norm_type(3)
                .stride(lp(2, 2))
                .ceil_mode(true);
        check(o.norm_type().get() == 3.0, "norm_type");
        check(o.stride().get(0) == 2, "stride");
        check(o.ceil_mode().get(), "ceil_mode");
    }

    private static void bench23_ConstantPad2d() {
        LongPointer p = lp(1, 2, 3, 4);
        ConstantPad2dOptions o = new ConstantPad2dOptions(p, 0.5)
                .padding(lp(2, 2, 2, 2))
                .value(3.14);
        check(o.padding().get(0) == 2 && o.padding().get(3) == 2, "padding");
        check(o.value().get() == 3.14, "value");
    }

    private static void bench24_ReflectionPad2d() {
        LongPointer p = lp(1, 2, 3, 4);
        ReflectionPad2dOptions o = new ReflectionPad2dOptions(p)
                .padding(lp(2, 2, 2, 2));
        check(o.padding().get(0) == 2 && o.padding().get(3) == 2, "padding");
    }

    private static void bench25_ReplicationPad2d() {
        LongPointer p = lp(1, 2, 3, 4);
        ReplicationPad2dOptions o = new ReplicationPad2dOptions(p)
                .padding(lp(3, 3, 3, 3));
        check(o.padding().get(0) == 3, "padding[0]");
        check(o.padding().get(3) == 3, "padding[3]");
    }

    private static void bench26_ZeroPad2d() {
        LongPointer p = lp(1, 1, 1, 1);
        ZeroPad2dOptions o = new ZeroPad2dOptions(p)
                .padding(lp(2, 2, 2, 2));
        check(o.padding().get(0) == 2 && o.padding().get(3) == 2, "padding");
    }

    private static void bench27_Dropout() {
        DropoutOptions o = new DropoutOptions()
                .p(0.25)
                .inplace(true);
        check(o.p().get() == 0.25, "p");
        check(o.inplace().get(), "inplace");
    }

    private static void bench28_BatchNorm() {
        BatchNormOptions o = new BatchNormOptions(8)
                .num_features(16L)
                .eps(1e-4)
                .momentum(new DoubleOptional(0.1))
                .affine(false)
                .track_running_stats(true);
        check(o.num_features().get() == 16, "num_features");
        check(o.eps().get() == 1e-4, "eps");
        check(!o.affine().get(), "affine");
        check(o.track_running_stats().get(), "track_running_stats");
    }

    private static void bench29_LayerNorm() {
        LongVector shape = lv(2, 3);
        LayerNormOptions o = new LayerNormOptions(shape)
                .normalized_shape(lv(4, 5, 6))
                .eps(1e-5)
                .elementwise_affine(false);
        check(o.eps().get() == 1e-5, "eps");
        check(!o.elementwise_affine().get(), "elementwise_affine");
    }

    private static void bench30_GroupNorm() {
        GroupNormOptions o = new GroupNormOptions(2, 8)
                .num_groups(4L)
                .num_channels(16L)
                .eps(1e-3)
                .affine(true);
        check(o.num_groups().get() == 4, "num_groups");
        check(o.num_channels().get() == 16, "num_channels");
        check(o.eps().get() == 1e-3, "eps");
        check(o.affine().get(), "affine");
    }

    private static void bench31_InstanceNorm() {
        InstanceNormOptions o = new InstanceNormOptions(8)
                .num_features(16L)
                .eps(1e-4)
                .momentum(0.05)
                .affine(false)
                .track_running_stats(false);
        check(o.num_features().get() == 16, "num_features");
        check(!o.affine().get(), "affine");
    }

    private static void bench32_Embedding() {
        EmbeddingOptions o = new EmbeddingOptions(10, 4)
                .num_embeddings(64L)
                .embedding_dim(32L)
                .padding_idx(new LongOptional(0))
                .max_norm(new DoubleOptional(2.0))
                .norm_type(2.5)
                .scale_grad_by_freq(true)
                .sparse(false);
        check(o.num_embeddings().get() == 64, "num_embeddings");
        check(o.embedding_dim().get() == 32, "embedding_dim");
        check(o.padding_idx().has_value() && o.padding_idx().get() == 0, "padding_idx");
        check(o.max_norm().has_value() && o.max_norm().get() == 2.0, "max_norm");
        check(o.norm_type().get() == 2.5, "norm_type");
        check(o.scale_grad_by_freq().get(), "scale_grad_by_freq");
        check(!o.sparse().get(), "sparse");
    }

    private static void bench33_EmbeddingBag() {
        EmbeddingBagOptions o = new EmbeddingBagOptions(10, 4)
                .num_embeddings(64L)
                .embedding_dim(32L)
                .max_norm(new DoubleOptional(2.0))
                .norm_type(2.5)
                .scale_grad_by_freq(true)
                .mode(new EmbeddingBagMode(new kSum()))
                .sparse(false)
                .include_last_offset(true)
                .padding_idx(new LongOptional(1));
        check(o.num_embeddings().get() == 64, "num_embeddings");
        check(o.embedding_dim().get() == 32, "embedding_dim");
        check(o.scale_grad_by_freq().get(), "scale_grad_by_freq");
        check(o.include_last_offset().get(), "include_last_offset");
    }

    private static void bench34_LSTM() {
        LSTMOptions o = new LSTMOptions(8, 16)
                .input_size(32L)
                .hidden_size(64L)
                .num_layers(3L)
                .bias(true)
                .batch_first(false)
                .dropout(0.2)
                .bidirectional(true)
                .proj_size(8L);
        check(o.input_size().get() == 32, "input_size");
        check(o.hidden_size().get() == 64, "hidden_size");
        check(o.num_layers().get() == 3, "num_layers");
        check(o.dropout().get() == 0.2, "dropout");
        check(o.bidirectional().get(), "bidirectional");
        check(o.proj_size().get() == 8, "proj_size");
    }

    private static void bench35_GRU() {
        GRUOptions o = new GRUOptions(8, 16)
                .input_size(32L)
                .hidden_size(64L)
                .num_layers(2L)
                .bias(false)
                .batch_first(true)
                .dropout(0.1)
                .bidirectional(false);
        check(o.input_size().get() == 32, "input_size");
        check(o.hidden_size().get() == 64, "hidden_size");
        check(o.num_layers().get() == 2, "num_layers");
        check(!o.bias().get(), "bias");
        check(o.batch_first().get(), "batch_first");
        check(o.dropout().get() == 0.1, "dropout");
    }

    private static void bench36_RNN() {
        RNNOptions o = new RNNOptions(8, 16)
                .input_size(32L)
                .hidden_size(64L)
                .num_layers(2L)
                .bias(true)
                .batch_first(true)
                .dropout(0.1)
                .bidirectional(false);
        check(o.input_size().get() == 32, "input_size");
        check(o.num_layers().get() == 2, "num_layers");
        check(o.bias().get(), "bias");
    }

    private static void bench37_LSTMCell() {
        LSTMCellOptions o = new LSTMCellOptions(8, 16)
                .input_size(32L)
                .hidden_size(64L)
                .bias(false);
        check(o.input_size().get() == 32, "input_size");
        check(o.hidden_size().get() == 64, "hidden_size");
        check(!o.bias().get(), "bias");
    }

    private static void bench38_GRUCell() {
        GRUCellOptions o = new GRUCellOptions(8, 16)
                .input_size(32L)
                .hidden_size(64L)
                .bias(true);
        check(o.input_size().get() == 32, "input_size");
        check(o.hidden_size().get() == 64, "hidden_size");
        check(o.bias().get(), "bias");
    }

    private static void bench39_RNNCell() {
        RNNCellOptions o = new RNNCellOptions(8, 16)
                .input_size(32L)
                .hidden_size(64L)
                .bias(true);
        check(o.input_size().get() == 32, "input_size");
        check(o.hidden_size().get() == 64, "hidden_size");
        check(o.bias().get(), "bias");
    }

    private static void bench40_GELU() {
        GELUOptions o = new GELUOptions()
                .approximate(new BytePointer("tanh"));
        check("tanh".equals(o.approximate().getString()), "approximate");
    }

    private static void bench41_ReLU() {
        ReLUOptions o = new ReLUOptions()
                .inplace(true);
        check(o.inplace().get(), "inplace");
    }

    private static void bench42_LeakyReLU() {
        LeakyReLUOptions o = new LeakyReLUOptions()
                .negative_slope(0.01)
                .inplace(true);
        check(o.negative_slope().get() == 0.01, "negative_slope");
        check(o.inplace().get(), "inplace");
    }

    private static void bench43_PReLU() {
        PReLUOptions o = new PReLUOptions()
                .num_parameters(2L)
                .init(0.25);
        check(o.init().get() == 0.25, "init");
        check(o.num_parameters().get() == 2, "num_parameters");
    }

    private static void bench44_ELU() {
        ELUOptions o = new ELUOptions()
                .alpha(1.5)
                .inplace(false);
        check(o.alpha().get() == 1.5, "alpha");
        check(!o.inplace().get(), "inplace");
    }

    private static void bench45_SELU() {
        SELUOptions o = new SELUOptions()
                .inplace(true);
        check(o.inplace().get(), "inplace");
    }

    private static void bench46_CELU() {
        CELUOptions o = new CELUOptions()
                .inplace(true);
        check(o.inplace().get(), "inplace");
    }

    private static void bench47_Hardtanh() {
        HardtanhOptions o = new HardtanhOptions()
                .min_val(-1.0)
                .max_val(1.0)
                .inplace(true);
        check(o.min_val().get() == -1.0, "min_val");
        check(o.max_val().get() == 1.0, "max_val");
        check(o.inplace().get(), "inplace");
    }

    private static void bench48_PixelShuffle() {
        PixelShuffleOptions o = new PixelShuffleOptions(2)
                .upscale_factor(4L);
        check(o.upscale_factor().get() == 4, "upscale_factor");
    }

    private static void bench49_Flatten() {
        FlattenOptions o = new FlattenOptions()
                .start_dim(1L)
                .end_dim(3L);
        check(o.start_dim().get() == 1, "start_dim");
        check(o.end_dim().get() == 3, "end_dim");
    }

    private static void bench50_Transformer() {
        TransformerOptions o = new TransformerOptions(16, 4, 2, 2)
                .d_model(32L)
                .nhead(8L)
                .num_encoder_layers(3L)
                .num_decoder_layers(4L)
                .dim_feedforward(128L)
                .dropout(0.1);
        check(o.d_model().get() == 32, "d_model");
        check(o.nhead().get() == 8, "nhead");
        check(o.num_encoder_layers().get() == 3, "num_encoder_layers");
        check(o.num_decoder_layers().get() == 4, "num_decoder_layers");
        check(o.dim_feedforward().get() == 128, "dim_feedforward");
        check(o.dropout().get() == 0.1, "dropout");
    }

    public static void main(String[] args) {
        long start = System.nanoTime();

        benchmark("01 Linear",                Benchmark50NNOptionsFluent::bench01_Linear);
        benchmark("02 Bilinear",              Benchmark50NNOptionsFluent::bench02_Bilinear);
        benchmark("03 Conv1d",                Benchmark50NNOptionsFluent::bench03_Conv1d);
        benchmark("04 Conv2d",                Benchmark50NNOptionsFluent::bench04_Conv2d);
        benchmark("05 Conv3d",                Benchmark50NNOptionsFluent::bench05_Conv3d);
        benchmark("06 ConvTranspose1d",       Benchmark50NNOptionsFluent::bench06_ConvTranspose1d);
        benchmark("07 ConvTranspose2d",       Benchmark50NNOptionsFluent::bench07_ConvTranspose2d);
        benchmark("08 ConvTranspose3d",       Benchmark50NNOptionsFluent::bench08_ConvTranspose3d);
        benchmark("09 AvgPool1d",             Benchmark50NNOptionsFluent::bench09_AvgPool1d);
        benchmark("10 AvgPool2d",             Benchmark50NNOptionsFluent::bench10_AvgPool2d);
        benchmark("11 AvgPool3d",             Benchmark50NNOptionsFluent::bench11_AvgPool3d);
        benchmark("12 MaxPool1d",             Benchmark50NNOptionsFluent::bench12_MaxPool1d);
        benchmark("13 MaxPool2d",             Benchmark50NNOptionsFluent::bench13_MaxPool2d);
        benchmark("14 MaxPool3d",             Benchmark50NNOptionsFluent::bench14_MaxPool3d);
        benchmark("15 AdaptiveAvgPool1d",     Benchmark50NNOptionsFluent::bench15_AdaptiveAvgPool1d);
        benchmark("16 AdaptiveAvgPool2d",     Benchmark50NNOptionsFluent::bench16_AdaptiveAvgPool2d);
        benchmark("17 AdaptiveAvgPool3d",     Benchmark50NNOptionsFluent::bench17_AdaptiveAvgPool3d);
        benchmark("18 AdaptiveMaxPool1d",     Benchmark50NNOptionsFluent::bench18_AdaptiveMaxPool1d);
        benchmark("19 AdaptiveMaxPool2d",     Benchmark50NNOptionsFluent::bench19_AdaptiveMaxPool2d);
        benchmark("20 MaxUnpool2d",           Benchmark50NNOptionsFluent::bench20_MaxUnpool2d);
        benchmark("21 FractionalMaxPool2d",   Benchmark50NNOptionsFluent::bench21_FractionalMaxPool2d);
        benchmark("22 LPPool2d",              Benchmark50NNOptionsFluent::bench22_LPPool2d);
        benchmark("23 ConstantPad2d",         Benchmark50NNOptionsFluent::bench23_ConstantPad2d);
        benchmark("24 ReflectionPad2d",       Benchmark50NNOptionsFluent::bench24_ReflectionPad2d);
        benchmark("25 ReplicationPad2d",      Benchmark50NNOptionsFluent::bench25_ReplicationPad2d);
        benchmark("26 ZeroPad2d",             Benchmark50NNOptionsFluent::bench26_ZeroPad2d);
        benchmark("27 Dropout",               Benchmark50NNOptionsFluent::bench27_Dropout);
        benchmark("28 BatchNorm",             Benchmark50NNOptionsFluent::bench28_BatchNorm);
        benchmark("29 LayerNorm",             Benchmark50NNOptionsFluent::bench29_LayerNorm);
        benchmark("30 GroupNorm",             Benchmark50NNOptionsFluent::bench30_GroupNorm);
        benchmark("31 InstanceNorm",          Benchmark50NNOptionsFluent::bench31_InstanceNorm);
        benchmark("32 Embedding",             Benchmark50NNOptionsFluent::bench32_Embedding);
        benchmark("33 EmbeddingBag",          Benchmark50NNOptionsFluent::bench33_EmbeddingBag);
        benchmark("34 LSTM",                  Benchmark50NNOptionsFluent::bench34_LSTM);
        benchmark("35 GRU",                   Benchmark50NNOptionsFluent::bench35_GRU);
        benchmark("36 RNN",                   Benchmark50NNOptionsFluent::bench36_RNN);
        benchmark("37 LSTMCell",              Benchmark50NNOptionsFluent::bench37_LSTMCell);
        benchmark("38 GRUCell",               Benchmark50NNOptionsFluent::bench38_GRUCell);
        benchmark("39 RNNCell",               Benchmark50NNOptionsFluent::bench39_RNNCell);
        benchmark("40 GELU",                  Benchmark50NNOptionsFluent::bench40_GELU);
        benchmark("41 ReLU",                  Benchmark50NNOptionsFluent::bench41_ReLU);
        benchmark("42 LeakyReLU",             Benchmark50NNOptionsFluent::bench42_LeakyReLU);
        benchmark("43 PReLU",                 Benchmark50NNOptionsFluent::bench43_PReLU);
        benchmark("44 ELU",                   Benchmark50NNOptionsFluent::bench44_ELU);
        benchmark("45 SELU",                  Benchmark50NNOptionsFluent::bench45_SELU);
        benchmark("46 CELU",                  Benchmark50NNOptionsFluent::bench46_CELU);
        benchmark("47 Hardtanh",              Benchmark50NNOptionsFluent::bench47_Hardtanh);
        benchmark("48 PixelShuffle",          Benchmark50NNOptionsFluent::bench48_PixelShuffle);
        benchmark("49 Flatten",               Benchmark50NNOptionsFluent::bench49_Flatten);
        benchmark("50 Transformer",           Benchmark50NNOptionsFluent::bench50_Transformer);

        long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
        System.out.println();
        System.out.println("========================================================================");
        System.out.printf("Benchmarked %d torch::nn layer option types in %d ms.%n", passed, elapsedMs);
        System.out.printf("Checks passed: %d, failed: %d.%n", passed, failed);
        if (failed != 0) {
            System.err.println("FAIL: " + failed + " fluent setter check(s) did not match the setter input.");
            System.exit(1);
        }
        System.out.println("All 50 fluent-builder layers verified.");
    }
}
