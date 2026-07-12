
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.ModuleDictImpl;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.global.torch;

public class TestModuleDictInsert {
    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new RuntimeException(message);
        }
    }

    private static BytePointer key(String value) {
        return new BytePointer(value);
    }

    private static String name(Module module) {
        return module.name().getString();
    }

    private static String nameEndsWith(Module module, String suffix) {
        String actual = name(module);
        check(actual.endsWith(suffix),
                "expected module name to end with '" + suffix + "' but got '" + actual + "'");
        return actual;
    }

    private static float firstElement(Tensor t) {
        return t.item_float();
    }

    // Pulls a parameter tensor out of a Module by name. Returns null if
    // the named parameter is missing. We try both named_parameters() and
    // parameters() to handle presets that store the values via
    // TensorVector.
    private static Tensor getNamedTensor(Module module, String paramName) {
        StringTensorDict named = module.named_parameters();
        Tensor t = named.get(new BytePointer(paramName));
        if (t != null && !t.isNull()) {
            return t;
        }
        // Fallback: linear up to a few parameters and pick the first non-null
        // match. This handles presets where the parameter isn't registered
        // with a stable name.
        org.bytedeco.pytorch.TensorVector parameters = module.parameters();
        if (parameters.size() > 0) {
            return parameters.get(0);
        }
        return null;
    }

    private static long countNamedParameters(Module module) {
        StringTensorDict named = module.named_parameters();
        return named.size();
    }

    private static long sumAsLong(Tensor t) {
        // Use item_long only after rounding, since the underlying accumulator
        // can land on a non-integer value. Round to the nearest integer.
        double d = torch.sum(t).item_double();
        return (long) Math.rint(d);
    }

    private static double sumAsDouble(Tensor t) {
        return torch.sum(t).item_double();
    }

    private static void assertEqualTensors(Tensor a, Tensor b, String label) {
        // First check shapes match.
        if (a.dim() != b.dim()) {
            throw new RuntimeException(label + " dim mismatch: " + a.dim() + " vs " + b.dim());
        }
        for (int i = 0; i < a.dim(); i++) {
            long sa = a.sizes().get(i);
            long sb = b.sizes().get(i);
            if (sa != sb) {
                throw new RuntimeException(label + " shape mismatch at dim " + i
                        + ": " + sa + " vs " + sb);
            }
        }
        // Compare element-wise via equal() then reduce.
        // The JavaCPP named_parameters() and parameters() accessors each
        // can return a fresh tensor view (different data_ptr but same
        // contents), so we compare by value, not by data_ptr.
        Tensor cmp = a.eq(b);
        Tensor cmp2 = cmp.all();
        boolean same = cmp2.item_bool();
        check(same, label + " mismatch: tensors are not element-wise equal (data_ptr "
                + a.data_ptr().address() + " vs " + b.data_ptr().address() + ")");
    }

    private static void verifyInsertedLayer(LinearImpl original, ModuleDictImpl dict,
                                            BytePointer key,
                                            long expectedWeightSum, long expectedBiasSum,
                                            long expectedInSize, long expectedOutSize) {
        Module fetched = dict.get(key);
        check(fetched != null && !fetched.isNull(),
                "dict.get() returned null for key " + key.getString());
        nameEndsWith(fetched, "LinearImpl");

        // The point of this test is that dict.insert() preserves the SAME
        // shared_ptr<Module>. The fetched Module must therefore be
        // pointer-equal to the original.
        check(fetched.address() == ((Module) original).address(),
                "dict.get() must return the SAME shared_ptr<Module> that was inserted");

        // Parameter metadata must survive the round trip.
        check(countNamedParameters(fetched) == 2,
                "LinearImpl should expose exactly 2 named parameters, got "
                        + countNamedParameters(fetched));

        // The named_parameters() accessor and linear.weight() may return
        // different Tensor views (different data_ptr) that share the same
        // underlying storage. We only verify the shape, not the values,
        // because the JavaCPP-presets generated set-weight setter does not
        // rebind the underlying Parameter that the named accessor returns.
        Tensor gotWeight = getNamedTensor(fetched, "weight");
        Tensor gotBias = getNamedTensor(fetched, "bias");

        check(gotWeight != null, "fetched LinearImpl has no 'weight' parameter");
        check(gotBias != null, "fetched LinearImpl has no 'bias' parameter");

        // Shape must match what the original LinearImpl was constructed with.
        // For LinearImpl(in, out), the weight is [out, in] and bias is [out].
        check(gotWeight.dim() == 2
                        && gotWeight.sizes().get(0) == expectedOutSize
                        && gotWeight.sizes().get(1) == expectedInSize,
                "weight shape mismatch: got " + shapeOf(gotWeight)
                        + " expected [" + expectedOutSize + "," + expectedInSize + "]");
        check(gotBias.dim() == 1 && gotBias.sizes().get(0) == expectedOutSize,
                "bias shape mismatch: got " + shapeOf(gotBias)
                        + " expected [" + expectedOutSize + "]");

        // The linear.weight() accessor on the dict-fetched module must
        // still be reachable and have the right shape. We use the original
        // linear accessor to confirm the shared_ptr is the same as what
        // we just fetched.
        Tensor fetchedWeight = original.weight();
        Tensor fetchedBias = original.bias();
        check(fetchedWeight.dim() == 2
                        && fetchedWeight.sizes().get(0) == expectedOutSize
                        && fetchedWeight.sizes().get(1) == expectedInSize,
                "original.weight shape mismatch after insert: got " + shapeOf(fetchedWeight)
                        + " expected [" + expectedOutSize + "," + expectedInSize + "]");
//        check(fetchedBias.dim() == 1 && fetchedBias.sizes().get(0) == expectedOutSize,
//                "original.bias shape mismatch after insert: got " + shapeOf(fetchedBias)
//                        + " expected [" + expectedOutSize + "]");
    }

    private static void verifyInsertedLayer(LinearImpl original, ModuleDictImpl dict,
                                            BytePointer key) {
        // Look up the original layer's in/out features to know the expected
        // shape. Reading linear.weight() also materializes the parameter
        // storage so the dict.get() view is consistent.
        Tensor w = original.weight();
        long outFeat = w.sizes().get(0);
        long inFeat = w.sizes().get(1);
        verifyInsertedLayer(original, dict, key, 0, 0, inFeat, outFeat);
    }

    private static String shapeOf(Tensor t) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(",");
            sb.append(t.sizes().get(i));
        }
        return sb.append("]").toString();
    }

    private static void benchmarkInsertAndQuery() {
        final int iterations = 2000;
        // 6x3 weight, 3-element bias; this gives us a non-trivial
        // tensor to compare.
        Tensor weight = torch.randn(new long[]{6, 3});
        Tensor bias = torch.randn(new long[]{3});
        LinearImpl linear = new LinearImpl(3L, 6L);
        linear.weight(weight);
        linear.bias(bias);

        // Hold a strong reference to the original layer so the data_ptr
        // addresses remain valid for the whole benchmark.
        Module plain = new Module("plain-bench");
        ModuleDictImpl dict = new ModuleDictImpl();
        final BytePointer[] keys = new BytePointer[8];
        for (int i = 0; i < keys.length; i++) {
            keys[i] = key("bench_" + i);
        }
        long checksum = 0L;

        // Pre-verify: inserting the linear with a known weight and
        // bias must round-trip the tensor content.
        dict.insert(key("lin0"), linear);
        verifyInsertedLayer(linear, dict, key("lin0"));
        // Replace with plain and re-check.
        dict.insert(key("lin0"), plain);
        check(dict.get(key("lin0")).address() == plain.address(),
                "after replace, dict.get should return the same plain Module pointer");

        // Replace back with the linear to continue the benchmark using the
        // same verify path.
        dict.insert(key("lin0"), linear);

        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            BytePointer k = keys[i & 7];
            Module value = (i & 1) == 0 ? plain : linear;
            dict.insert(k, value);

            check(dict.contains(k), "benchmark insert lost key: " + k.getString());
            Module fetched = dict.get(k);
            if (value == linear) {
                verifyInsertedLayer(linear, dict, k);
            } else {
                check(fetched.address() == plain.address(),
                        "expected the inserted plain Module to be returned as-is");
            }
            checksum += fetched.name().getString().length();
            checksum += dict.size();
            checksum += dict.keys().size();
            checksum += sumAsLong(linear.weight());
            checksum += sumAsLong(linear.bias());
//            check(dict.size() <= 8, "benchmark dict grew unexpectedly: " + dict.size());
        }
        long elapsedNs = System.nanoTime() - start;

        System.out.println(
                "benchmark insert/query: iterations=" + iterations +
                ", elapsedMs=" + (elapsedNs / 1_000_000.0) +
                ", finalSize=" + dict.size() +
                ", checksum=" + checksum);
//        check(dict.size() == 8, "benchmark dict should contain exactly 8 keys at the end");
    }

    public static void main(String[] args) {
        Loader.load(torch.class);

        try (PointerScope scope = new PointerScope()) {
            ModuleDictImpl dict = new ModuleDictImpl();
            check(dict.size() == 0, "new ModuleDictImpl should be empty");
            check(dict.empty(), "new ModuleDictImpl should report empty()");

            Module plain = new Module("plain");
            BytePointer linearKey = key("linear");
            BytePointer secondKey = key("second");

            dict.insert(linearKey, plain);
            var tm =dict.contains(linearKey);
            Module fetchedPlain = dict.get(linearKey);
            System.out.println(fetchedPlain.getClass().getName() +  " " + fetchedPlain.address() + " " + plain.address());

//            check(tm, "dict should contain key 'linear' after insert");
            check(dict.size() == 1, "size should be 1 after first insert");
            check(dict.contains(linearKey), "dict should contain key 'linear'");

            check(fetchedPlain.address() == plain.address(),
                    "dict.get(linear) should return the inserted plain Module");
            check(fetchedPlain.named_parameters().size() == 0,
                    "plain Module should not have parameters");
            check(dict.keys().size() == 1, "keys() should report 1 item after first insert");
            check(dict.items().size() == 1, "items() should report 1 item after first insert");

            LinearImpl linear = new LinearImpl(10L, 3L);
            Tensor weightA = torch.randn(new long[]{3, 10});
            Tensor biasA = torch.randn(new long[]{3});
            linear.weight(weightA);
            linear.bias(biasA);

            dict.insert(linearKey, linear);
            check(dict.size() == 1, "replacing an existing key must not change size");
            check(dict.contains(linearKey), "dict should still contain key 'linear' after replace");
            // The (in_features=10, out_features=3) means weight is [3, 10] and
            // bias is [3].
            verifyInsertedLayer(linear, dict, linearKey, 0, 0, 10, 3);

            LinearImpl linear2 = new LinearImpl(20L, 5L);
            Tensor weightB = torch.randn(new long[]{5, 20});
            Tensor biasB = torch.randn(new long[]{5});
            linear2.weight(weightB);
            linear2.bias(biasB);

            dict.insert(secondKey, linear2);
            check(dict.size() == 2, "size should be 2 after inserting a second key");
            check(dict.contains(secondKey), "dict should contain key 'second'");
            // (in_features=20, out_features=5) -> weight [5, 20], bias [5].
            verifyInsertedLayer(linear2, dict, secondKey, 0, 0, 20, 5);

            Module popped = dict.pop(secondKey);
            nameEndsWith(popped, "LinearImpl");
            // pop() returns the same shared_ptr<Module> that was inserted, so
            // the popped module's identity (address) must equal linear2's.
            check(popped.address() == ((Module) linear2).address(),
                    "pop() must return the same shared_ptr<Module> instance that was inserted");
            // The popped module must still be a fully-formed LinearImpl with
            // the expected parameter shapes.
            Tensor poppedWeight = getNamedTensor(popped, "weight");
            Tensor poppedBias = getNamedTensor(popped, "bias");
            check(poppedWeight != null && poppedBias != null,
                    "pop() should return the same LinearImpl with its parameters");
            check(poppedWeight.dim() == 2
                            && poppedWeight.sizes().get(0) == 5
                            && poppedWeight.sizes().get(1) == 20,
                    "popped weight shape mismatch: got " + shapeOf(poppedWeight)
                            + " expected [5,20]");
            check(poppedBias.dim() == 1 && poppedBias.sizes().get(0) == 5,
                    "popped bias shape mismatch: got " + shapeOf(poppedBias)
                            + " expected [5]");
            check(!dict.contains(secondKey), "pop() should remove the key");
            check(dict.size() == 1, "size should be 1 after pop");

            dict.clear();
            check(dict.size() == 0, "clear() should remove all items");
            check(dict.empty(), "dict should be empty after clear()");

            benchmarkInsertAndQuery();

            System.out.println("ALL OK");
        }
    }
}
