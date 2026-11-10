import java.lang.reflect.Method;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class TestSequentialDropout {
    public static void main(String[] args) throws Exception {
        // Check before allocating anything: the old bindings can crash natively.
        for (Class<?>[] parameters : new Class<?>[][] {
                {}, {double.class}, {DropoutOptions.class}}) {
            Method allocate = DropoutImpl.class.getDeclaredMethod("allocate", parameters);
            Name name = allocate.getAnnotation(Name.class);
            if (!allocate.isAnnotationPresent(SharedPtr.class) || name == null
                    || name.value().length != 1
                    || !name.value()[0].equals("std::make_shared<torch::nn::DropoutImpl>")) {
                throw new AssertionError("DropoutImpl allocation must preserve shared ownership: " + allocate);
            }
        }
        if (Boolean.getBoolean("org.bytedeco.javacpp.nopointergc")) {
            throw new AssertionError("Run with pointer GC enabled");
        }

        for (boolean named : new boolean[] {false, true}) {
            // All constructors in eval mode, then deterministic training endpoints.
            for (int variant = 0; variant < 5; variant++) {
                try (PointerScope scope = new PointerScope()) {
                    Tensor input = ones(2, 3); // Default tensor options use the CPU.
                    DropoutImpl dropout = variant == 0 ? new DropoutImpl()
                            : variant == 1 ? new DropoutImpl(0.5)
                            : new DropoutImpl(new DropoutOptions(variant == 2 ? 0.5 : variant - 3));
                    SequentialImpl sequence = new SequentialImpl();
                    if (named) {
                        sequence.push_back("dropout", dropout);
                    } else {
                        sequence.push_back(dropout);
                    }
                    sequence.train(variant >= 3);
                    dropout.deallocate();
                    Tensor expected = variant == 4 ? zeros_like(input) : input;
                    check(sequence.forward(input), expected);
                    System.gc();
                    check(sequence.forward(input), expected);
                }
            }
        }

        try (PointerScope scope = new PointerScope()) {
            Tensor input = ones(2, 3);
            DropoutImpl dropout = new DropoutImpl(0.0);
            AnyModule module = new AnyModule(dropout);
            dropout.deallocate();
            check(module.forward(input), input);
            System.gc();
            check(module.forward(input), input);
        }

        try (PointerScope scope = new PointerScope()) {
            Tensor input = ones(2, 3);
            LinearImpl linear = new LinearImpl(3, 2);
            ReLUImpl relu = new ReLUImpl();
            DropoutImpl dropout = new DropoutImpl();
            Tensor expected = relu.forward(linear.forward(input));
            SequentialImpl sequence = new SequentialImpl();
            sequence.push_back("linear", linear);
            sequence.push_back("relu", relu);
            sequence.push_back("dropout", dropout);
            sequence.eval();
            linear.deallocate();
            relu.deallocate();
            dropout.deallocate();
            check(sequence.forward(input), expected);
            System.gc();
            check(sequence.forward(input), expected);
        }

        try (PointerScope scope = new PointerScope()) {
            try {
                new DropoutImpl(1.5);
                throw new AssertionError("Invalid dropout probability was accepted");
            } catch (RuntimeException e) {
                if (e.getMessage() == null || !e.getMessage().contains("dropout probability")) {
                    throw e;
                }
            }
        }
        System.out.println("Sequential and AnyModule retain shared ownership of DropoutImpl.");
    }

    private static void check(Tensor output, Tensor expected) {
        if (!output.is_same_size(expected) || output.scalar_type() != expected.scalar_type()
                || !output.equal(expected)) {
            throw new AssertionError("Unexpected forward output values, shape, or dtype");
        }
    }
}
