import java.nio.FloatBuffer;

import org.bytedeco.javacpp.*;
import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

/**
 * Reference
 * https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/cc/tutorials/example_trainer.cc
 *
 * @author Nico Hezel
 *
 */
public class EigenvalueExample {

    public static class Options {
        int num_iterations = 20;    // Each step repeats this many times
        boolean use_gpu = false;    // Whether to use GPU in the training
    }

    public static void main(String args[]) throws Exception {

        // Platform-specific initialization routine that may be invoked
        // by a main() program that uses TensorFlow.
        //
        // We need to call this to set up global state for TensorFlow.
        InitMain("trainer", (int[])null, null);

        // Construct a Tensorflow session and run a computation graph
        buildAndRun(new Options());
    }

    /**
     * Checks the status and throws an Exception in case any error occurred
     *
     * @param s
     * @throws Exception
     */
    public static void checkStatus(Status s) throws Exception {
        if (!s.ok())
            throw new Exception(s.error_message().getString());
        s.close();
    }

    /**
     * Constructs a simple Tensorflow graph
     *
     * A = [3 2; -1 0]; x = rand(2, 1);
     * We want to compute the largest eigenvalue for A.
     * repeat x = y / y.norm(); y = A * x; end
     *
     * @return
     * @throws Exception
     */
    public static GraphDef CreateGraphDef() throws Exception {

        // Create a new empty graph
        Scope root = Scope.NewRootScope();

        // a = [3 2; -1 0]
        Output a = Const(root, Tensor.create(new float[] {3.f, 2.f, -1.f, 0.f}, new TensorShape(2, 2)));

        // x = rand(2, 1)
        Placeholder x = new Placeholder(root.WithOpName("x"), DT_FLOAT);

        // y = a * x
        MatMul y = new MatMul(root.WithOpName("y"), new Input(a), x.asInput());

        // y2 = y.^2
        Square y2 = new Square(root, y.asInput());

        // y2_sum = sum(y2)
        Sum y2_sum = new Sum(root, y2.asInput(), new Input(0));

        // y_norm = sqrt(y2_sum)
        Sqrt y_norm = new Sqrt(root, y2_sum.asInput());

        // y_normalized = y ./ y_norm
        Div y_normalized = new Div(root.WithOpName("y_normalized"), y.asInput(), y_norm.asInput());

        // construct a graph definition object
        GraphDef def = new GraphDef();
        checkStatus(root.ToGraphDef(def));

        return def;
    }


    /**
     * Constructs a graph and a new Tensorflow session.
     * Runs the graph inside the session multiple times.
     *
     * @param opts
     * @throws Exception
     */
    public static void buildAndRun(final Options opts) throws Exception {

        // Construct a computation graph
        try(GraphDef def = CreateGraphDef()) {

            // Creates a session.
            SessionOptions options = new SessionOptions();
            try(final Session session = new Session(options)) {

                // Copy the graph to the GPU if needed
                if (options.target() == null)
                    SetDefaultDevice(opts.use_gpu ? "/gpu:0" : "/cpu:0", def);

                // Create the graph to be used for the session.
                checkStatus(session.Create(def));

                // Randomly initialize the input.
                Tensor x = new Tensor(DT_FLOAT, new TensorShape(2, 1));
                FloatBuffer x_flat = x.createBuffer();
                x_flat.put(0, (float)Math.random());
                x_flat.put(1, (float)Math.random());
                float inv_norm = 1 / (float)Math.sqrt(x_flat.get(0) *  x_flat.get(0) + x_flat.get(1) *  x_flat.get(1));
                x_flat.put(0, x_flat.get(0) * inv_norm);
                x_flat.put(1, x_flat.get(1) * inv_norm);

                // Iterations
                for (int iter = 0; iter < opts.num_iterations; iter++) {

                    // Input and output of a single session run.
                    StringTensorPairVector input_feed = new StringTensorPairVector(new String[] {"x"}, new Tensor[] {x});
                    StringVector output_tensor_name = new StringVector("y:0", "y_normalized:0");
                    StringVector target_tensor_name = new StringVector();
                    TensorVector outputs = new TensorVector();

                    // Run the session once
                    checkStatus(session.Run(input_feed, output_tensor_name, target_tensor_name, outputs));

                    // Get and print the output
                    assert outputs.size() == 2;
                    Tensor y = outputs.get(0);
                    Tensor y_norm = outputs.get(1);
                    System.out.println(DebugString(x, y));

                    // Copies y_normalized to x and adds the new x in the input_feed array
                    x.put(y_norm);
                }

                // Graceful closing of the session
                checkStatus(session.Close());
            }
        }
    }

    /**
     * Concert the content of the tensors to readable output
     *
     * @param x
     * @param y
     * @return
     */
    public static String DebugString(Tensor x, Tensor y) {
        assert x.NumElements() == 2;
        assert y.NumElements() == 2;
        FloatBuffer x_flat = x.createBuffer();
        FloatBuffer y_flat = y.createBuffer();
        // Compute an estimate of the eigenvalue via
        //      (x' A x) / (x' x) = (x' y) / (x' x)
        // and exploit the fact that x' x = 1 by assumption
        float lambda = x_flat.get(0) * y_flat.get(0) + x_flat.get(1) * y_flat.get(1);
        return String.format("lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]",
                lambda, x_flat.get(0), x_flat.get(1), y_flat.get(0), y_flat.get(1));
    }
}
