import java.nio.IntBuffer;

import org.bytedeco.javacpp.*;
import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

/**
 * Showcase the usage of InputList and the concat operator.
 *
 * @author Nico Hezel
 *
 */
public class InputListExample {

    public static void main(String[] args) throws Exception {

        // Platform-specific initialization routine
        InitMain("trainer", (int[])null, null);

        // Create a new empty graph
        Scope scope = Scope.NewRootScope();

        // (3,2) matrix of ones and sixes
        TensorShape shape = new TensorShape(3, 2);
        Output ones = Const(scope.WithOpName("ones"), 1, shape);
        Output sixes = Const(scope.WithOpName("sixes"), 6, shape);

        // Vertical concatenation of those matrices
        OutputVector ov = new OutputVector(ones, sixes);
        InputList inputList = new InputList(ov);
        Input axis = new Input(Const(scope.WithOpName("axis"), 0));
        Concat concat = new Concat(scope.WithOpName("concat"), inputList, axis);

        // Build a graph definition object
        GraphDef def = new GraphDef();
        TF_CHECK_OK(scope.ToGraphDef(def));

        // Creates a session.
        SessionOptions options = new SessionOptions();
        try(final Session session = new Session(options)) {

            // Create the graph to be used for the session.
            TF_CHECK_OK(session.Create(def));

            // Input and output of a single session run.
            StringTensorPairVector input_feed = new StringTensorPairVector();
            StringVector output_tensor_name = new StringVector("concat:0");
            StringVector target_tensor_name = new StringVector();
            TensorVector outputs = new TensorVector();

            // Run the session once
            TF_CHECK_OK(session.Run(input_feed, output_tensor_name, target_tensor_name, outputs));

            // Print the concatenation output
            for (Tensor output : outputs.get()) {
                IntBuffer y_flat = output.createBuffer();
                for (int i = 0; i < output.NumElements(); i++)
                    System.out.println(y_flat.get(i));
            }
        }
    }
}
