package org.bytedeco.javacpp.samples.tensorflow;

import static org.bytedeco.javacpp.tensorflow.Const;
import static org.bytedeco.javacpp.tensorflow.InitMain;
import static org.bytedeco.javacpp.tensorflow.TF_CHECK_OK;

import java.nio.IntBuffer;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.tensorflow.AddN;
import org.bytedeco.javacpp.tensorflow.GraphDef;
import org.bytedeco.javacpp.tensorflow.Input;
import org.bytedeco.javacpp.tensorflow.InputList;
import org.bytedeco.javacpp.tensorflow.L2Loss;
import org.bytedeco.javacpp.tensorflow.Output;
import org.bytedeco.javacpp.tensorflow.OutputVector;
import org.bytedeco.javacpp.tensorflow.Scope;
import org.bytedeco.javacpp.tensorflow.Session;
import org.bytedeco.javacpp.tensorflow.SessionOptions;
import org.bytedeco.javacpp.tensorflow.StringTensorPairVector;
import org.bytedeco.javacpp.tensorflow.StringVector;
import org.bytedeco.javacpp.tensorflow.Tensor;
import org.bytedeco.javacpp.tensorflow.TensorShape;
import org.bytedeco.javacpp.tensorflow.TensorVector;

/**
 * Showcase the usage of OutputVector and the AddN operator.
 *
 * @author Nico Hezel
 */
public class AddNExample {

    public static void main(String[] args) {

        // Load all javacpp-preset classes and native libraries
        Loader.load(org.bytedeco.javacpp.tensorflow.class);

        // Platform-specific initialization routine
        InitMain("trainer", (int[])null, null);

        // Create a new empty graph
        Scope scope = Scope.NewRootScope();

        // (2,1) matrix of ones, sixes and tens
        TensorShape shape = new TensorShape(2, 1);
        Output ones = Const(scope.WithOpName("ones"), 1, shape);
        Output sixes = Const(scope.WithOpName("sixes"), 6, shape);
        Output tens = Const(scope.WithOpName("tens"), 10, shape);

        // Adding all matrices element-wise
        OutputVector ov = new OutputVector(ones, sixes, tens);
        InputList inputList = new InputList(ov);
        AddN add = new AddN(scope.WithOpName("add"), inputList);

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
            StringVector output_tensor_name = new StringVector("add:0");
            StringVector target_tensor_name = new StringVector();
            TensorVector outputs = new TensorVector();

            // Run the session once
            TF_CHECK_OK(session.Run(input_feed, output_tensor_name, target_tensor_name, outputs));

            // Print the add-output
            for (Tensor output : outputs.get()) {
                IntBuffer y_flat = output.createBuffer();
                for (int i = 0; i < output.NumElements(); i++)
                    System.out.println(y_flat.get(i));
            }
        }
    }
}
