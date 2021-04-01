// © Copyright 2019, Torch Contributors.

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;

public class SimpleMNIST {

    // Define a new Module.
    static class Net extends Module {
        Net() {
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", new LinearImpl(784, 64));
            fc2 = register_module("fc2", new LinearImpl(64, 32));
            fc3 = register_module("fc3", new LinearImpl(32, 10));
        }

        // Implement the Net's algorithm.
        Tensor forward(Tensor x) {
            // Use one of many tensor manipulation functions.
            x = relu(fc1.forward(x.reshape(x.size(0), 784)));
            x = dropout(x, /*p=*/0.5, /*train=*/is_training());
            x = relu(fc2.forward(x));
            x = log_softmax(fc3.forward(x), /*dim=*/1);
            return x;
        }

        // Use one of many "standard library" modules.
        LinearImpl fc1 = null, fc2 = null, fc3 = null;
    }

    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        // Create a new Net.
        Net net = new Net();

        // Create a multi-threaded data loader for the MNIST dataset.
        MNISTMapDataset data_set = new MNIST("./data").map(new ExampleStack());
        MNISTRandomDataLoader data_loader = new MNISTRandomDataLoader(
                data_set, new RandomSampler(data_set.size().get()),
                new DataLoaderOptions(/*batch_size=*/64));

        // Instantiate an SGD optimization algorithm to update our Net's parameters.
        SGD optimizer = new SGD(net.parameters(), new SGDOptions(/*lr=*/0.01));

        for (int epoch = 1; epoch <= 10; ++epoch) {
            int batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (ExampleIterator it = data_loader.begin(); !it.equals(data_loader.end()); it = it.increment()) {
                Example batch = it.access();
                // Reset gradients.
                optimizer.zero_grad();
                // Execute the model on the input data.
                Tensor prediction = net.forward(batch.data());
                // Compute a loss value to judge the prediction of our model.
                Tensor loss = nll_loss(prediction, batch.target());
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                // Output the loss and checkpoint every 100 batches.
                if (++batch_index % 100 == 0) {
                    System.out.println("Epoch: " + epoch + " | Batch: " + batch_index
                                     + " | Loss: " + loss.item_float());
                    // Serialize your model periodically as a checkpoint.
                    OutputArchive archive = new OutputArchive();
                    net.save(archive);
                    archive.save_to("net.pt");
                }
            }
        }
    }
}
