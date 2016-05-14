JavaCPP Presets for Caffe
=========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Caffe 1.0-rc3  http://caffe.berkeleyvision.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/caffe/apidocs/


Sample Usage
------------
Here is the main tool for training of Caffe ported to Java from this C++ source file:

 * https://github.com/BVLC/caffe/blob/master/tools/caffe.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, instead of the original `caffe` executable tool, after creating the `pom.xml` and `src/main/java/caffe.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="..."
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.caffe</groupId>
    <artifactId>caffe</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>caffe</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>caffe</artifactId>
            <version>rc3-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/caffe.java` source file
```java
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.bytedeco.javacpp.FloatPointer;
import static org.bytedeco.javacpp.caffe.*;

public class caffe {
    static final Logger logger = Logger.getLogger(caffe.class.getSimpleName());

    static String usageMessage;
    static void SetUsageMessage(String usageMessage) {
        caffe.usageMessage = usageMessage;
    }

    static abstract class Flag {
        Object value;
        String desc;
        Flag(Object value, String desc) {
            this.value = value;
            this.desc = desc;
        }
        abstract void set(String arg);
    }
    static class IntFlag extends Flag {
        IntFlag(int value, String desc) {
            super(value, desc);
        }
        @Override void set(String arg) {
            value = Integer.parseInt(arg);
        }
        @Override public String toString() {
            return ((Integer)value).toString();
        }
    }
    static class StringFlag extends Flag {
        StringFlag(String value, String desc) {
            super(value, desc);
        }
        @Override void set(String arg) {
            value = arg;
        }
        @Override public String toString() {
            return "\"" + value + "\"";
        }
    }
    static class Flags extends TreeMap<String,Flag> {
        Flags() {
            put("gpu", new IntFlag(-1,
                "Run in GPU mode on given device ID."));
            put("solver", new StringFlag("",
                "The solver definition protocol buffer text file."));
            put("model", new StringFlag("",
                "The model definition protocol buffer text file.."));
            put("snapshot", new StringFlag("",
                "Optional; the snapshot solver state to resume training."));
            put("weights", new StringFlag("",
                "Optional; the pretrained weights to initialize finetuning. "
                + "Cannot be set simultaneously with snapshot."));
            put("iterations", new IntFlag(50,
                "The number of iterations to run."));
        }
        void init(String[] args) {
            for (int i = 0; i < args.length; i++) {
                String arg = args[i], value;
                if (arg.startsWith("--")) {
                    arg = arg.substring(2);
                } else if (arg.startsWith("-")) {
                    arg = arg.substring(1);
                } else {
                    continue;
                }
                int j = arg.indexOf('=');
                if (j < 0) {
                    value = args[++i];
                } else {
                    value = arg.substring(j + 1);
                    arg = arg.substring(0, j);
                }
                Flag flag = get(arg);
                if (flag != null) {
                    flag.set(value);
                } else {
                    throw new RuntimeException("Unknown command line flag: " + arg);
                }
            }
        }
        <T> T getValue(String name) {
            return (T)(super.get(name).value);
        }
    }
    static final Flags flags = new Flags();

    static void ShowUsageWithFlags() {
        System.out.println(caffe.class.getSimpleName() + " " + usageMessage + "\n");
        System.out.println("Flags from " + caffe.class.getSimpleName() + ":");
        for (Map.Entry<String,Flag> e : flags.entrySet()) {
            Flag f = e.getValue();
            System.out.println("    -" + e.getKey() + " (" + f.desc + ") type: "
                    + f.value.getClass().getSimpleName() + " default: " + f);
        }
    }

    // A simple registry for caffe commands.
    interface BrewFunction {
        int command();
    }
    static final TreeMap<String,BrewFunction> brewMap = new TreeMap<String,BrewFunction>();

    static void RegisterBrewFunction(String name, BrewFunction func) {
        brewMap.put(name, func);
    }

    static BrewFunction GetBrewFunction(String name) {
        if (brewMap.containsKey(name)) {
            return brewMap.get(name);
        } else {
            String msg = "Available caffe actions:";
            for (String s : brewMap.keySet()) {
                msg += "\t" + s;
            }
            logger.severe(msg);
            throw new RuntimeException("Unknown action: " + name);
        }
    }

    // Load the weights from the specified caffemodel(s) into the train and test nets.
    static void CopyLayers(FloatSolver solver, String model_list) {
        String[] model_names = model_list.split(",");
        for (int i = 0; i < model_names.length; i++) {
            logger.info("Finetuning from " + model_names[i]);
            solver.net().CopyTrainedLayersFrom(model_names[i]);
            for (int j = 0; j < solver.test_nets().size(); j++) {
                solver.test_nets().get(j).CopyTrainedLayersFrom(model_names[i]);
            }
        }
    }

    static {
    // caffe commands to call by
    //     caffe <command> <args>
    //
    // To add a command, define a function "int command()" and register it with
    // RegisterBrewFunction(name, func);

    // Device Query: show diagnostic information for a GPU device.
    RegisterBrewFunction("device_query", new BrewFunction() {
        public int command() {
            Integer gpu = flags.getValue("gpu");
            if (gpu < 0) {
                throw new RuntimeException("Need a device ID to query.");
            }
            logger.info("Querying device ID = " + gpu);
            Caffe.SetDevice(gpu);
            Caffe.DeviceQuery();
            return 0;
        }
    });

    // Train / Finetune a model.
    RegisterBrewFunction("train", new BrewFunction() {
        public int command() {
            Integer gpu = flags.getValue("gpu");
            String solverFlag = flags.getValue("solver");
            String snapshot = flags.getValue("snapshot");
            String weights = flags.getValue("weights");
            if (solverFlag.length() == 0) {
                throw new RuntimeException("Need a solver definition to train.");
            }
            if (snapshot.length() > 0 && weights.length() > 0) {
                throw new RuntimeException(
                        "Give a snapshot to resume training or weights to finetune "
                      + "but not both.");
            }

            SolverParameter solver_param = new SolverParameter();
            ReadProtoFromTextFileOrDie(solverFlag, solver_param);

            // If the gpu flag is not provided, allow the mode and device to be set
            // in the solver prototxt.
            if (gpu < 0 && solver_param.solver_mode() == SolverParameter_SolverMode_GPU) {
                gpu = solver_param.device_id();
            }

            // Set device id and mode
            if (gpu >= 0) {
                logger.info("Use GPU with device ID " + gpu);
                Caffe.SetDevice(gpu);
                Caffe.set_mode(Caffe.GPU);
            } else {
                logger.info("Use CPU.");
                Caffe.set_mode(Caffe.CPU);
            }

            logger.info("Starting Optimization");
            FloatSolver solver = FloatSolverRegistry.CreateSolver(solver_param);

            if (snapshot.length() > 0) {
                logger.info("Resuming from " + snapshot);
                solver.Solve(snapshot);
            } else if (weights.length() > 0) {
                CopyLayers(solver, weights);
                solver.Solve();
            } else {
                solver.Solve();
            }
            logger.info("Optimization Done.");
            return 0;
        }
    });

    // Test: score a model.
    RegisterBrewFunction("test", new BrewFunction() {
        public int command() {
            Integer gpu = flags.getValue("gpu");
            String model = flags.getValue("model");
            String weights = flags.getValue("weights");
            Integer iterations = flags.getValue("iterations");
            if (model.length() == 0) {
                throw new RuntimeException("Need a model definition to score.");
            }
            if (weights.length() == 0) {
                throw new RuntimeException("Need model weights to score.");
            }

            // Set device id and mode
            if (gpu >= 0) {
                logger.info("Use GPU with device ID " + gpu);
                Caffe.SetDevice(gpu);
                Caffe.set_mode(Caffe.GPU);
            } else {
                logger.info("Use CPU.");
                Caffe.set_mode(Caffe.CPU);
            }
            // Instantiate the caffe net.
            FloatNet caffe_net = new FloatNet(model, TEST);
            caffe_net.CopyTrainedLayersFrom(weights);
            logger.info("Running for " + iterations + " iterations.");

            FloatBlobVector bottom_vec = new FloatBlobVector();
            ArrayList<Integer> test_score_output_id = new ArrayList<Integer>();
            ArrayList<Float> test_score = new ArrayList<Float>();
            float loss = 0;
            for (int i = 0; i < iterations; i++) {
                float[] iter_loss = new float[1];
                FloatBlobVector result = caffe_net.Forward(bottom_vec, iter_loss);
                loss += iter_loss[0];
                int idx = 0;
                for (int j = 0; j < result.size(); j++) {
                    FloatPointer result_vec = result.get(j).cpu_data();
                    for (int k = 0; k < result.get(j).count(); k++, idx++) {
                        float score = result_vec.get(k);
                        if (i == 0) {
                            test_score.add(score);
                            test_score_output_id.add(j);
                        } else {
                            test_score.set(idx, test_score.get(idx) + score);
                        }
                        String output_name = caffe_net.blob_names().get(
                                caffe_net.output_blob_indices().get(j)).getString();
                        logger.info("Batch " + i + ", " + output_name + " = " + score);
                    }
                }
            }
            loss /= iterations;
            logger.info("Loss: " + loss);
            for (int i = 0; i < test_score.size(); i++) {
                String output_name = caffe_net.blob_names().get(
                        caffe_net.output_blob_indices().get(test_score_output_id.get(i))).getString();
                float loss_weight =
                        caffe_net.blob_loss_weights().get(caffe_net.output_blob_indices().get(i));
                String loss_msg_stream = "";
                float mean_score = test_score.get(i) / iterations;
                if (loss_weight != 0) {
                    loss_msg_stream = " (* " + loss_weight
                                    + " = " + (loss_weight * mean_score) + " loss)";
                }
                logger.info(output_name + " = " + mean_score + loss_msg_stream);
            }
            return 0;
        }
    });

    // Time: benchmark the execution time of a model.
    RegisterBrewFunction("time", new BrewFunction() {
        public int command() {
            Integer gpu = flags.getValue("gpu");
            String model = flags.getValue("model");
            Integer iterations = flags.getValue("iterations");
            if (model.length() == 0) {
                throw new RuntimeException("Need a model definition to time.");
            }

            // Set device id and mode
            if (gpu >= 0) {
                logger.info("Use GPU with device ID " + gpu);
                Caffe.SetDevice(gpu);
                Caffe.set_mode(Caffe.GPU);
            } else {
                logger.info("Use CPU.");
                Caffe.set_mode(Caffe.CPU);
            }
            // Instantiate the caffe net.
            FloatNet caffe_net = new FloatNet(model, TRAIN);

            // Do a clean forward and backward pass, so that memory allocation are done
            // and future iterations will be more stable.
            logger.info("Performing Forward");
            // Note that for the speed benchmark, we will assume that the network does
            // not take any input blobs.
            float[] initial_loss = new float[1];
            caffe_net.Forward(new FloatBlobVector(), initial_loss);
            logger.info("Initial loss: " + initial_loss[0]);
            logger.info("Performing Backward");
            caffe_net.Backward();

            FloatLayerSharedVector layers = caffe_net.layers();
            FloatBlobVectorVector bottom_vecs = caffe_net.bottom_vecs();
            FloatBlobVectorVector top_vecs = caffe_net.top_vecs();
            BoolVectorVector bottom_need_backward = caffe_net.bottom_need_backward();
            logger.info("*** Benchmark begins ***");
            logger.info("Testing for " + iterations + " iterations.");
            Timer total_timer = new Timer();
            total_timer.Start();
            Timer forward_timer = new Timer();
            Timer backward_timer = new Timer();
            Timer timer = new Timer();
            double[] forward_time_per_layer = new double[(int)layers.size()];
            double[] backward_time_per_layer = new double[(int)layers.size()];
            double forward_time = 0.0;
            double backward_time = 0.0;
            for (int j = 0; j < iterations; j++) {
                Timer iter_timer = new Timer();
                iter_timer.Start();
                forward_timer.Start();
                for (int i = 0; i < layers.size(); i++) {
                    timer.Start();
                    // Although Reshape should be essentially free, we include it here
                    // so that we will notice Reshape performance bugs.
                    layers.get(i).Reshape(bottom_vecs.get(i), top_vecs.get(i));
                    layers.get(i).Forward(bottom_vecs.get(i), top_vecs.get(i));
                    forward_time_per_layer[i] += timer.MicroSeconds();
                }
                forward_time += forward_timer.MicroSeconds();
                backward_timer.Start();
                for (int i = (int)layers.size() - 1; i >= 0; i--) {
                    timer.Start();
                    layers.get(i).Backward(top_vecs.get(i), bottom_need_backward.get(i), bottom_vecs.get(i));
                    backward_time_per_layer[i] += timer.MicroSeconds();
                }
                backward_time += backward_timer.MicroSeconds();
                logger.info("Iteration: " + (j + 1) + " forward-backward time: "
                        + iter_timer.MilliSeconds() + " ms.");
            }
            logger.info("Average time per layer: ");
            for (int i = 0; i < layers.size(); ++i) {
                String layername = layers.get(i).layer_param().name().getString();
                logger.info(layername + "\tforward: "
                        + String.format("%10g ms.", forward_time_per_layer[i] / 1000 / iterations));
                logger.info(layername + "\tbackward: "
                        + String.format("%10g ms.", backward_time_per_layer[i] / 1000 / iterations));
            }
            total_timer.Stop();
            logger.info("Average Forward pass: " + forward_time / 1000 / iterations + " ms.");
            logger.info("Average Backward pass: " + backward_time / 1000 / iterations + " ms.");
            logger.info("Average Forward-Backward: " + total_timer.MilliSeconds() / iterations + " ms.");
            logger.info("Total Time: " + total_timer.MilliSeconds() + " ms.");
            logger.info("*** Benchmark ends ***");
            return 0;
        }
    });

    }

    public static void main(String[] args) {
        // Print output to stderr (while still logging).
        logger.setLevel(Level.ALL);
        // Usage message.
        SetUsageMessage("command line brew\n"
              + "usage: caffe <command> <args>\n\n"
              + "commands:\n"
              + "  train           train or finetune a model\n"
              + "  test            score a model\n"
              + "  device_query    show GPU diagnostic information\n"
              + "  time            benchmark model execution time");
        // Run tool or show usage.
        flags.init(args);
        //GlobalInit(args);
        if (args.length > 0) {
            System.exit(GetBrewFunction(args[0]).command());
        } else {
            ShowUsageWithFlags();
        }
    }
}
```
