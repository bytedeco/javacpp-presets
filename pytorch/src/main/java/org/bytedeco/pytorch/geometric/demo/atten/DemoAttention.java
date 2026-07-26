package org.bytedeco.pytorch.geometric.demo.atten;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.attention.PerformerAttention;
import org.bytedeco.pytorch.geometric.attention.PolynormerAttention;
import org.bytedeco.pytorch.geometric.attention.QFormer;
import org.bytedeco.pytorch.geometric.attention.SGFormerAttention;

import java.util.Arrays;

public class DemoAttention {

    public static void main(String[] args) {
        System.out.println("=== Testing Advanced Attention Modules ===");

        // CUDA Check
//        Device device = torch_cuda.is_available() ? new Device("cuda") : new Device("cpu");
//        System.out.println("Using Device: " + (device.is_cuda() ? "CUDA" : "CPU"));

        if (!torch.hasCUDA()) {
            System.out.println("===(CUDA) not support ===");
//            throw new RuntimeException("Need CUDA for this enterprise demo!");
        }
        Device device = new Device("mps");// new Device("cuda");
        long N = 100; // Nodes
        long D = 64;  // Dim
        long Heads = 4;

        Tensor x = torch.randn(new long[]{N, D}).to(device,torch.ScalarType.Float);

        try (PointerScope scope = new PointerScope()) {

            //2. Q-Former
            System.out.println("\n[Q-Former]");
            long numQueries = 10;
            QFormer qformer = new QFormer(D, Heads, numQueries);
            qformer.to(device,true);
            Tensor outQF = qformer.forward(x); // Expect [10, 64]
            System.out.println("Output: " + Arrays.toString(outQF.shape()));
            if (outQF.size(0) == numQueries) System.out.println("PASS: Query dimension matches.");

            // 1. Performer
            System.out.println("\n[Performer Attention]");
            PerformerAttention performer = new PerformerAttention(D, Heads, 16); // 16 random features
            performer.to(device, true);
            Tensor outPerf = performer.forward(x);
            System.out.println("Output: " + Arrays.toString(outPerf.shape()));


            // 3. Polynormer (Linear)
            System.out.println("\n[Polynormer / Linear Attention]");
            PolynormerAttention poly = new PolynormerAttention(D, Heads);
            poly.to(device,true);
            Tensor outPoly = poly.forward(x);
            System.out.println("Output: " + Arrays.toString(outPoly.shape()));

            // 4. SGFormer
            System.out.println("\n[SGFormer Attention]");
            SGFormerAttention sg = new SGFormerAttention(D, Heads);
            sg.to(device,true);
            Tensor outSG = sg.forward(x);
            System.out.println("Output: " + Arrays.toString(outSG.shape()));
        }
    }
}