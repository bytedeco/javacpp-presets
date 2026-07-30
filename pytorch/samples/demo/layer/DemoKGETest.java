package samples.demo.layer;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.kge.ComplEx;
import org.bytedeco.pytorch.geometric.nn.kge.DistMult;
import org.bytedeco.pytorch.geometric.nn.kge.RotatE;
import org.bytedeco.pytorch.geometric.nn.kge.TransE;

import java.util.Arrays;

public class DemoKGETest {

   public static void main(String[] args) {
        long numEntities = 10;
        long numRelations = 5;
        long hiddenDim = 16;
        double epsilon = 0.001;

        // 构造模拟输入: 3个三元组的索引 (head, relation, tail)
        Tensor h = torch.tensor(new long[]{0, 2, 5}, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
        Tensor r = torch.tensor(new long[]{1, 3, 4}, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));
        Tensor t = torch.tensor(new long[]{2, 5, 0}, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        System.out.println("=== Testing KGE Models ===");

        // 1. TransE 测试
        testTransE(numEntities, numRelations, hiddenDim, h, r, t);

        // 2. DistMult 测试
        testDistMult(numEntities, numRelations, hiddenDim, h, r, t);

        // 3. ComplEx 测试
        testComplEx(numEntities, numRelations, hiddenDim, h, r, t);

        // 4. RotatE 测试
        testRotatE(numEntities, numRelations, hiddenDim, epsilon, h, r, t);
    }

    // --- TransE: f = -||h + r - t|| ---
    public static void testTransE(long numEnt, long numRel, long dim, Tensor h, Tensor r, Tensor t) {
        System.out.println("\n[TransE Test]");
        TransE model = new TransE(numEnt, numRel, dim, 1L); // p=1.0 (L1 norm)
        Tensor score = model.forward(h, r, t);
        System.out.println("Score Shape: " + Arrays.toString(score.shape())); // [3]

        // 验证梯度
        score.sum().backward();
        System.out.println("Backward Pass: SUCCESS");
    }

    // --- DistMult: f = <h, r, t> (双线性) ---
    public static void testDistMult(long numEnt, long numRel, long dim, Tensor h, Tensor r, Tensor t) {
        System.out.println("\n[DistMult Test]");
        DistMult model = new DistMult(numEnt, numRel, dim);
        Tensor score = model.forward(h, r, t);
        System.out.println("Score Shape: " + Arrays.toString(score.shape()));

        score.sum().backward();
        System.out.println("Backward Pass: SUCCESS");
    }

    // --- ComplEx: f = Re(<h, r, conj(t)>) ---
    public static void testComplEx(long numEnt, long numRel, long dim, Tensor h, Tensor r, Tensor t) {
        System.out.println("\n[ComplEx Test]");
        ComplEx model = new ComplEx(numEnt, numRel, dim);
        Tensor score = model.forward(h, r, t);
        System.out.println("Score Shape: " + Arrays.toString(score.shape()));

        score.sum().backward();
        System.out.println("Backward Pass: SUCCESS");
    }

    // --- RotatE: f = -||h * r - t|| (复数旋转) ---
    public static void testRotatE(long numEnt, long numRel, long dim, double epsilon, Tensor h, Tensor r, Tensor t) {
        System.out.println("\n[RotatE Test]");
        RotatE model = new RotatE(numEnt, numRel, dim, epsilon);
        Tensor score = model.forward(h, r, t);
        System.out.println("Score Shape: " + Arrays.toString(score.shape()));

        score.sum().backward();
        System.out.println("Backward Pass: SUCCESS");
    }
}