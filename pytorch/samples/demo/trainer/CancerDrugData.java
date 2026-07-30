package samples.demo.trainer;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;

import static org.bytedeco.pytorch.global.torch.*;

public class CancerDrugData {
    public static DrugTargetPair generateMockData() {
        DrugTargetPair pair = new DrugTargetPair();
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        // 模拟一个小分子 (如伊马替尼/Imatinib 结构简化版)
        pair.drugX = randn(new long[]{20, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        pair.drugedge_index = randint(0, 20, new long[]{2, 40}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        // 模拟 BCR-ABL 蛋白激酶序列 (长度 1000, 20 种氨基酸编码)
        pair.proteinSeq = rand(new long[]{1, 20, 1000}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        // 根据药物和蛋白质特征生成有意义的亲和力值 (5-10之间)
        // 通过特征均值生成一定的关联性
        float drugMean = pair.drugX.mean().item().toFloat();
        float proteinMean = pair.proteinSeq.mean().item().toFloat();
        float baseAffinity = 7.5f + (drugMean + proteinMean) * 0.5f;
        pair.affinity = tensor(new float[]{baseAffinity}, floatOpts);
//        pair.affinity = rand(new long[]{16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));//.multiply( new Scalar(5.0)).add(new Scalar(5.0)); 
//        pair.affinity = tensor(new float[]{7.5f}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))); // 亲和力 10^-7.5 M
        return pair;
    }

    public static class DrugTargetPair {
        public Tensor drugX;           // 原子特征 (原子序数, 杂化类型等)
        public Tensor drugedge_index;    // 化学键 (单键, 双键, 芳香键)
        public Tensor proteinSeq;      // 蛋白质特征 (One-hot 氨基酸)
        public Tensor affinity;        // 标签：亲和力 (IC50 转换值)
    }

//    public static DrugTargetPair generateMockData2() {
//        DrugTargetPair pair = new DrugTargetPair();
//        // 模拟一个小分子 (固定结构，添加少量噪声)
//        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
//        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));
//
//        // 固定药物结构，添加少量噪声
//        pair.drugX = randn(new long[]{20, 16}, floatOpts).multiply(new Scalar(0.1));
//        var drugEI1 = tensor(new long[]{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 0},longOpts);
//        var drugEI2 = tensor(new long[]{1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19, 18, 0, 19}, longOpts);
//        
//        pair.drugedge_index = tensor(new Tensor[][]{drugEI1, drugEI2}, longOpts);
//
//        // 模拟蛋白质序列 (固定特征，添加少量噪声)
//        pair.proteinSeq = randn(new long[]{1, 20, 1000}, floatOpts).multiply(new Scalar(0.1));
//
//        // 根据药物和蛋白质特征生成有意义的亲和力值 (5-10之间)
//        // 通过特征均值生成一定的关联性
//        float drugMean = pair.drugX.mean().item().toFloat();
//        float proteinMean = pair.proteinSeq.mean().item().toFloat();
//        float baseAffinity = 7.5f + (drugMean + proteinMean) * 0.5f;
//        pair.affinity = tensor(new float[]{baseAffinity}, floatOpts);
//
//        return pair;
//    }
}
