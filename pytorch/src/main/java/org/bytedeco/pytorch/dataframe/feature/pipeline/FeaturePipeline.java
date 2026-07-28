package org.bytedeco.pytorch.dataframe.feature.pipeline;

 import org.bytedeco.pytorch.dataframe.DataFrame;
 import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * 特征工程流水线（修复：逐次拟合+转换，确保列名传递正确）
 */
public class FeaturePipeline implements Serializable {
    private static final long serialVersionUID = 1L;
    private List<BaseTransformer> transformers = new ArrayList<>();

    // 添加转换器到流水线
    public FeaturePipeline addTransformer(BaseTransformer transformer) {
        transformers.add(transformer);
        return this;
    }

    // 修复：fit方法（逐次拟合，每次用最新的转换后数据）
    public FeaturePipeline fit(DataFrame df) throws Exception {
        DataFrame current = df.copy();
        for (BaseTransformer transformer : transformers) {
            try {
                // 先拟合当前转换器（基于最新数据）
                transformer.fit(current);
                // 转换后更新current，供下一个转换器使用
                current = transformer.transform(current);
            } catch (Exception e) {
                throw new RuntimeException("流水线拟合失败：" + transformer.getClass().getSimpleName() + "，原因：" + e.getMessage(), e);
            }
        }
        return this;
    }

    // 修复：transform方法（逐次转换）
    public DataFrame transform(DataFrame df) throws Exception {
        DataFrame current = df.copy();
        for (BaseTransformer transformer : transformers) {
            if (!transformer.isFitted()) {
                throw new IllegalStateException(transformer.getClass().getSimpleName() + "未拟合！");
            }
            current = transformer.transform(current);
        }
        return current;
    }

    // 拟合并转换
    public DataFrame fitTransform(DataFrame df) throws Exception {
        return this.fit(df).transform(df);
    }

    // 保存流水线（可序列化）
    public void save(String filepath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(this);
        }
        System.out.println("✅ 特征流水线已保存到: " + filepath);
    }

    // 加载流水线
    public static FeaturePipeline load(String filepath) throws IOException, ClassNotFoundException {
        File file = new File(filepath);
        if (!file.exists()) {
            throw new FileNotFoundException("流水线文件不存在: " + filepath);
        }
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            FeaturePipeline pipeline = (FeaturePipeline) ois.readObject();
            System.out.println("✅ 特征流水线已从 " + filepath + " 加载");
            return pipeline;
        }
    }

    // 获取流水线中的转换器
    public List<BaseTransformer> getTransformers() {
        return new ArrayList<>(transformers);
    }
}