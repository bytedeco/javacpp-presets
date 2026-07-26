package org.bytedeco.pytorch.rl;

import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * 轨迹容器：用于暂存一个 Episode 内的所有交互数据
 * 方便在 Episode 结束后计算 GAE (优势函数) 和 Returns (回报)
 */
public class Trajectory {
    // 使用 List 存储，因为每个 Episode 的步数是不固定的
    public List<Tensor> states = new ArrayList<>();
    public List<Tensor> actions = new ArrayList<>();
    public List<Tensor> logProbs = new ArrayList<>();
    public List<Float> rewards = new ArrayList<>();
    public List<Float> values = new ArrayList<>();
    public List<Boolean> masks = new ArrayList<>();

    /** 
     * 添加一步交互数据
     * 注意：必须进行 .detach().clone()，否则当 PointerScope 关闭时，这些 Tensor 会失效
     */
    public void add(Tensor s, Tensor a, Tensor lp, float r, Tensor v, boolean m) {
        states.add(s.detach().clone());
        actions.add(a.detach().clone());
        logProbs.add(lp.detach().clone());
        rewards.add(r);
        values.add(v.item_float()); // 价值 V(s) 存 float 即可，节省内存
        masks.add(m);               // 标记是否结束
    }

    public int size() {
        return rewards.size();
    }

    /**
     * 清空轨迹，为下一个 Episode 做准备
     */
    public void clear() {
        states.clear();
        actions.clear();
        logProbs.clear();
        rewards.clear();
        values.clear();
        masks.clear();
    }
}
