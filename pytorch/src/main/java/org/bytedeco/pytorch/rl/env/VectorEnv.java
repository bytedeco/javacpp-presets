package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.rl.StepResult;

import java.util.stream.IntStream;

import static org.bytedeco.pytorch.global.torch.tensor;

public class VectorEnv {
    private final CartPoleEnv[] envs;
    private final int numEnvs;
    private final float[][] currentObs;
//    private final ExecutorService threadPool;

    public int getNumEnvs() { return numEnvs; }
    
    public VectorEnv(int numEnvs) {
        this.numEnvs = numEnvs;
        this.envs = new CartPoleEnv[numEnvs];
        this.currentObs = new float[numEnvs][];
        for (int i = 0; i < numEnvs; i++) {
            envs[i] = new CartPoleEnv();
            currentObs[i] = envs[i].reset();
        }
    }



    // 将所有环境的当前观测值打包成一个 [N, state_dim] 的 Tensor
    public Tensor getStackedObs() {
        // 将 float[][] 转换为一维数组，方便一次性创建 Tensor
        int obsDim = currentObs[0].length;
        float[] flatObs = new float[numEnvs * obsDim];
        for (int i = 0; i < numEnvs; i++) {
            System.arraycopy(currentObs[i], 0, flatObs, i * obsDim, obsDim);
        }
        return tensor(flatObs).reshape(numEnvs, obsDim);
    }

    // 并行执行一步
    public StepResult[] step(int[] actions) {
        return IntStream.range(0, numEnvs)
                .parallel() // 并行执行环境步进
                .mapToObj(i -> {
                    StepResult res = envs[i].step(actions[i]);
                    if (res.done) {
                        currentObs[i] = envs[i].reset(); // 自动重置环境
                    } else {
                        currentObs[i] = res.nextState;
                    }
                    return res;
                }).toArray(StepResult[]::new);
    }
}