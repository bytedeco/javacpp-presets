package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.rl.StepResult;

public interface Env {
    float[] reset();
    StepResult step(int action);
}