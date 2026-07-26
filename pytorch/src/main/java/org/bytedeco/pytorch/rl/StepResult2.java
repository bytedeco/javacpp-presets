package org.bytedeco.pytorch.rl;

import org.bytedeco.pytorch.Tensor;

public record StepResult2(Tensor observation, double reward, boolean done) {}