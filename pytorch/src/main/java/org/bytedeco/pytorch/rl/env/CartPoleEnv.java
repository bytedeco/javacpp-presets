package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.rl.StepResult;

import java.util.Random;

public class CartPoleEnv implements Env {
    private double gravity = 9.8;
    private double masscart = 1.0;
    private double masspole = 0.1;
    private double length = 0.5; // 杆长的一半
    private double force_mag = 10.0;
    private double tau = 0.02; // 每步时间间隔

    private float[] state = new float[4]; // [x, x_dot, theta, theta_dot]

    public CartPoleEnv() { reset(); }

    public float[] reset() {
        Random r = new Random();
        for(int i=0; i<4; i++) state[i] = (r.nextFloat() - 0.5f) * 0.1f;
        return state;
    }

    public StepResult step(int action) {
        float x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = Math.cos(theta);
        double sintheta = Math.sin(theta);

        double temp = (force + masspole * length * theta_dot * theta_dot * sintheta) / (masscart + masspole);
        double theta_acc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * costheta * costheta / (masscart + masspole)));
        double x_acc = temp - masspole * length * theta_acc * costheta / (masscart + masspole);

        state[0] += x_dot * tau;
        state[1] += x_acc * tau;
        state[2] += theta_dot * tau;
        state[3] += theta_acc * tau;

        boolean done = state[0] < -2.4 || state[0] > 2.4 || state[2] < -0.209 || state[2] > 0.209;
        float reward = done ? 0.0f : 1.0f;
        return new StepResult(state.clone(), reward, done);
    }
}