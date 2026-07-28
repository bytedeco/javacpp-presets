/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * Lightweight SLIC-style superpixel graph (pure Tensor, no OpenCV).
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.MemoryFormatOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Cluster pixels ({@code x}=color, {@code pos}=xy) into {@code k} superpixels
 * and replace {@code data} with the coarse graph (KNN edges).
 */
public class ToSLIC implements BaseTransform {
    private final int k;
    private final float m;

    public ToSLIC(int k, float m) {
        if (k <= 0) throw new IllegalArgumentException("k > 0");
        this.k = k;
        this.m = m;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor x = TransformUtils.requireX(data);
        Tensor pos = TransformUtils.requirePos(data);
        long numPixels = x.size(0);
        int kk = (int) Math.min(k, numPixels);
        float s = (float) Math.sqrt(numPixels / (double) Math.max(kk, 1));

        Tensor indices = randperm(numPixels, TransformUtils.longOptsLike(x))
                .slice(0, new LongOptional(0), new LongOptional(kk), 1);
        Tensor centersPos = pos.index_select(0, indices).clone();
        Tensor centersCol = x.index_select(0, indices).clone();
        Tensor labels = zeros(new long[]{numPixels}, TransformUtils.longOptsLike(x));

        for (int iter = 0; iter < 5; iter++) {
            Tensor dSpace = pos.unsqueeze(1).sub(centersPos.unsqueeze(0)).pow(new Scalar(2)).sum(2);
            Tensor dColor = x.unsqueeze(1).sub(centersCol.unsqueeze(0)).pow(new Scalar(2)).sum(2);
            Tensor dists = dColor.add(dSpace.mul(new Scalar(Math.pow(m / Math.max(s, 1e-6), 2))));
            labels = dists.argmin(new LongOptional(1), false);
            centersPos = scatterMean(pos, labels, kk);
            centersCol = scatterMean(x, labels, kk);
        }

        // mutate data in place (demo asserts on same instance)
        data.x = centersCol;
        data.pos = centersPos;
        data.edge_index = computeKNN(centersPos, Math.min(6, kk - 1));
        return data;
    }

    static Tensor computeKNN(Tensor pos, int k) {
        long n = pos.size(0);
        int actualK = (int) Math.min(Math.max(k, 0), Math.max(0, n - 1));
        if (actualK <= 0) {
            return empty(new long[]{2, 0},
                    pos.options().dtype(new ScalarTypeOptional(kLong())),
                    new MemoryFormatOptional());
        }
        Tensor distInner = pos.pow(new Scalar(2)).sum(new long[]{1}, true, new ScalarTypeOptional());
        Tensor distMat = distInner.add(distInner.t())
                .addmm(pos, pos.t(), new Scalar(1.0), new Scalar(-2.0));
        T_TensorTensor_T topkResult = distMat.neg().topk(actualK + 1, 1, true, true);
        Tensor indices = topkResult.get1()
                .slice(1, new LongOptional(1), new LongOptional(actualK + 1), 1);
        Tensor row = arange(new Scalar(n), TransformUtils.longOptsLike(pos))
                .view(-1, 1).expand(new long[]{n, actualK}).reshape(-1);
        Tensor col = indices.reshape(-1);
        return stack(new TensorVector(row, col), 0);
    }

    private static Tensor scatterMean(Tensor src, Tensor index, int numClusters) {
        long featDim = src.size(1);
        Tensor out = zeros(new long[]{numClusters, featDim}, src.options());
        Tensor count = zeros(new long[]{numClusters, 1}, src.options());
        Tensor expandIdx = index.unsqueeze(1).expand_as(src);
        out.scatter_add_(0, expandIdx, src);
        count.scatter_add_(0, index.unsqueeze(1), ones(new long[]{src.size(0), 1}, src.options()));
        return out.div(count.add(new Scalar(1e-7)));
    }
}
