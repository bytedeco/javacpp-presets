/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * LapPE / RWPE / LambdaMax
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/** Spectral position encodings and related utilities. */
public final class SpectralAndStructuralTransforms {
    private SpectralAndStructuralTransforms() {}

    /** Append k non-trivial Laplacian eigenvectors to {@code x}. */
    public static class AddLaplacianEigenvectorPE implements BaseTransform {
        private final int k;
        public AddLaplacianEigenvectorPE(int k) {
            if (k <= 0) throw new IllegalArgumentException("k > 0");
            this.k = k;
        }

        @Override
        public GraphData apply(GraphData data) {
            TransformUtils.requireX(data);
            data = new AdvancedStructuralTransforms.ToDense().apply(data);
            Tensor adj = data.adj;
            long n = adj.size(0);
            int kk = (int) Math.min(k, Math.max(0, n - 1));
            Tensor deg = adj.sum(1);
            Tensor L = diag(deg).sub(adj);
            T_TensorTensor_T eig = linalg_eigh(L);
            Tensor eigVecs = eig.get1();
            Tensor pe = eigVecs.narrow(1, 1, kk);
            data.x = cat(new TensorVector(data.x, pe.to(data.x.dtype())), 1);
            return data;
        }
    }

    /** Append {@code walkSteps} random-walk return probabilities to {@code x}. */
    public static class AddRandomWalkPE implements BaseTransform {
        private final int walkSteps;
        public AddRandomWalkPE(int walkSteps) {
            if (walkSteps <= 0) throw new IllegalArgumentException("walkSteps > 0");
            this.walkSteps = walkSteps;
        }

        @Override
        public GraphData apply(GraphData data) {
            TransformUtils.requireX(data);
            data = new AdvancedStructuralTransforms.ToDense().apply(data);
            Tensor adj = data.adj;
            Tensor degInv = adj.sum(1).pow(new Scalar(-1));
            degInv.masked_fill_(degInv.isinf(), new Scalar(0));
            Tensor P = diag(degInv).matmul(adj);

            java.util.List<Tensor> peList = new java.util.ArrayList<>();
            Tensor pk = P.clone();
            for (int i = 0; i < walkSteps; i++) {
                peList.add(pk.diagonal().view(-1, 1));
                if (i + 1 < walkSteps) {
                    pk = pk.matmul(P);
                }
            }
            Tensor pe = cat(new TensorVector(peList.toArray(new Tensor[0])), 1);
            data.x = cat(new TensorVector(data.x, pe.to(data.x.dtype())), 1);
            return data;
        }
    }

    /** Store largest Laplacian eigenvalue as {@code data['lambda_max']}. */
    public static class LaplacianLambdaMax implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            data = new AdvancedStructuralTransforms.ToDense().apply(data);
            Tensor adj = data.adj;
            Tensor deg = adj.sum(1);
            Tensor L = diag(deg).sub(adj);
            Tensor eigVals = linalg_eigvalsh(L);
            data.put("lambda_max", eigVals.max());
            return data;
        }
    }
}
