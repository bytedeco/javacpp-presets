/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.vision.ops;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * torchvision.ops subset: NMS, box IoU (pure Java on float arrays / Tensor).
 */
public final class Boxes {
    private Boxes() {}

    /**
     * Greedy NMS.
     *
     * @param boxes  float array of shape [N*4] as x1,y1,x2,y2
     * @param scores length N
     * @param iouThreshold IoU threshold
     * @return kept indices in score-descending order
     */
    public static int[] nms(float[] boxes, float[] scores, float iouThreshold) {
        Objects.requireNonNull(boxes, "boxes");
        Objects.requireNonNull(scores, "scores");
        int n = scores.length;
        if (boxes.length != n * 4) {
            throw new IllegalArgumentException("boxes length must be N*4");
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> scores[i]).reversed());
        boolean[] suppressed = new boolean[n];
        List<Integer> keep = new ArrayList<>();
        for (int _i = 0; _i < n; _i++) {
            int i = order[_i];
            if (suppressed[i]) continue;
            keep.add(i);
            float x1 = boxes[i * 4];
            float y1 = boxes[i * 4 + 1];
            float x2 = boxes[i * 4 + 2];
            float y2 = boxes[i * 4 + 3];
            float area_i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            for (int _j = _i + 1; _j < n; _j++) {
                int j = order[_j];
                if (suppressed[j]) continue;
                float xx1 = Math.max(x1, boxes[j * 4]);
                float yy1 = Math.max(y1, boxes[j * 4 + 1]);
                float xx2 = Math.min(x2, boxes[j * 4 + 2]);
                float yy2 = Math.min(y2, boxes[j * 4 + 3]);
                float w = Math.max(0, xx2 - xx1);
                float h = Math.max(0, yy2 - yy1);
                float inter = w * h;
                float area_j = Math.max(0, boxes[j * 4 + 2] - boxes[j * 4])
                        * Math.max(0, boxes[j * 4 + 3] - boxes[j * 4 + 1]);
                float union = area_i + area_j - inter + 1e-6f;
                if (inter / union > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        int[] out = new int[keep.size()];
        for (int i = 0; i < keep.size(); i++) out[i] = keep.get(i);
        return out;
    }

    /** Tensor NMS: boxes [N,4], scores [N] → Long tensor of indices. */
    public static Tensor nms(Tensor boxes, Tensor scores, float iouThreshold) {
        float[] b = ImageTensors.toFloatArray(boxes.reshape(-1));
        float[] s = ImageTensors.toFloatArray(scores.reshape(-1));
        int[] keep = nms(b, s, iouThreshold);
        long[] idx = new long[keep.length];
        for (int i = 0; i < keep.length; i++) idx[i] = keep[i];
        return torch.tensor(idx);
    }

    public static float box_iou(float[] a, float[] b) {
        float xx1 = Math.max(a[0], b[0]);
        float yy1 = Math.max(a[1], b[1]);
        float xx2 = Math.min(a[2], b[2]);
        float yy2 = Math.min(a[3], b[3]);
        float w = Math.max(0, xx2 - xx1);
        float h = Math.max(0, yy2 - yy1);
        float inter = w * h;
        float areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
        float areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
        return inter / (areaA + areaB - inter + 1e-6f);
    }

    public static float boxIou(float[] a, float[] b) {
        return box_iou(a, b);
    }
}
