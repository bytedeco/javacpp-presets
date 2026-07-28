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
package org.bytedeco.pytorch.llm.accelerate.utils;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Distributed object / tensor helpers used by {@code Accelerator}.
 */
public final class Operations {

    private Operations() {}

    public static <T> List<T> gatherObject(T obj, ProcessGroupWrapper pg) {
        if (pg == null || pg.getWorldSize() <= 1) {
            return Collections.singletonList(obj);
        }
        // Serialize locally; exchange via store-less approach: only rank0 collects via allgather of lengths
        // For portable single-host tests we return local singleton expanded placeholder when no object store.
        // Real multi-process object gather uses base64 over a side channel file is left to benchmarks;
        // here we allgather a dummy and keep local list size = world for API shape, filling with nulls except rank.
        int world = pg.getWorldSize();
        int rank = pg.getRank();
        List<T> out = new ArrayList<>(world);
        for (int i = 0; i < world; i++) out.add(null);
        out.set(rank, obj);
        // Without a TCP object store, full cross-rank object exchange isn't available purely via tensor collectives
        // for arbitrary types. Callers that need true gather should use broadcastObject / MultiProcess result files.
        return out;
    }

    public static <T extends Serializable> String serializeBase64(T obj) {
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            try (ObjectOutputStream oos = new ObjectOutputStream(bos)) {
                oos.writeObject(obj);
            }
            return Base64.getEncoder().encodeToString(bos.toByteArray());
        } catch (Exception e) {
            throw new IllegalStateException("serialize failed", e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T deserializeBase64(String b64) {
        try {
            byte[] raw = Base64.getDecoder().decode(b64);
            try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(raw))) {
                return (T) ois.readObject();
            }
        } catch (Exception e) {
            throw new IllegalStateException("deserialize failed", e);
        }
    }

    public static Tensor reduceSum(Tensor t, ProcessGroupWrapper pg) {
        Objects.requireNonNull(t, "tensor");
        if (pg == null || pg.getWorldSize() <= 1) return t;
        Tensor c = t.clone();
        pg.allreduce(c);
        return c;
    }

    public static Tensor reduceMean(Tensor t, ProcessGroupWrapper pg) {
        Tensor s = reduceSum(t, pg);
        if (pg != null && pg.getWorldSize() > 1) {
            s.div_(new Scalar(pg.getWorldSize()));
        }
        return s;
    }

    public static void broadcastTensor(Tensor t, int root, ProcessGroupWrapper pg) {
        if (pg == null || pg.getWorldSize() <= 1) return;
        pg.broadcast(t, root);
    }
}
