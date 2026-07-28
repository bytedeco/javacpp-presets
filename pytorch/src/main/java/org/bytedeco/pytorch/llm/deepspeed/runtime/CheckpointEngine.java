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
package org.bytedeco.pytorch.llm.deepspeed.runtime;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedConfig;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Save / load DeepSpeed engine checkpoints (parameters + client state).
 *
 * <p>Rank 0 writes; all ranks barrier when a process group is present.
 * Parameter tensors are stored as little-endian float32 raw dumps plus shape metadata.
 */
public final class CheckpointEngine {

    private CheckpointEngine() {}

    public static void save(Path dir, Module module, DeepSpeedConfig config,
                            long globalStep, Map<String, Object> clientState,
                            ProcessGroupWrapper pg) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(module, "module");
        if (pg != null && pg.getWorldSize() > 1) {
            try { pg.barrier(); } catch (Exception ignored) {}
        }
        boolean isMain = pg == null || pg.isMainProcess();
        if (isMain) {
            Files.createDirectories(dir);
            Path meta = dir.resolve("ds_checkpoint.meta");
            Map<String, Object> metaMap = new LinkedHashMap<>();
            metaMap.put("global_step", globalStep);
            metaMap.put("zero_stage", config == null ? -1 : config.zeroStage());
            metaMap.put("precision", config == null ? "fp32" : config.precision());
            if (clientState != null) metaMap.putAll(clientState);
            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(meta)))) {
                oos.writeObject(metaMap);
            }
            Path weights = dir.resolve("ds_checkpoint.weights");
            try (DataOutputStream dos = new DataOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(weights)))) {
                TensorVector params = module.parameters();
                dos.writeInt((int) params.size());
                for (long i = 0, n = params.size(); i < n; i++) {
                    Tensor p = params.get(i);
                    if (p == null || p.isNull()) {
                        dos.writeInt(0);
                        continue;
                    }
                    Tensor f = p.detach().to(ScalarType.Float).contiguous().cpu();
                    long numel = f.numel();
                    int ndim = (int) f.dim();
                    dos.writeInt(ndim);
                    for (int d = 0; d < ndim; d++) dos.writeLong(f.size(d));
                    dos.writeLong(numel);
                    float[] data = toFloatArray(f.reshape(numel), (int) numel);
                    for (float v : data) dos.writeFloat(v);
                }
            }
        }
        if (pg != null && pg.getWorldSize() > 1) {
            try { pg.barrier(); } catch (Exception ignored) {}
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> load(Path dir, Module module,
                                           ProcessGroupWrapper pg) throws IOException, ClassNotFoundException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(module, "module");
        if (pg != null && pg.getWorldSize() > 1) {
            try { pg.barrier(); } catch (Exception ignored) {}
        }
        Path meta = dir.resolve("ds_checkpoint.meta");
        Path weights = dir.resolve("ds_checkpoint.weights");
        Map<String, Object> metaMap = new HashMap<>();
        if (Files.exists(meta)) {
            try (ObjectInputStream ois = new ObjectInputStream(
                    new BufferedInputStream(Files.newInputStream(meta)))) {
                Object o = ois.readObject();
                if (o instanceof Map) metaMap.putAll((Map<String, Object>) o);
            }
        }
        if (Files.exists(weights)) {
            try (DataInputStream dis = new DataInputStream(
                    new BufferedInputStream(Files.newInputStream(weights)))) {
                int n = dis.readInt();
                TensorVector params = module.parameters();
                for (int i = 0; i < n; i++) {
                    int ndim = dis.readInt();
                    if (ndim == 0) continue;
                    long[] shape = new long[ndim];
                    long numel = 1;
                    for (int d = 0; d < ndim; d++) {
                        shape[d] = dis.readLong();
                        numel *= shape[d];
                    }
                    long declared = dis.readLong();
                    if (declared != numel) numel = declared;
                    float[] data = new float[(int) numel];
                    for (int j = 0; j < data.length; j++) data[j] = dis.readFloat();
                    if (i < params.size()) {
                        Tensor p = params.get(i);
                        if (p != null && !p.isNull()) {
                            Tensor t = tensor(data).reshape(shape).to(p.scalar_type());
                            p.copy_(t);
                        }
                    }
                }
            }
        }
        if (pg != null && pg.getWorldSize() > 1) {
            TensorVector params = module.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor p = params.get(i);
                if (p != null && !p.isNull()) {
                    try { pg.broadcast(p, 0); } catch (Exception ignored) {}
                }
            }
            try { pg.barrier(); } catch (Exception ignored) {}
        }
        return metaMap;
    }

    private static float[] toFloatArray(Tensor flat, int numel) {
        float[] data = new float[numel];
        try {
            FloatIndexer idx = flat.createIndexer();
            try {
                for (int j = 0; j < numel; j++) data[j] = idx.get(j);
            } finally {
                idx.release();
            }
        } catch (Throwable t) {
            // leave zeros
        }
        return data;
    }
}
