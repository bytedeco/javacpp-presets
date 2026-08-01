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
package org.bytedeco.pytorch.llm.llamafactory.train;

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.TrainingArgs;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Selects the distributed backend for a factory train run.
 *
 * <p>Resolution order (LLaMA-Factory compatible):
 * <ol>
 *   <li>{@code deepspeed} config path → DeepSpeed</li>
 *   <li>{@code fsdp=true} → FSDP</li>
 *   <li>else single-process / optional DDP via env {@code WORLD_SIZE}</li>
 * </ol>
 *
 * <p>Actual engine objects live in {@code llm.accelerate} / {@code llm.deepspeed} /
 * {@code distributed.*}; this class only decides + records the choice so hosts
 * can launch the right process group without re-parsing hparams.
 */
public final class ParallelLauncher {

    private static final Logger LOG = Logger.getLogger(ParallelLauncher.class.getName());

    public enum Backend {
        SINGLE,
        DDP,
        FSDP,
        DEEPSPEED
    }

    public static final class Plan {
        private final Backend backend;
        private final int worldSize;
        private final int localRank;
        private final String deepspeedConfig;
        private final Map<String, Object> meta;

        public Plan(Backend backend, int worldSize, int localRank, String deepspeedConfig,
                    Map<String, Object> meta) {
            this.backend = Objects.requireNonNull(backend, "backend");
            this.worldSize = Math.max(1, worldSize);
            this.localRank = Math.max(0, localRank);
            this.deepspeedConfig = deepspeedConfig;
            this.meta = meta == null ? Map.of() : Map.copyOf(meta);
        }

        public Backend backend() { return backend; }
        public int worldSize() { return worldSize; }
        public int localRank() { return localRank; }
        public String deepspeedConfig() { return deepspeedConfig; }
        public boolean distributed() { return backend != Backend.SINGLE && worldSize > 1; }
        public Map<String, Object> meta() { return meta; }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("backend", backend.name().toLowerCase(Locale.ROOT));
            m.put("world_size", worldSize);
            m.put("local_rank", localRank);
            m.put("deepspeed_config", deepspeedConfig);
            m.put("distributed", distributed());
            m.putAll(meta);
            return m;
        }
    }

    private ParallelLauncher() {}

    public static Plan resolve(FactoryArgs args) {
        Objects.requireNonNull(args, "args");
        TrainingArgs t = args.training();
        int world = envInt("WORLD_SIZE", 1);
        int rank = envInt("LOCAL_RANK", envInt("RANK", 0));

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("fsdp_flag", t.fsdp());
        meta.put("ddp_timeout", t.ddpTimeout());

        String ds = t.deepspeed();
        if (ds != null && !ds.isBlank()) {
            LOG.info("ParallelLauncher → DeepSpeed config=" + ds + " world=" + world);
            return new Plan(Backend.DEEPSPEED, Math.max(1, world), rank, ds, meta);
        }
        if (t.fsdp()) {
            LOG.info("ParallelLauncher → FSDP world=" + world);
            return new Plan(Backend.FSDP, Math.max(1, world), rank, null, meta);
        }
        if (world > 1) {
            LOG.info("ParallelLauncher → DDP world=" + world + " local_rank=" + rank);
            return new Plan(Backend.DDP, world, rank, null, meta);
        }
        return new Plan(Backend.SINGLE, 1, 0, null, meta);
    }

    /** True when this process should run the main logging / checkpoint rank. */
    public static boolean isMain(Plan plan) {
        return plan == null || plan.localRank() == 0;
    }

    private static int envInt(String key, int def) {
        try {
            String v = System.getenv(key);
            if (v == null || v.isBlank()) return def;
            return Integer.parseInt(v.trim());
        } catch (Exception e) {
            return def;
        }
    }
}
