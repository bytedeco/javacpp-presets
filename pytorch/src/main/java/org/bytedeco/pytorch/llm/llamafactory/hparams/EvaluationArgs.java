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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.LinkedHashMap;
import java.util.Map;

/** Evaluation harness args. */
public final class EvaluationArgs {
    private final String task;
    private final String taskDir;
    private final int batchSize;
    private final long seed;
    private final String lang;
    private final int nShot;
    private final String saveDir;
    private final String downloadMode;

    private EvaluationArgs(Builder b) {
        this.task = b.task == null ? "mmlu_test" : b.task;
        this.taskDir = b.taskDir == null ? "evaluation" : b.taskDir;
        this.batchSize = b.batchSize;
        this.seed = b.seed;
        this.lang = b.lang == null ? "en" : b.lang;
        this.nShot = b.nShot;
        this.saveDir = b.saveDir;
        this.downloadMode = b.downloadMode == null ? "reuse_dataset_if_exists" : b.downloadMode;
    }

    public String task() { return task; }
    public String taskDir() { return taskDir; }
    public int batchSize() { return batchSize; }
    public long seed() { return seed; }
    public String lang() { return lang; }
    public int nShot() { return nShot; }
    public String saveDir() { return saveDir; }
    public String downloadMode() { return downloadMode; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "task", task);
        HparamsMaps.put(m, "task_dir", taskDir);
        HparamsMaps.put(m, "batch_size", batchSize);
        HparamsMaps.put(m, "seed", seed);
        HparamsMaps.put(m, "lang", lang);
        HparamsMaps.put(m, "n_shot", nShot);
        HparamsMaps.put(m, "save_dir", saveDir);
        HparamsMaps.put(m, "download_mode", downloadMode);
        return m;
    }

    public static EvaluationArgs defaults() { return builder().build(); }

    public static EvaluationArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.task(HparamsMaps.str(m, b.task, "task", "eval_task"));
        b.taskDir(HparamsMaps.str(m, b.taskDir, "task_dir"));
        b.batchSize(HparamsMaps.integer(m, b.batchSize, "batch_size", "eval_batch_size"));
        b.seed(HparamsMaps.lng(m, b.seed, "seed"));
        b.lang(HparamsMaps.str(m, b.lang, "lang", "language"));
        b.nShot(HparamsMaps.integer(m, b.nShot, "n_shot", "nshot"));
        b.saveDir(HparamsMaps.strOrNull(m, "save_dir"));
        b.downloadMode(HparamsMaps.str(m, b.downloadMode, "download_mode"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String task = "mmlu_test";
        private String taskDir = "evaluation";
        private int batchSize = 4;
        private long seed = 42L;
        private String lang = "en";
        private int nShot = 5;
        private String saveDir;
        private String downloadMode = "reuse_dataset_if_exists";

        public Builder task(String v) { this.task = v; return this; }
        public Builder taskDir(String v) { this.taskDir = v; return this; }
        public Builder batchSize(int v) { this.batchSize = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder lang(String v) { this.lang = v; return this; }
        public Builder nShot(int v) { this.nShot = v; return this; }
        public Builder saveDir(String v) { this.saveDir = v; return this; }
        public Builder downloadMode(String v) { this.downloadMode = v; return this; }
        public EvaluationArgs build() { return new EvaluationArgs(this); }
    }
}
