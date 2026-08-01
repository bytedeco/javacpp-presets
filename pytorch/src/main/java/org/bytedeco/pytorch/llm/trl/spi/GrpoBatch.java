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

package org.bytedeco.pytorch.llm.trl.spi;

import org.bytedeco.pytorch.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.tensor;

/** Helper to build GRPO / precomputed-RL batches. */
public final class GrpoBatch {

    private GrpoBatch() {}

    public static Map<String, Tensor> of(int[] promptIds, int[] completionIds, float[] rewards) {
        Objects.requireNonNull(promptIds);
        Objects.requireNonNull(completionIds);
        Objects.requireNonNull(rewards);
        Map<String, Tensor> m = new LinkedHashMap<>();
        m.put("prompt_ids", tensor(promptIds).reshape(1, promptIds.length));
        m.put("completion_ids", tensor(completionIds).reshape(1, completionIds.length));
        m.put("input_ids", tensor(completionIds).reshape(1, completionIds.length));
        m.put("rewards", tensor(rewards));
        return m;
    }

    public static Map<String, Tensor> precomputed(float[] rewards, float[] logprobs, float[] oldLogprobs) {
        Map<String, Tensor> m = new LinkedHashMap<>();
        if (rewards != null) m.put("rewards", tensor(rewards));
        if (logprobs != null) m.put("logprobs", tensor(logprobs));
        if (oldLogprobs != null) m.put("old_logprobs", tensor(oldLogprobs));
        return m;
    }

    public static Map<String, Tensor> toMap(Map<String, Tensor> batch) {
        return batch;
    }
}
