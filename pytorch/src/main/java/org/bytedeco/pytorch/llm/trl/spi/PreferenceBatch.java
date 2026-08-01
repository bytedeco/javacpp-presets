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

/** Helper to build DPO/ORPO batches expected by TRL trainers. */
public final class PreferenceBatch {

    private PreferenceBatch() {}

    public static Map<String, Tensor> ofIds(int[] chosen, int[] rejected) {
        Objects.requireNonNull(chosen);
        Objects.requireNonNull(rejected);
        Tensor c = tensor(chosen).reshape(1, chosen.length);
        Tensor r = tensor(rejected).reshape(1, rejected.length);
        Map<String, Tensor> m = new LinkedHashMap<>();
        m.put("chosen_input_ids", c);
        m.put("rejected_input_ids", r);
        m.put("chosen_labels", c.clone());
        m.put("rejected_labels", r.clone());
        return m;
    }

    public static Map<String, Tensor> ofLogps(float[] policyChosen, float[] policyRejected,
                                              float[] refChosen, float[] refRejected) {
        Map<String, Tensor> m = new LinkedHashMap<>();
        m.put("policy_chosen_logps", tensor(policyChosen));
        m.put("policy_rejected_logps", tensor(policyRejected));
        if (refChosen != null) m.put("ref_chosen_logps", tensor(refChosen));
        if (refRejected != null) m.put("ref_rejected_logps", tensor(refRejected));
        return m;
    }

    public static Map<String, Tensor> toMap(Map<String, Tensor> batch) {
        return batch;
    }
}
