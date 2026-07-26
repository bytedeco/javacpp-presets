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
package org.bytedeco.pytorch.utils.vllm.runner;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.vllm.Sequence;

import java.util.List;

/** Interface for model execution inside the engine (prefill + decode). */
public interface ModelRunner {

    /**
     * Run one forward pass for a sequence using cache-aware forward.
     * @param seq  the sequence (prompt tokens already set, output tokens growing)
     * @param cacheSeqId  paged cache sequence id for this request
     * @return logits for the last position [V]
     */
    Tensor forwardOne(Sequence seq, long cacheSeqId);

    /** Return [logits V] for the last position of each sequence in the batch. */
    List<Tensor> forwardBatch(List<Sequence> prefillSeqs, List<Sequence> decodeSeqs, long[] cacheSeqIds);

    void close();
}
