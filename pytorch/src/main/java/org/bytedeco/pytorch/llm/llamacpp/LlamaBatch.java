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

package org.bytedeco.pytorch.llm.llamacpp;

import java.util.Arrays;
import java.util.Objects;

/**
 * Decode batch (llama_batch subset): token ids + positions + optional logits flags.
 */
public final class LlamaBatch {
    private final int[] token;
    private final int[] pos;
    private final int[] nSeqId;   // always 1 for simple path
    private final boolean[] logits;
    private int nTokens;

    public LlamaBatch(int capacity) {
        int cap = Math.max(1, capacity);
        this.token = new int[cap];
        this.pos = new int[cap];
        this.nSeqId = new int[cap];
        this.logits = new boolean[cap];
        this.nTokens = 0;
    }

    public static LlamaBatch ofTokens(int[] tokens, int startPos, boolean logitsLastOnly) {
        Objects.requireNonNull(tokens);
        LlamaBatch b = new LlamaBatch(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            b.add(tokens[i], startPos + i, !logitsLastOnly || i == tokens.length - 1);
        }
        return b;
    }

    public void clear() { nTokens = 0; }

    public void add(int tok, int position, boolean needLogits) {
        if (nTokens >= token.length) {
            throw new IllegalStateException("batch full capacity=" + token.length);
        }
        token[nTokens] = tok;
        pos[nTokens] = position;
        nSeqId[nTokens] = 0;
        logits[nTokens] = needLogits;
        nTokens++;
    }

    public int nTokens() { return nTokens; }
    public int capacity() { return token.length; }
    public int token(int i) { return token[i]; }
    public int pos(int i) { return pos[i]; }
    public boolean logits(int i) { return logits[i]; }

    public int[] tokensCopy() { return Arrays.copyOf(token, nTokens); }
    public int[] positionsCopy() { return Arrays.copyOf(pos, nTokens); }
}
