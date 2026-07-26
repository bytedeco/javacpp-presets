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
package org.bytedeco.pytorch.utils.vllm;

import org.bytedeco.pytorch.utils.vllm.multimodal.MultimodalPrompt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One generation request / sequence in the continuous-batching engine.
 *
 * <p>Tracks prompt tokens, generated tokens, sampling params, paged-cache id,
 * and scheduler status.
 */
public final class Sequence {

    private static final AtomicLong NEXT_ID = new AtomicLong(1);

    private final long requestId;
    private final int[] promptTokenIds;
    private final List<Integer> outputTokenIds = new ArrayList<>();
    private final SamplingParams samplingParams;
    private final long arrivalNs;
    private final MultimodalPrompt multimodalPrompt; // optional

    private SequenceStatus status = SequenceStatus.WAITING;
    private long cacheSeqId = -1L;
    private int numComputedTokens = 0;
    private long firstTokenNs = -1L;
    private long finishedNs = -1L;
    private String finishReason = null;
    private String promptText = null;

    public Sequence(int[] promptTokenIds, SamplingParams samplingParams) {
        this(promptTokenIds, samplingParams, null, null);
    }

    public Sequence(int[] promptTokenIds, SamplingParams samplingParams,
                    MultimodalPrompt multimodalPrompt, String promptText) {
        this.requestId = NEXT_ID.getAndIncrement();
        this.promptTokenIds = Objects.requireNonNull(promptTokenIds, "promptTokenIds").clone();
        this.samplingParams = Objects.requireNonNull(samplingParams, "samplingParams");
        this.multimodalPrompt = multimodalPrompt;
        this.promptText = promptText;
        this.arrivalNs = System.nanoTime();
    }

    public long requestId() { return requestId; }
    public int[] promptTokenIds() { return promptTokenIds.clone(); }
    public int promptLen() { return promptTokenIds.length; }
    public List<Integer> outputTokenIds() { return Collections.unmodifiableList(outputTokenIds); }
    public SamplingParams samplingParams() { return samplingParams; }
    public SequenceStatus status() { return status; }
    public void setStatus(SequenceStatus s) { this.status = Objects.requireNonNull(s); }
    public long cacheSeqId() { return cacheSeqId; }
    public void setCacheSeqId(long id) { this.cacheSeqId = id; }
    public int numComputedTokens() { return numComputedTokens; }
    public void setNumComputedTokens(int n) { this.numComputedTokens = n; }
    public MultimodalPrompt multimodalPrompt() { return multimodalPrompt; }
    public String promptText() { return promptText; }
    public void setPromptText(String t) { this.promptText = t; }
    public long arrivalNs() { return arrivalNs; }
    public long firstTokenNs() { return firstTokenNs; }
    public long finishedNs() { return finishedNs; }
    public String finishReason() { return finishReason; }

    public int numOutputTokens() { return outputTokenIds.size(); }

    public int totalTokens() { return promptTokenIds.length + outputTokenIds.size(); }

    /** Tokens not yet prefilled/decoded (for prefill budget). */
    public int numUncomputedTokens() {
        return Math.max(0, promptTokenIds.length - numComputedTokens);
    }

    public boolean isPrefillDone() {
        return numComputedTokens >= promptTokenIds.length;
    }

    public boolean isFinished() {
        return status == SequenceStatus.FINISHED || status == SequenceStatus.ABORTED;
    }

    /**
     * Append a newly sampled output token.
     * Does <b>not</b> bump {@link #numComputedTokens} — that tracks tokens whose
     * K/V are already in the cache, and is advanced by the model runner after
     * the forward that materializes those K/V entries.
     */
    public void appendToken(int tokenId) {
        outputTokenIds.add(tokenId);
        if (firstTokenNs < 0) firstTokenNs = System.nanoTime();
    }

    public void markFinished(String reason) {
        this.status = SequenceStatus.FINISHED;
        this.finishReason = reason == null ? "stop" : reason;
        this.finishedNs = System.nanoTime();
    }

    public void markAborted() {
        this.status = SequenceStatus.ABORTED;
        this.finishReason = "abort";
        this.finishedNs = System.nanoTime();
    }

    /** Full token sequence = prompt + outputs (for repetition penalty). */
    public List<Integer> allTokenIds() {
        List<Integer> all = new ArrayList<>(promptTokenIds.length + outputTokenIds.size());
        for (int id : promptTokenIds) all.add(id);
        all.addAll(outputTokenIds);
        return all;
    }

    public int[] outputTokenIdsArray() {
        int[] a = new int[outputTokenIds.size()];
        for (int i = 0; i < a.length; i++) a[i] = outputTokenIds.get(i);
        return a;
    }

    public double ttftMs() {
        if (firstTokenNs < 0) return -1;
        return (firstTokenNs - arrivalNs) / 1_000_000.0;
    }

    public double latencyMs() {
        long end = finishedNs > 0 ? finishedNs : System.nanoTime();
        return (end - arrivalNs) / 1_000_000.0;
    }

    @Override
    public String toString() {
        return "Sequence{id=" + requestId + ", status=" + status
                + ", prompt=" + promptTokenIds.length
                + ", out=" + outputTokenIds.size()
                + ", computed=" + numComputedTokens + "}";
    }
}
