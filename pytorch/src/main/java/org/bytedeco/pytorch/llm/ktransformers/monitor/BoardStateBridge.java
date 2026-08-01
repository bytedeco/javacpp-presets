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
package org.bytedeco.pytorch.llm.ktransformers.monitor;

import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;

import java.util.Map;
import java.util.Objects;

/**
 * Pushes {@link KtMetrics} into factory {@link BoardState} for visual training UIs.
 */
public final class BoardStateBridge {

    private final BoardState board;
    private final KtMetrics metrics;

    public BoardStateBridge(BoardState board, KtMetrics metrics) {
        this.board = Objects.requireNonNull(board, "board");
        this.metrics = metrics != null ? metrics : new KtMetrics();
    }

    public BoardState board() { return board; }
    public KtMetrics metrics() { return metrics; }

    public void onTrainStep(int step, double loss, double lr, double gradNorm) {
        metrics.recordTrainStep(step, loss, lr, gradNorm);
        board.setGlobalStep(step);
        board.recordLoss(loss);
        board.putMetric("loss", loss);
        board.putMetric("lr", lr);
        board.putMetric("grad_norm", gradNorm);
        board.putMetrics(metrics.snapshot());
        board.setStatus(BoardState.Status.RUNNING);
        board.setMessage(String.format("step=%d loss=%.6f lr=%.2e", step, loss, lr));
        board.log(board.message());
    }

    public void publishAll(Map<String, Double> extra) {
        if (extra != null) {
            metrics.setAll(extra);
            board.putMetrics(extra);
        }
        board.putMetrics(metrics.snapshot());
    }

    public void markCompleted() {
        board.setStatus(BoardState.Status.COMPLETED);
        board.setMessage("completed");
    }

    public void markFailed(String reason) {
        board.setStatus(BoardState.Status.FAILED);
        board.setMessage(reason != null ? reason : "failed");
    }

    public void markRunning() {
        board.clearStop();
        board.setStatus(BoardState.Status.RUNNING);
        board.setMessage("running");
    }
}
