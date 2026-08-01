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
package org.bytedeco.pytorch.llm.llamafactory.extras.monitor;

import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;

import java.util.Map;
import java.util.Objects;

/**
 * {@link TrainerCallback} that mirrors step metrics into a {@link BoardState}.
 * Prefer {@link MonitorBundle} when multi-backend logging is needed.
 */
public final class BoardMonitor implements TrainerCallback {

    private final BoardState board;

    public BoardMonitor(BoardState board) {
        this.board = Objects.requireNonNull(board, "board");
    }

    public BoardState board() {
        return board;
    }

    @Override
    public void onTrainBegin(BaseTrainer trainer) {
        board.clearStop();
        board.setStatus(BoardState.Status.RUNNING);
        board.setMessage("running");
        board.log("[board] train begin");
    }

    @Override
    public void onTrainEnd(BaseTrainer trainer) {
        board.setGlobalStep(trainer.globalStep());
        if (board.stopRequested()) {
            board.setStatus(BoardState.Status.STOPPED);
            board.setMessage("stopped");
        } else {
            board.setStatus(BoardState.Status.COMPLETED);
            board.setMessage("completed");
        }
        board.log("[board] train end step=" + trainer.globalStep());
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        board.setGlobalStep(step);
        if (metrics != null) {
            board.putMetrics(metrics);
            Double loss = metrics.get("loss");
            if (loss != null) {
                board.recordLoss(loss);
            }
        }
    }

    @Override
    public void onLog(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        board.putMetrics(metrics);
        if (metrics != null && metrics.get("loss") != null) {
            board.log("step=" + step + " loss=" + metrics.get("loss"));
        }
    }
}
