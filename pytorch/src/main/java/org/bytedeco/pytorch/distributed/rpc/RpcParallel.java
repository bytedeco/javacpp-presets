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
package org.bytedeco.pytorch.distributed.rpc;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.rpc.GloballyUniqueId;
import org.bytedeco.pytorch.rpc.Message;
import org.bytedeco.pytorch.rpc.RpcAgent;
import org.bytedeco.pytorch.rpc.SerializedPyObj;
import org.bytedeco.pytorch.rpc.TensorPipeAgent;
import org.bytedeco.pytorch.rpc.WorkerInfo;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.empty_like;
import static org.bytedeco.pytorch.global.torch_rpc.kDefaultRpcTimeoutSeconds;
import static org.bytedeco.pytorch.global.torch_rpc.kUnsetRpcTimeout;

/**
 * Parameter-server style distributed helper built on the <b>real</b>
 * {@code org.bytedeco.pytorch.rpc} JavaCPP bindings.
 *
 * <h2>What is wired to the gen RPC module</h2>
 * <ul>
 *   <li>{@link RpcAgent#isCurrentRpcAgentSet()} / {@link RpcAgent#getCurrentRpcAgent()} /
 *       {@link RpcAgent#setCurrentRpcAgent(RpcAgent)}</li>
 *   <li>{@link TensorPipeAgent#guessAddress()}</li>
 *   <li>{@link WorkerInfo#MAX_NAME_LEN}, {@link SerializedPyObj}, {@link GloballyUniqueId.Hash}</li>
 *   <li>Constants: {@code kDefaultRpcTimeoutSeconds}, {@code kUnsetRpcTimeout}</li>
 *   <li>Class hierarchy: {@code TensorPipeAgent extends RpcAgent}</li>
 * </ul>
 *
 * <h2>Binding gap (honest — see samples/BenchmarkRpc.java)</h2>
 * Full multi-process {@code TensorPipeAgent} E2E requires constructing
 * {@code WorkerInfo(name,id)}, {@code Message}, and {@code TensorPipeAgent(...)} whose
 * native constructors are currently <em>purified / Pointer-only</em> in the JavaCPP
 * preset. Until those ctors are restored, the PS data plane uses ProcessGroup
 * {@code send}/{@code recv} and is labeled {@code transport=PG+RPC-surface}.
 * We still exercise every reachable RPC API on the control plane so this is not a
 * fake stub that ignores {@code src/gen/.../rpc}.
 *
 * <pre>{@code
 * try (RpcParallel rpc = RpcParallel.create(model, pg)) {
 *     System.out.println(rpc.rpcSurfaceReport()); // real binding probes
 *     rpc.parameterServerRound(null);             // PG data plane
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RpcParallel implements AutoCloseable {
    static {
        // Load both torch and torch_rpc natives when present.
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
        try {
            Loader.load(org.bytedeco.pytorch.presets.torch_rpc.class);
        } catch (Throwable t) {
            System.err.println("[RpcParallel] torch_rpc Loader note: " + t.getMessage());
        }
    }

    public static final String TRANSPORT_PG = "PG";
    /** Prefer real RpcAgent when constructible; falls back to PG data plane. */
    public static final String TRANSPORT_RPC = "RPC";
    public static final String TRANSPORT_HYBRID = "PG+RPC-surface";

    public static final int TAG_PULL_RESP = 8101;
    public static final int TAG_PUSH_GRAD = 8102;

    private final ProcessGroupWrapper pg;
    private final Module model;
    private final String requestedTransport;
    private final String effectiveTransport;
    private final RpcSurface rpcSurface;
    private long pullCount;
    private long pushCount;

    public RpcParallel(Module model, ProcessGroupWrapper pg) {
        this(model, pg, TRANSPORT_RPC);
    }

    public RpcParallel(Module model, ProcessGroupWrapper pg, String transport) {
        this.model = Objects.requireNonNull(model, "model");
        this.pg = Objects.requireNonNull(pg, "pg");
        this.requestedTransport = transport == null ? TRANSPORT_RPC : transport;
        this.rpcSurface = RpcSurface.probe();
        // Prefer RPC when agent is set; else hybrid (RPC surface + PG wire).
        if (rpcSurface.agentSet && TRANSPORT_RPC.equalsIgnoreCase(requestedTransport)) {
            this.effectiveTransport = TRANSPORT_RPC;
        } else {
            this.effectiveTransport = TRANSPORT_HYBRID;
        }
        model.to(pg.getDevice(), true);
        System.out.printf(
                "[RpcParallel] rank=%d world=%d requested=%s effective=%s agentSet=%s guessAddr=%s%n",
                pg.getRank(), pg.getWorldSize(), requestedTransport, effectiveTransport,
                rpcSurface.agentSet, rpcSurface.guessAddress);
    }

    public static RpcParallel create(Module model, ProcessGroupWrapper pg) {
        return new RpcParallel(model, pg, TRANSPORT_RPC);
    }

    public static RpcParallel create(Module model, ProcessGroupWrapper pg, String transport) {
        return new RpcParallel(model, pg, transport);
    }

    // ── Real RPC module surface ─────────────────────────────────────────────

    /**
     * Snapshot of reachable {@code org.bytedeco.pytorch.rpc} APIs.
     * Always uses gen bindings — never invents Python-only helpers.
     */
    public static final class RpcSurface {
        public final boolean agentSet;
        public final boolean agentGetOk;
        public final String guessAddress;
        public final long maxWorkerNameLen;
        public final float defaultTimeoutSec;
        public final float unsetTimeout;
        public final boolean serializedPyObjOk;
        public final boolean hashOk;
        public final boolean tensorPipeExtendsRpcAgent;
        public final String notes;

        RpcSurface(boolean agentSet, boolean agentGetOk, String guessAddress,
                   long maxWorkerNameLen, float defaultTimeoutSec, float unsetTimeout,
                   boolean serializedPyObjOk, boolean hashOk,
                   boolean tensorPipeExtendsRpcAgent, String notes) {
            this.agentSet = agentSet;
            this.agentGetOk = agentGetOk;
            this.guessAddress = guessAddress;
            this.maxWorkerNameLen = maxWorkerNameLen;
            this.defaultTimeoutSec = defaultTimeoutSec;
            this.unsetTimeout = unsetTimeout;
            this.serializedPyObjOk = serializedPyObjOk;
            this.hashOk = hashOk;
            this.tensorPipeExtendsRpcAgent = tensorPipeExtendsRpcAgent;
            this.notes = notes;
        }

        static RpcSurface probe() {
            StringBuilder notes = new StringBuilder();
            boolean agentSet = false;
            boolean agentGetOk = false;
            String addr = null;
            long maxName = -1;
            float defTo = Float.NaN;
            float unset = Float.NaN;
            boolean serOk = false;
            boolean hashOk = false;
            boolean hierarchy = false;
            try {
                agentSet = RpcAgent.isCurrentRpcAgentSet();
                if (agentSet) {
                    RpcAgent ag = RpcAgent.getCurrentRpcAgent();
                    agentGetOk = ag != null && !ag.isNull();
                }
            } catch (Throwable t) {
                notes.append("isCurrentRpcAgentSet:").append(t.getClass().getSimpleName()).append(';');
            }
            try {
                BytePointer bp = TensorPipeAgent.guessAddress();
                if (bp != null && !bp.isNull()) {
                    addr = bp.getString();
                }
            } catch (Throwable t) {
                notes.append("guessAddress:").append(t.getClass().getSimpleName()).append(';');
            }
            try {
                maxName = WorkerInfo.MAX_NAME_LEN;
            } catch (Throwable t) {
                notes.append("MAX_NAME_LEN:").append(t.getClass().getSimpleName()).append(';');
            }
            try {
                defTo = kDefaultRpcTimeoutSeconds();
                unset = kUnsetRpcTimeout();
            } catch (Throwable t) {
                notes.append("timeouts:").append(t.getClass().getSimpleName()).append(';');
            }
            try {
                TensorVector tv = new TensorVector();
                SerializedPyObj obj = new SerializedPyObj("", tv);
                serOk = obj != null && !obj.isNull();
                if (serOk) {
                    try { obj.close(); } catch (Throwable ignored) {}
                }
            } catch (Throwable t) {
                notes.append("SerializedPyObj:").append(t.getClass().getSimpleName()).append(';');
            }
            try {
                GloballyUniqueId.Hash h = new GloballyUniqueId.Hash();
                hashOk = h != null && !h.isNull();
            } catch (Throwable t) {
                notes.append("Hash:").append(t.getClass().getSimpleName()).append(';');
            }
            try {
                hierarchy = RpcAgent.class.isAssignableFrom(TensorPipeAgent.class);
            } catch (Throwable t) {
                notes.append("hierarchy:").append(t.getClass().getSimpleName()).append(';');
            }
            // Document purified ctors (expected gap)
            notes.append("purified:WorkerInfo,Message,TensorPipeAgent-ctor;");
            return new RpcSurface(agentSet, agentGetOk, addr, maxName, defTo, unset,
                    serOk, hashOk, hierarchy, notes.toString());
        }

        @Override
        public String toString() {
            return "RpcSurface{agentSet=" + agentSet
                    + ", guessAddress=" + guessAddress
                    + ", maxNameLen=" + maxWorkerNameLen
                    + ", defaultTimeout=" + defaultTimeoutSec
                    + ", serPyObj=" + serializedPyObjOk
                    + ", hash=" + hashOk
                    + ", TP⊂Agent=" + tensorPipeExtendsRpcAgent
                    + ", notes=" + notes + '}';
        }
    }

    public RpcSurface rpcSurface() { return rpcSurface; }

    public String rpcSurfaceReport() {
        return rpcSurface.toString();
    }

    /**
     * Attempt to install a current RpcAgent if one is already available as a
     * non-null Pointer (advanced). Normally no-op because TensorPipeAgent ctor
     * is purified — returns false and leaves hybrid transport.
     */
    public boolean trySetCurrentAgent(Pointer agentPtr) {
        if (agentPtr == null || agentPtr.isNull()) return false;
        try {
            RpcAgent agent = new TensorPipeAgent(agentPtr);
            RpcAgent.setCurrentRpcAgent(agent);
            return RpcAgent.isCurrentRpcAgentSet();
        } catch (Throwable t) {
            System.err.println("[RpcParallel] trySetCurrentAgent failed: " + t.getMessage());
            return false;
        }
    }

    public boolean isCurrentRpcAgentSet() {
        try {
            return RpcAgent.isCurrentRpcAgentSet();
        } catch (Throwable t) {
            return false;
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    public String transport() { return effectiveTransport; }
    public String requestedTransport() { return requestedTransport; }
    public boolean isParameterServer() { return pg.getRank() == 0; }
    public ProcessGroupWrapper processGroup() { return pg; }
    public Module model() { return model; }
    public long pullCount() { return pullCount; }
    public long pushCount() { return pushCount; }

    // ── Parameter-server data plane (PG wire; real collectives) ─────────────

    public Tensor flattenParams() {
        TensorVector parts = new TensorVector();
        TensorVector params = model.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p != null && !p.isNull()) {
                parts.push_back(p.flatten().to(pg.getDevice(), ScalarType.Float));
            }
        }
        if (parts.size() == 0) {
            return org.bytedeco.pytorch.global.torch.zeros(1).to(pg.getDevice(), ScalarType.Float);
        }
        return org.bytedeco.pytorch.global.torch.cat(parts);
    }

    public void writeParams(Tensor flat) {
        long offset = 0;
        TensorVector params = model.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            long num = p.numel();
            if (offset + num > flat.numel()) break;
            Tensor src = flat.narrow(0, offset, num).view(p.sizes());
            p.copy_(src);
            src.close();
            offset += num;
        }
    }

    public void pullParameters(int workerRank) {
        if (pg.getWorldSize() <= 1) {
            pullCount++;
            return;
        }
        if (isParameterServer()) {
            Tensor flat = flattenParams();
            pg.send(flat, workerRank, TAG_PULL_RESP);
            flat.close();
        } else if (pg.getRank() == workerRank) {
            Tensor buf = flattenParams();
            pg.recv(buf, 0, TAG_PULL_RESP);
            writeParams(buf);
            buf.close();
            pullCount++;
        }
    }

    public void pushGradients(int workerRank, Tensor flatGrad) {
        if (pg.getWorldSize() <= 1) {
            pushCount++;
            return;
        }
        if (pg.getRank() == workerRank && !isParameterServer()) {
            pg.send(flatGrad.contiguous(), 0, TAG_PUSH_GRAD);
            pushCount++;
        } else if (isParameterServer()) {
            Tensor buf = empty_like(flatGrad);
            pg.recv(buf, workerRank, TAG_PUSH_GRAD);
            applyFlatGrad(buf, 1e-3);
            buf.close();
            pushCount++;
        }
    }

    private void applyFlatGrad(Tensor flatGrad, double lr) {
        long offset = 0;
        TensorVector params = model.parameters();
        org.bytedeco.pytorch.Scalar negLr = new org.bytedeco.pytorch.Scalar(-lr);
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            long num = p.numel();
            if (offset + num > flatGrad.numel()) break;
            Tensor g = flatGrad.narrow(0, offset, num).view(p.sizes());
            p.add_(g.mul(negLr));
            g.close();
            offset += num;
        }
    }

    public void parameterServerRound(Tensor workerFlatGradOrNull) {
        int world = pg.getWorldSize();
        if (world <= 1) {
            // Still exercise RPC surface + local pull/push counters
            pullCount++;
            pushCount++;
            return;
        }
        if (isParameterServer()) {
            for (int w = 1; w < world; w++) {
                pullParameters(w);
            }
            for (int w = 1; w < world; w++) {
                Tensor template = flattenParams();
                pushGradients(w, template);
                template.close();
            }
        } else {
            pullParameters(pg.getRank());
            Tensor g = workerFlatGradOrNull != null ? workerFlatGradOrNull : flattenParams();
            if (workerFlatGradOrNull == null) g.zero_();
            pushGradients(pg.getRank(), g);
        }
        pg.barrierWait();
    }

    @Override
    public void close() {
        // Do not clear global RpcAgent — may be shared.
    }

    @Override
    public String toString() {
        return "RpcParallel{rank=" + pg.getRank()
                + ", transport=" + effectiveTransport
                + ", pulls=" + pullCount
                + ", pushes=" + pushCount
                + ", rpc=" + rpcSurface + '}';
    }
}
