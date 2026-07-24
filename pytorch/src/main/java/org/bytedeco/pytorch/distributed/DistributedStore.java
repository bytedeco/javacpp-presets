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
package org.bytedeco.pytorch.distributed;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.chrono.Milliseconds;
import org.bytedeco.pytorch.ByteVector;
import org.bytedeco.pytorch.StringVector;

import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Rank-prefixed key/value store used for process-group rendezvous.
 *
 * <p>Supports {@link FileStore} (single-machine) and {@link TCPStore}
 * (multi-machine). Keys are automatically prefixed with {@code _rank_N_}
 * so concurrent ranks do not collide on shared stores.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DistributedStore implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Set<DistributedStore> INSTANCES =
            ConcurrentHashMap.newKeySet();

    private final Options options;
    private final int rank;
    private final int worldSize;
    private final String prefix;
    private final StoreType selectedType;
    private final Store store;

    public DistributedStore(Options options, int rank, int worldSize) {
        this.options = Objects.requireNonNull(options, "options");
        this.rank = rank;
        this.worldSize = worldSize;
        this.prefix = "_rank_" + rank + "_";
        this.selectedType = resolveType(options);
        if (selectedType == StoreType.TCP) {
            this.store = createTcpStore(options);
        } else if (selectedType == StoreType.HASH) {
            this.store = new HashStore();
        } else {
            this.store = createFileStore(worldSize);
        }
        INSTANCES.add(this);
    }

    public static DistributedStore create(int rank, int worldSize) {
        return create(new Options(), rank, worldSize);
    }

    public static DistributedStore create(Options options, int rank, int worldSize) {
        return new DistributedStore(options, rank, worldSize);
    }

    public void set(String key, String value) {
        store.set(prefix + key, value);
    }

    public void set(String key, byte[] value) {
        store.set(prefix + key, new ByteVector(value));
    }

    public String getString(String key) {
        try {
            ByteVector bv = store.get(prefix + key);
            if (bv == null || bv.empty()) {
                return null;
            }
            return new String(bv.get(), StandardCharsets.UTF_8);
        } catch (RuntimeException e) {
            return null;
        }
    }

    public byte[] getBytes(String key) {
        try {
            ByteVector bv = store.get(prefix + key);
            if (bv == null || bv.empty()) {
                return null;
            }
            return bv.get();
        } catch (RuntimeException e) {
            return null;
        }
    }

    public long add(String key, long value) {
        return store.add(prefix + key, value);
    }

    public int getInteger(String key) {
        return (int) store.add(prefix + key, 0L);
    }

    public void delete(String key) {
        store.deleteKey(prefix + key);
    }

    public boolean exists(String key) {
        return store.check(new StringVector(prefix + key));
    }

    public void waitFor(String key) {
        store._wait(new StringVector(prefix + key));
    }

    public void waitFor(String key, int timeoutMs) {
        store._wait(new StringVector(prefix + key), new Milliseconds(timeoutMs));
    }

    public Store getNativeStore() { return store; }
    public int getRank() { return rank; }
    public int getWorldSize() { return worldSize; }
    public StoreType getType() { return selectedType; }

    @Override
    public void close() {
        INSTANCES.remove(this);
    }

    @Override
    public String toString() {
        return "DistributedStore{type=" + selectedType
                + ", rank=" + rank + ", worldSize=" + worldSize + '}';
    }

    private static StoreType resolveType(Options opts) {
        if (opts.getType() != StoreType.AUTO) {
            return opts.getType();
        }
        String addr = opts.getMasterAddr();
        boolean multi = addr != null
                && !"127.0.0.1".equals(addr)
                && !"localhost".equalsIgnoreCase(addr);
        return multi ? StoreType.TCP : StoreType.FILE;
    }

    private static Store createFileStore(int worldSize) {
        String path = "/tmp/pytorch_ddp_store_" + System.getProperty("user.name", "user");
        return new FileStore(path, worldSize);
    }

    private static Store createTcpStore(Options opts) {
        TCPStoreOptions tcpOpts = new TCPStoreOptions();
        tcpOpts.port((short) opts.getMasterPort());
        tcpOpts.isServer(false);
        SizeTPointer workers = tcpOpts.numWorkers();
        if (workers != null) {
            workers.put(opts.getNumWorkers());
        }
        tcpOpts.timeout(new Milliseconds(opts.getTimeoutMs()));
        return new TCPStore(opts.getMasterAddr(), tcpOpts);
    }

    /** Builder-style options for {@link DistributedStore}. */
    public static final class Options {
        private StoreType storeType = StoreType.AUTO;
        private int timeoutMs = 300_000;
        private String masterAddr = "127.0.0.1";
        private int masterPort = 29_500;
        private int numWorkers = 1;

        public Options type(StoreType t) { this.storeType = t; return this; }
        public Options timeout(int ms) { this.timeoutMs = ms; return this; }
        public Options masterAddr(String addr) { this.masterAddr = addr; return this; }
        public Options masterPort(int port) { this.masterPort = port; return this; }
        public Options numWorkers(int n) { this.numWorkers = n; return this; }

        public StoreType getType() { return storeType; }
        public int getTimeoutMs() { return timeoutMs; }
        public String getMasterAddr() { return masterAddr; }
        public int getMasterPort() { return masterPort; }
        public int getNumWorkers() { return numWorkers; }
    }
}
