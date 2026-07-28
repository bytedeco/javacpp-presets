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
import org.bytedeco.javacpp.annotation.Properties;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-dimensional device mesh (Java topology layer).
 *
 * <p>Not a binding of PyTorch 2.x C++ {@code DeviceMesh} — rank coordinates and
 * named dimensions over an existing {@link ProcessGroupWrapper}. Sub-meshes
 * share the parent process group; dimension-local collectives use the full
 * group with rank filtering helpers, or callers may construct dedicated
 * subgroups externally.
 *
 * <pre>{@code
 * // 1D DP mesh over all ranks
 * DeviceMesh mesh = DeviceMesh.init(pg, new int[]{worldSize}, new String[]{"dp"});
 *
 * // 2D: dp * tp == worldSize
 * DeviceMesh mesh2 = DeviceMesh.init(pg, new int[]{dp, tp}, new String[]{"dp", "tp"});
 * DeviceMesh tpMesh = mesh2.get("tp");
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DeviceMesh implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final ProcessGroupWrapper processGroup;
    private final int[] meshShape;
    private final String[] dimNames;
    private final int[] coords;          // this rank's coordinate per dim
    private final int ndim;
    private final Map<String, Integer> nameToDim = new HashMap<>();
    /** Optional parent mesh when this is a sliced view. */
    private final DeviceMesh parent;
    private final int sliceDim;          // -1 if root
    private final int[] globalRanks;     // ranks belonging to this (sub)mesh

    private DeviceMesh(
            ProcessGroupWrapper processGroup,
            int[] meshShape,
            String[] dimNames,
            DeviceMesh parent,
            int sliceDim,
            int[] globalRanks) {
        this.processGroup = Objects.requireNonNull(processGroup, "processGroup");
        this.meshShape = meshShape.clone();
        this.dimNames = dimNames.clone();
        this.ndim = meshShape.length;
        this.parent = parent;
        this.sliceDim = sliceDim;
        this.globalRanks = globalRanks.clone();

        if (dimNames.length != meshShape.length) {
            throw new IllegalArgumentException("meshShape and dimNames length mismatch");
        }
        long product = 1;
        for (int s : meshShape) {
            if (s <= 0) throw new IllegalArgumentException("mesh dim must be > 0");
            product *= s;
        }
        if (parent == null && product != processGroup.getWorldSize()) {
            throw new IllegalArgumentException(
                    "product(meshShape)=" + product + " != worldSize=" + processGroup.getWorldSize());
        }
        for (int i = 0; i < dimNames.length; i++) {
            if (dimNames[i] == null || dimNames[i].isEmpty()) {
                throw new IllegalArgumentException("dim name empty at " + i);
            }
            if (nameToDim.put(dimNames[i], i) != null) {
                throw new IllegalArgumentException("duplicate dim name: " + dimNames[i]);
            }
        }
        int rank = processGroup.getRank();
        this.coords = rankToCoords(indexOfRank(rank), meshShape);
    }

    /**
     * Initialize a root mesh spanning the entire process group.
     *
     * @param pg        process group (world)
     * @param meshShape sizes per dimension; product must equal {@code pg.getWorldSize()}
     * @param dimNames  names aligned with {@code meshShape} (e.g. {@code "dp"}, {@code "tp"})
     */
    public static DeviceMesh init(ProcessGroupWrapper pg, int[] meshShape, String[] dimNames) {
        int[] ranks = new int[pg.getWorldSize()];
        for (int i = 0; i < ranks.length; i++) ranks[i] = i;
        return new DeviceMesh(pg, meshShape, dimNames, null, -1, ranks);
    }

    /** 1D mesh named {@code "dp"} over all ranks. */
    public static DeviceMesh init1d(ProcessGroupWrapper pg) {
        return init(pg, new int[]{pg.getWorldSize()}, new String[]{"dp"});
    }

    /** 1D mesh with custom dim name. */
    public static DeviceMesh init1d(ProcessGroupWrapper pg, String dimName) {
        return init(pg, new int[]{pg.getWorldSize()}, new String[]{dimName});
    }

    /**
     * 2D mesh for DP×TP hybrid: {@code dpSize * tpSize == worldSize}.
     */
    public static DeviceMesh initDpTp(ProcessGroupWrapper pg, int tpSize) {
        int world = pg.getWorldSize();
        if (tpSize <= 0 || world % tpSize != 0) {
            throw new IllegalArgumentException(
                    "worldSize=" + world + " not divisible by tpSize=" + tpSize);
        }
        int dpSize = world / tpSize;
        return init(pg, new int[]{dpSize, tpSize}, new String[]{"dp", "tp"});
    }

    /** Sub-mesh along a named dimension (logical view; shares parent PG). */
    public DeviceMesh get(String dimName) {
        Integer d = nameToDim.get(dimName);
        if (d == null) {
            throw new IllegalArgumentException("unknown dim: " + dimName + " in " + Arrays.toString(dimNames));
        }
        return get(d);
    }

    /**
     * Sub-mesh for dimension {@code dim}: all ranks that share this rank's
     * coordinates on every other dimension (the fiber along {@code dim}).
     */
    public DeviceMesh get(int dim) {
        if (dim < 0 || dim >= ndim) {
            throw new IllegalArgumentException("dim out of range: " + dim);
        }
        // Collect global ranks in the same fiber
        int[] my = coords;
        java.util.ArrayList<Integer> fiber = new java.util.ArrayList<>();
        for (int gr : globalRanks) {
            int localIdx = indexOfRank(gr);
            if (localIdx < 0) continue;
            int[] c = rankToCoords(localIdx, meshShape);
            boolean same = true;
            for (int d = 0; d < ndim; d++) {
                if (d == dim) continue;
                if (c[d] != my[d]) {
                    same = false;
                    break;
                }
            }
            if (same) fiber.add(gr);
        }
        int[] subRanks = fiber.stream().mapToInt(Integer::intValue).toArray();
        int[] subShape = new int[]{meshShape[dim]};
        String[] subNames = new String[]{dimNames[dim]};
        return new DeviceMesh(processGroup, subShape, subNames, this, dim, subRanks);
    }

    public ProcessGroupWrapper processGroup() { return processGroup; }
    public int[] meshShape() { return meshShape.clone(); }
    public String[] dimNames() { return dimNames.clone(); }
    public int ndim() { return ndim; }
    public int size() { return globalRanks.length; }
    public int size(int dim) { return meshShape[dim]; }
    public int size(String dimName) { return meshShape[nameToDim.get(dimName)]; }
    public int[] getCoordinate() { return coords.clone(); }
    public int getCoordinate(int dim) { return coords[dim]; }
    public int getCoordinate(String dimName) { return coords[nameToDim.get(dimName)]; }
    public int[] globalRanks() { return globalRanks.clone(); }
    public boolean containsRank(int rank) {
        for (int r : globalRanks) if (r == rank) return true;
        return false;
    }
    public int getRank() { return processGroup.getRank(); }
    public DeviceMesh parent() { return parent; }
    public int sliceDim() { return sliceDim; }

    /** Local index of this rank inside {@link #globalRanks()}, or -1. */
    public int localRank() {
        return indexOfRank(processGroup.getRank());
    }

    private int indexOfRank(int rank) {
        for (int i = 0; i < globalRanks.length; i++) {
            if (globalRanks[i] == rank) return i;
        }
        // Root mesh: global rank == index when ranks are 0..N-1 identity
        if (parent == null && rank >= 0 && rank < globalRanks.length && globalRanks[rank] == rank) {
            return rank;
        }
        return -1;
    }

    /** Row-major coords from flat local index. */
    static int[] rankToCoords(int flat, int[] shape) {
        if (flat < 0) flat = 0;
        int[] c = new int[shape.length];
        int x = flat;
        for (int d = shape.length - 1; d >= 0; d--) {
            c[d] = x % shape[d];
            x /= shape[d];
        }
        return c;
    }

    static int coordsToRank(int[] coords, int[] shape) {
        int flat = 0;
        for (int d = 0; d < shape.length; d++) {
            flat = flat * shape[d] + coords[d];
        }
        return flat;
    }

    @Override
    public String toString() {
        return "DeviceMesh{shape=" + Arrays.toString(meshShape)
                + ", names=" + Arrays.toString(dimNames)
                + ", coords=" + Arrays.toString(coords)
                + ", ranks=" + Arrays.toString(globalRanks)
                + ", pgRank=" + processGroup.getRank() + '}';
    }

    /**
     * No native resources owned — mesh is a pure topology view over an existing
     * {@link ProcessGroupWrapper}. close() is a no-op so try-with-resources works.
     */
    @Override
    public void close() {
        // topology-only; PG lifetime owned by caller
    }
}
