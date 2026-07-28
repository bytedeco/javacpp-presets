/**
 * Pure-Java FAISS-compatible vector index API under DataFrame.
 *
 * <p>Mirrors the Python {@code faiss} surface used in
 * {@code org/lance/ipc/faiss.md}:
 * {@link org.bytedeco.pytorch.dataframe.faiss.IndexFlatL2},
 * {@link org.bytedeco.pytorch.dataframe.faiss.IndexFlatIP},
 * {@link org.bytedeco.pytorch.dataframe.faiss.IndexHNSWFlat},
 * {@link org.bytedeco.pytorch.dataframe.faiss.IndexIVFPQ},
 * {@link org.bytedeco.pytorch.dataframe.faiss.IndexIDMap},
 * {@link org.bytedeco.pytorch.dataframe.faiss.IndexShards},
 * {@link org.bytedeco.pytorch.dataframe.faiss.Faiss#normalize_L2},
 * {@link org.bytedeco.pytorch.dataframe.faiss.Faiss#write_index} /
 * {@link org.bytedeco.pytorch.dataframe.faiss.Faiss#read_index}.
 *
 * <p><b>Not</b> a JNI binding to libfaiss — pure Java search kernels (optional
 * torch CUDA GEMM for Flat). Persistence uses the <b>native FAISS binary</b>
 * fourcc format ({@code IxF2}/{@code IxFI}/{@code IHNf}/{@code IxMp}/{@code IwPQ})
 * so files round-trip with Python {@code faiss.write_index}/{@code faiss.read_index}.
 * Legacy Java-only JDF1 remains via {@code write_index_jdf1}/{@code read_index_jdf1}.
 *
 * <pre>
 *   IndexHNSWFlat index = new IndexHNSWFlat(d, 32);
 *   index.hnsw.efConstruction = 128;
 *   index.hnsw.efSearch = 64;
 *   IndexIDMap withId = new IndexIDMap(index);
 *   withId.add_with_ids(xb, ids);
 *   SearchResult r = withId.search(xq, k);  // r.D, r.I
 *   Faiss.write_index(withId, "idx.faiss"); // Python can faiss.read_index this
 * </pre>
 *
 * <p>DataFrame hooks: {@code df.buildFaiss("emb").hnsw(32).build()},
 * {@code df.faissSearch(index, query, k)}.
 */
package org.bytedeco.pytorch.dataframe.faiss;
