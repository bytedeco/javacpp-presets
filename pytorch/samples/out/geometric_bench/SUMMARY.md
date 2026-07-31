# Geometric Component Benchmark Summary

Date: 2026-07-29

## Scope

- `org.bytedeco.pytorch.geometric.aggr`
- `org.bytedeco.pytorch.geometric.nn` (conv / norm / pooling / model / kge)
- `org.bytedeco.pytorch.geometric.transforms` (incl. HighOrder)

## Results

| Suite | PASS | FAIL | Total | Log |
|-------|------|------|-------|-----|
| AggregationBenchmark | 29 | 0 | 29 | AggregationBenchmark.log |
| NNConvBenchmark (core conv+norm+global pool) | 60 | 0 | 60 | NNConvBenchmark.log |
| TransformBenchmark | 53 | 0 | 53 | TransformBenchmark.log |
| SpecializedRemainingBenchmark | 46 | 0 | 46 | SpecializedRemainingBenchmark.log |
| DemoPooling (TopK/SAG/Global/KNN, MPS) | OK | 0 | — | DemoPooling.log |
| DemoPoolingAll (Edge/Graclus/SAG) | OK | 0 | — | DemoPoolingAll.log |
| Dense / Neighbor / Explainability pooling | OK | 0 | — | existing demos |

**Dedicated micro-benchmarks: 188/188 PASS.**

### SpecializedRemainingBenchmark coverage

- **CuGraph\***: CuGraphSAGEConv, CuGraphGATConv, CuGraphRGCNConv
- **FusedGATConv** (CSR/CSC via `toGraphFormat`)
- **Pseudo-coord**: GMMConv, SplineConv, GravNetConv, XConv
- **Point cloud**: PointNetConv, PPFConv, PointGNNConv, PointTransformerConv, DynamicEdgeConv
- **Specialized**: MixHop, DNA, PDN, NNConv, EGConv, RGAT, GeneralConv, DirGNN, GPS, AntiSymmetric, WLConv, HEAT
- **Hetero**: HANConv, HGTConv, HeteroConv
- **KGE**: TransE, DistMult, ComplEx, RotatE
- **nn.model**: GIN, GAT, PNA, LightGCN, GCNEncoder, NeuralFingerprint, GraphUNet, EdgeCNN, JumpingKnowledge, GAE, InnerProductDecoder
- **HighOrder**: AddMetaPaths, LargestConnectedComponents (real BFS, not stub)

## Bugs fixed this session (implementation)

### aggr
1. SortAggregation — dense `.indices()` misuse → real SortPool
2. SetTransformerAggregation — multi-seed PMA path

### transforms
3. LocalDegreeProfile — broken scatter → LDP 5-stats
4. IndexToMask — Bool `tensor(true)` → `index_fill_`
5. NormalizeRotation — undefined `new Tensor(N-1)` → Scalar + linalg_eigh
6. **LargestConnectedComponents** — was no-op stub → real weakly-connected BFS + relabel
7. **AddMetaPaths** — records `_metapath_len` for assertability

### nn/pooling
8. TopKPooling — CPU/MPS device mismatch on masks/maps
9. EdgePooling.greedyMatch — LongIndexer OOB → `data_ptr_long()`

### nn/model
10. **PNA / GCNEncoder** — only `forward(Tensor...)`; added `forward(x, edge_index)` for ModuleAsHelper dispatch
11. **GraphUNet** — `index_put_(TensorIndexVector(perm))` SIGSEGV → `index_copy_`
12. **GAE.encode** — hard cast to GCNConv → typed dispatch (GCNEncoder/GCNConv/GenericModule)
13. **JumpingKnowledge.cat** — `TensorVector.put` unreliable → array ctor
14. **InnerProductDecoder** — package-private → public

## How to re-run

```bash
FULL_CP="target/classes:$HOME/.m2/.../pytorch-…SNAPSHOT.jar:…-macosx-arm64.jar:$(cat /tmp/pytorch-cp.txt)"
java -cp "target/demo-compile:$FULL_CP" samples.demo.aggr.AggregationBenchmark
java -cp "target/demo-compile:$FULL_CP" samples.demo.layer.NNConvBenchmark
java -cp "target/demo-compile:$FULL_CP" samples.demo.transform.TransformBenchmark
java -cp "target/demo-compile:$FULL_CP" samples.demo.layer.SpecializedRemainingBenchmark
```
