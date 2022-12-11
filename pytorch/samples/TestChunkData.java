import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class TestChunkData {
    public static void main(String[] args) throws Exception {
        try (PointerScope scope = new PointerScope()) {
            long batch_size = 10;
            long prefetch_count = 1;
            ChunkDataReader data_reader = new ChunkDataReader() {
                public ExampleVector read_chunk(long chunk_index) {
                    return new ExampleVector(
                            new Example(Tensor.create(100.0), Tensor.create(200.0)),
                            new Example(Tensor.create(300.0), Tensor.create(400.0)));
                }
                public long chunk_count() { return 1; }
                public void reset() { }
            };
            RandomSampler sampler = new RandomSampler(0);
            ChunkMapDataset data_set = new ChunkSharedBatchDataset(
                    new ChunkDataset(data_reader, sampler, sampler,
                            new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack());
            ChunkRandomDataLoader data_loader = new ChunkRandomDataLoader(
                    data_set, new DataLoaderOptions(batch_size));
            for (int epoch = 1; epoch <= 10; ++epoch) {
                for (ExampleIterator it = data_loader.begin(); !it.equals(data_loader.end()); it = it.increment()) {
                    Example batch = it.access();
                    System.out.println(batch.data().createIndexer() + " " + batch.target().createIndexer());
                }
            }
        }
    }
}
