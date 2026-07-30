import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Column;

/**
 * Smoke-test nested LIST parquet reading (MicroLens fixed_size_list&lt;int64&gt;[64]).
 *
 * Run (from pytorch module root):
 *   javac -cp "target/classes:$(echo target/dependency/*.jar | tr ' ' ':')" \
 *         samples/TestParquetListRead.java -d /tmp/df-test
 *   java  -cp "/tmp/df-test:target/classes:$(echo target/dependency/*.jar | tr ' ' ':')" \
 *         TestParquetListRead [path/to/valid.parquet]
 */
public class TestParquetListRead {
    public static void main(String[] args) throws Exception {
        String path = args.length > 0
            ? args[0]
            : "/Users/muller/Documents/code/cpp/VideoMMCTR/data/MicroLens_1M_x1/valid.parquet";

        System.out.println("Reading: " + path);
        DataFrame df = DataFrame.readParquet(path);

        System.out.println("\n=== printSchema() ===");
        df.printSchema();

        System.out.println("\n=== info() ===");
        df.info();

        System.out.println("\n=== rowCount / countRows ===");
        System.out.println("rowCount()  = " + df.rowCount());
        System.out.println("countRows() = " + df.countRows());
        // Note: count() is pandas-style non-null counts per column
        System.out.println("count()     = " + df.count());

        System.out.println("\n=== show(5) ===");
        df.show(5);

        // Verify item_seq is a long[64]
        Column itemSeq = df.column("item_seq");
        System.out.println("\n=== item_seq dtype / first cell ===");
        System.out.println("dtype = " + itemSeq.dtype());
        Object cell0 = itemSeq.get(0);
        System.out.println("cell0 class = " + (cell0 == null ? "null" : cell0.getClass().getName()));
        if (cell0 instanceof long[]) {
            long[] a = (long[]) cell0;
            System.out.println("long[" + a.length + "] = " + java.util.Arrays.toString(a));
        } else {
            System.out.println("cell0 = " + cell0);
        }

        System.out.println("\n=== describeFrame() ===");
        DataFrame desc = df.describeFrame();
        desc.show(desc.rowCount());

        // Round-trip: write + read back LIST column
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("df-list-rt-", ".parquet");
        try {
            System.out.println("\n=== writeParquet LIST round-trip → " + tmp + " ===");
            df.head(3).writeParquet(tmp.toString());
            DataFrame back = DataFrame.readParquet(tmp.toString());
            back.printSchema();
            back.show(3);
            Object b0 = back.column("item_seq").get(0);
            System.out.println("round-trip cell0 class = " + (b0 == null ? "null" : b0.getClass().getName()));
            if (b0 instanceof long[]) {
                System.out.println("round-trip long[" + ((long[]) b0).length + "] ok");
                System.out.println("equal to original? " + java.util.Arrays.equals((long[]) cell0, (long[]) b0));
            }
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }

        System.out.println("\nOK");
    }
}
