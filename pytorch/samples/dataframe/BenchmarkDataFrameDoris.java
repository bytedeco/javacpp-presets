package dataframe;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.dataframe.DataFrame;

/**
 * Simple benchmark / smoke test for connecting to Apache Doris via MySQL protocol (JDBC).
 *
 * Requirements:
 *  - MySQL JDBC driver on the classpath (e.g. mysql-connector-java)
 *
 * Example (Doris connection from user request):
 *  jdbc:mysql://192.168.113.244:9030/  user=root  no password
 */
public class BenchmarkDataFrameDoris {
    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameDoris ===");
        String jdbc = "jdbc:mysql://192.168.113.244:9030/"; // provided in request
        String user = "root";
        String pass = "";

        try {
            DataFrame df = DataFrame.readMySQL(jdbc, "SELECT 1 AS n", user, pass);
            System.out.println("rows=" + df.rowCount());
            if (df.rowCount() > 0) {
                Object v = df.get(0, "n");
                System.out.println("value n=" + v);
                System.out.println("Doris/MySQL JDBC read seems successful.");
                System.exit(0);
            } else {
                System.err.println("No rows returned");
                System.exit(2);
            }
        } catch (Throwable t) {
            t.printStackTrace();
            System.err.println("Failed to query Doris via JDBC: " + t.getMessage());
            System.exit(3);
        }
    }
}
