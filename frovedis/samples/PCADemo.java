import java.util.Arrays;
import java.util.List;
import scala.Tuple2;

import com.nec.frovedis.matrix.FrovedisPCAModel;
import com.nec.frovedis.matrix.RowMatrixUtils;
import org.bytedeco.frovedis.frovedis_server;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

public class PCADemo {
  public static void main(String[] args) throws Exception {
    Logger.getLogger("org").setLevel(Level.ERROR);

    // -------- configurations --------
    SparkConf conf = new SparkConf().setAppName("PCADemo").setMaster("local[2]");
    SparkContext sc = new SparkContext(conf);
    JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

    // initializing Frovedis server with "personalized command", if provided in command line
    frovedis_server.initialize(args.length != 0 ? args[0] : "-np 1");

    List<Vector> data = Arrays.asList(
            Vectors.dense(1.0, 0.0, 7.0, 0.0, 0.0),
            Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
            Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    );
    JavaRDD<Vector> rows = jsc.parallelize(data);
    RowMatrix mat = new RowMatrix(rows.rdd());

    // (SPARK WAY) Compute the top 2 principal components.
    System.out.println("\nComputation using Spark native APIs:");
    Matrix s_pc1 = mat.computePrincipalComponents(2);
    System.out.println("Principal Components: ");
    System.out.println(s_pc1);

    // with variance
    System.out.println("\nWith variance: ");
    Tuple2<Matrix, Vector> s_pc2var = mat.computePrincipalComponentsAndExplainedVariance(2);
    Matrix s_pc2 = s_pc2var._1;
    Vector s_var = s_pc2var._2;
    System.out.println("Principal Components: ");
    System.out.println(s_pc2);
    System.out.println("Variance: ");
    System.out.println(s_var);

    // (FROVEDIS WAY) Compute the top 2 principal components.
    System.out.println("\n\nComputation using Frovedis APIs getting called from Spark client:");
    FrovedisPCAModel res1 = RowMatrixUtils.computePrincipalComponents(mat,2); // res: Frovedis side result pointer
    Matrix f_pc1 = res1.to_spark_result()._1;
    System.out.println("Principal Components: ");
    System.out.println(f_pc1);

    // with variance
    System.out.println("\nWith variance: ");
    FrovedisPCAModel res2 = RowMatrixUtils.computePrincipalComponentsAndExplainedVariance(mat,2);
    Tuple2<Matrix, Vector> f_pc2var = res2.to_spark_result();
    Matrix f_pc2 = f_pc2var._1;
    Vector f_var = f_pc2var._2;
    System.out.println("Principal Components: ");
    System.out.println(f_pc2);
    System.out.println("Variance: ");
    System.out.println(f_var);

    frovedis_server.shut_down();
    jsc.stop();
  }
}
