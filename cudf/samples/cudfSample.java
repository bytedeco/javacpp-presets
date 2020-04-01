import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.cudf.column;
import org.bytedeco.cudf.column_view;
import org.bytedeco.cudf.data_type;
import org.bytedeco.cudf.global.cudf;
import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;

public class cudfSample {

  static final int EXIT_FAILURE = 1;
  static final int EXIT_SUCCESS = 0;
  static final int EXIT_WAIVED = 0;

  static void FatalError(String s) {
    System.err.println(s);
    Thread.dumpStack();
    System.err.println("Aborting...");
    cudaDeviceReset();
    System.exit(EXIT_FAILURE);
  }

  static void checkCudaErrors(int status) {
    if (status != 0) {
      FatalError("Cuda failure: " + cudaGetErrorString(status).getString());
    }
  }

  static void printDeviceVector(int size, FloatPointer vec_d) {
    FloatPointer vec = new FloatPointer(size);
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*Float.BYTES, cudaMemcpyDeviceToHost);
    System.out.print("device vector: ");
    for (int i = 0; i < size; i++) {
      System.out.print(vec.get(i) + " ");
    }
    System.out.println();
  }

  static void printHostVector(BoolPointer vec) {
    System.out.print("result: ");
    for (int i = 0; i < vec.limit(); i++) {
      System.out.print(vec.get(i) + " ");
    }
    System.out.println();
  }

  public static void main(String[] args) {
    Loader.load(column_view.class);
    // initialize host data
    try (FloatPointer fp = new FloatPointer(new float[]{Float.NaN, 3.224f})) {
      // pointer to hold the device pointer
      FloatPointer srcDevicePointer = new FloatPointer();
      // allocate memory on device
      checkCudaErrors(cudaMalloc(srcDevicePointer, fp.limit() * Float.BYTES));
      // copy host data to device pointer
      checkCudaErrors(cudaMemcpy(srcDevicePointer, fp, fp.limit() * Float.BYTES, cudaMemcpyHostToDevice));
      // create a column_view wrapper
      column_view columnView = new column_view(new data_type(cudf.type_id.FLOAT32), (int) fp.limit(), srcDevicePointer);
      // call unary operation
      column c = cudf.is_nan(columnView);
      // get the view to inspect the column
      column_view view = c.view();
      // create a host pointer to copy the result
      BoolPointer result = new BoolPointer(fp.limit());
      cudaDeviceSynchronize();
      // copy the data from device to host
      checkCudaErrors(cudaMemcpy(result, view.dataBoolean(), fp.limit(), cudaMemcpyDeviceToHost));
      // print result
      printDeviceVector((int)fp.limit(), srcDevicePointer);
      printHostVector(result);
      // clean up
      checkCudaErrors(cudaFree(srcDevicePointer));
      checkCudaErrors(cudaFree(view.dataBoolean()));
    }
 }
}
