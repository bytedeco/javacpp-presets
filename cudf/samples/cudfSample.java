import org.bytedeco.cuda.global.cudart;
import org.bytedeco.cudf.column;
import org.bytedeco.cudf.column.contents;
import org.bytedeco.cudf.column_view;
import org.bytedeco.cudf.data_type;
import org.bytedeco.cudf.global.cudf;
import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;

import java.nio.FloatBuffer;

public class cudfSample {

  public static void main(String[] args) {
    Loader.load(contents.class);
    float[] f = new float[]{Float.NaN, 2};
    FloatBuffer floatBuffer = FloatBuffer.wrap(f);


    FloatPointer fp = new FloatPointer(floatBuffer);
    column_view columnView = new column_view(new data_type(cudf.type_id.FLOAT32), (int)fp.limit(), fp);
    column c = cudf.is_nan(columnView);

    column_view conts = c.view();

    BoolPointer result = new BoolPointer(2);
    Loader.load(cudart.class);

    for (int i = 0 ; i < 2 ; i++) {
      System.out.println("value before: " + result.get(i));
    }
    cudart.cuMemcpyDtoH(result, conts.address(), 4);
    for (int i = 0 ; i < 2 ; i++) {
      System.out.println("value: " + result.get(i));
    }
 }


}