import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.libraw.LibRaw;
import org.bytedeco.libraw.libraw_output_params_t;

import static org.bytedeco.libraw.global.LibRaw.*;

public class LibRawDemo {
  public static void main(String[] args) {

    BytePointer version = LibRaw.version();
    System.out.println("LibRaw.version(): " + version.getString());

    LibRaw rawProcessor = new LibRaw();
    libraw_output_params_t params = rawProcessor.imgdata().params();

    String srcFile = "my_sample_image.dng";
    System.out.println("Reading: " + srcFile);
    int ret = rawProcessor.open_file(srcFile);
    if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
      BytePointer msg = libraw_strerror(ret);
      System.out.println("Cannot unpack " + srcFile + " : " + msg.getString());
      System.exit(ret);
    }

    System.out.println("Unpacking: " + srcFile);
    ret = rawProcessor.unpack();
    if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
      BytePointer msg = libraw_strerror(ret);
      System.out.println("Cannot unpack " + srcFile + " : " + msg.getString());
      System.exit(ret);
    }

    System.out.println("Processing");
    ret = rawProcessor.dcraw_process();
    if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
      BytePointer msg = libraw_strerror(ret);
      System.out.println("Cannot process : " + msg.getString());
      System.exit(ret);
    }

    String dstFile = "my_sample_image.ppm";
    System.out.println("Writing file: " + dstFile);
    ret = rawProcessor.dcraw_ppm_tiff_writer(dstFile);
    if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
      BytePointer msg = libraw_strerror(ret);
      System.out.println("Cannot write file : " + msg.getString());
      System.exit(ret);
    }

    System.out.println("Cleaning up");
    rawProcessor.recycle();

    System.out.println("Done");
    System.exit(LibRaw_errors.LIBRAW_SUCCESS.value);
  }
}