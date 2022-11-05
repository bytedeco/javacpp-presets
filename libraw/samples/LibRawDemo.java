import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.libraw.LibRaw;
import org.bytedeco.libraw.libraw_output_params_t;

import static org.bytedeco.libraw.global.LibRaw.*;

public class LibRawDemo {
    public static String libRawVersion() {
        try (BytePointer version = LibRaw.version()) {
            return version.getString();
        }
    }

    public static void handleError(int err, String message) {
        if (err != LibRaw_errors.LIBRAW_SUCCESS.value) {
            final String msg;
            try (BytePointer e = libraw_strerror(err)) {
                msg = e.getString();
            }
            System.err.println(message + " : " + msg);
            System.exit(err);
        }
    }

    public static void main(String[] args) {
        System.out.println("");
        System.out.println("LibRaw.version(): " + libRawVersion());

        try (LibRaw rawProcessor = new LibRaw()) {
            // Set processing parameters
            libraw_output_params_t params = rawProcessor.imgdata().params();
            params.half_size(1); // Create half size image
            params.output_tiff(1); // Save as TIFF

            String srcFile = "my_sample_image.dng";
            System.out.println("Reading: " + srcFile);
            int ret = rawProcessor.open_file(srcFile);
            handleError(ret, "Cannot open " + srcFile);

            System.out.println("Unpacking: " + srcFile);
            ret = rawProcessor.unpack();
            handleError(ret, "Cannot unpack " + srcFile);

            System.out.println("Processing");
            ret = rawProcessor.dcraw_process();
            handleError(ret, "Cannot process" + srcFile);

            String dstFile = "my_sample_image.tif";
            System.out.println("Writing file: " + dstFile);
            ret = rawProcessor.dcraw_ppm_tiff_writer(dstFile);
            handleError(ret, "Cannot write " + dstFile);

            System.out.println("Cleaning up");
            rawProcessor.recycle();
        }

        System.out.println("Done");
        System.exit(LibRaw_errors.LIBRAW_SUCCESS.value);
    }
}
