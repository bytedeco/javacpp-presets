import org.bytedeco.javacpp.*;
import org.bytedeco.libffi.*;
import static org.bytedeco.libffi.global.ffi.*;

public class SimpleExample {
     static Pointer puts = Loader.addressof("puts");

     public static void main(String[] a) {
       ffi_cif cif = new ffi_cif();
       PointerPointer<ffi_type> args = new PointerPointer<>(1);
       PointerPointer<PointerPointer> values = new PointerPointer<>(1);
       PointerPointer<BytePointer> s = new PointerPointer<>(1);
       LongPointer rc = new LongPointer(1);

       /* Initialize the argument info vectors */
       args.put(0, ffi_type_pointer());
       values.put(0, s);

       /* Initialize the cif */
       if (ffi_prep_cif(cif, FFI_DEFAULT_ABI(), 1,
                        ffi_type_sint(), args) == FFI_OK)
         {
           s.putString("Hello World!");
           ffi_call(cif, puts, rc, values);
           /* rc now holds the result of the call to puts */

           /* values holds a pointer to the function's arg, so to
              call puts() again all we need to do is change the
              value of s */
           s.putString("This is cool!");
           ffi_call(cif, puts, rc, values);
         }

       System.exit(0);
     }
}
