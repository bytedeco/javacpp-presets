import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.python.*;
import static org.bytedeco.cpython.global.python.*;

public class Simple {
    public static void main(String[] args) {
        Pointer program = Py_DecodeLocale(Simple.class.getSimpleName(), null);
        if (program == null) {
            System.err.println("Fatal error: cannot get class name");
            System.exit(1);
        }
        Py_SetProgramName(program);  /* optional but recommended */
        Py_Initialize();
        PyRun_SimpleStringFlags("from time import time,ctime\n"
                              + "print('Today is', ctime(time()))\n", null);
        if (Py_FinalizeEx() < 0) {
            System.exit(120);
        }
        PyMem_RawFree(program);
        System.exit(0);
    }
}
