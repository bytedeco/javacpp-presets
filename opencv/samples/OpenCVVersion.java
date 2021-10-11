import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class OpenCVVersion {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        Py_Initialize(org.bytedeco.opencv.opencv_python3.cachePackages());
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        int argc = args.length + 1;
        PointerPointer argv = new PointerPointer(argc);
        argv.put(0, Py_DecodeLocale(OpenCVVersion.class.getSimpleName(), null));
        for (int i = 1; i < argc; i++) {
            argv.put(i, Py_DecodeLocale(args[i - 1], null));
        }
        PySys_SetArgv(argc, argv);
        PyRun_StringFlags("\n"
                + "'''\n"
                + "prints OpenCV version\n"
                + "\n"
                + "Usage:\n"
                + "    opencv_version.py [<params>]\n"
                + "    params:\n"
                + "        --build: print complete build info\n"
                + "        --help:  print this help\n"
                + "'''\n"
                + "\n"
                + "import numpy as np\n"
                + "import cv2 as cv\n"
                + "\n"
                + "def main():\n"
                + "    import sys\n"
                + "\n"
                + "    try:\n"
                + "        param = sys.argv[1]\n"
                + "    except IndexError:\n"
                + "        param = \"\"\n"
                + "\n"
                + "    if \"--build\" == param:\n"
                + "        print(cv.getBuildInformation())\n"
                + "    elif \"--help\" == param:\n"
                + "        print(\"\t--build\\\n\t\tprint complete build info\")\n"
                + "        print(\"\t--help\\\n\t\tprint this help\")\n"
                + "    else:\n"
                + "        print(\"Welcome to OpenCV\")\n"
                + "\n"
                + "    print('Done')\n"
                + "\n"
                + "\n"
                + "if __name__ == '__main__':\n"
                + "    print(__doc__)\n"
                + "    main()\n"
                + "    cv.destroyAllWindows()\n", Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }

        for (int i = 0; i < argc; i++) {
            PyMem_RawFree(argv.get(i));
        }
    }
}
