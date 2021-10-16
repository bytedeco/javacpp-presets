import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class MatMul {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        Py_Initialize(org.bytedeco.numpy.global.numpy.cachePackages());
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        long[] dimsx = {2, 2};
        DoublePointer datax = new DoublePointer(1, 2, 3, 4);
        PyObject x = PyArray_New(PyArray_Type(), dimsx.length, new SizeTPointer(dimsx),
                                 NPY_DOUBLE, null, datax, 0, NPY_ARRAY_CARRAY, null);
        PyDict_SetItemString(globals, "x", x);
        System.out.println("x = " + DoubleIndexer.create(datax, dimsx));

        PyRun_StringFlags("import numpy; y = numpy.matmul(x, x)", Py_single_input, globals, globals, null);

        PyArrayObject y = new PyArrayObject(PyDict_GetItemString(globals, "y"));
        DoublePointer datay = new DoublePointer(PyArray_BYTES(y)).capacity(PyArray_Size(y));
        long[] dimsy = new long[PyArray_NDIM(y)];
        PyArray_DIMS(y).get(dimsy);
        System.out.println("y = " + DoubleIndexer.create(datay, dimsy));
    }
}
