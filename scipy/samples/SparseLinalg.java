import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class SparseLinalg {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        Py_Initialize(org.bytedeco.scipy.presets.scipy.cachePackages());
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        PyRun_StringFlags("import numpy as np\n"
                + "from scipy.linalg import eig, eigh\n"
                + "from scipy.sparse.linalg import eigs, eigsh\n"
                + "np.set_printoptions(suppress=True)\n"

                + "np.random.seed(0)\n"
                + "X = np.random.random((100,100)) - 0.5\n"
                + "X = np.dot(X, X.T) #create a symmetric matrix\n"

                + "evals_all, evecs_all = eigh(X)\n"
                + "evals_large, evecs_large = eigsh(X, 3, which='LM')\n"
                + "print(evals_all[-3:])\n"
                + "print(evals_large)\n"
                + "print(np.dot(evecs_large.T, evecs_all[:,-3:]))\n"

                + "evals_small, evecs_small = eigsh(X, 3, sigma=0, which='LM')\n"
                + "print(evals_all[:3])\n"
                + "print(evals_small)\n"
                + "print(np.dot(evecs_small.T, evecs_all[:,:3]))\n", Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
    }
}
