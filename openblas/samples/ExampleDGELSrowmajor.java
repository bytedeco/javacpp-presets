/*
   LAPACKE Example : Calling DGELS using row-major order
   =====================================================
 
   The program computes the solution to the system of linear
   equations with a square matrix A and multiple
   right-hand sides B, where A is the coefficient matrix
   and b is the right-hand side matrix:
  
   Description
   ===========
 
   In this example, we wish solve the least squares problem min_x || B - Ax || 
   for two right-hand sides using the LAPACK routine DGELS. For input we will
   use the 5-by-3 matrix

         ( 1  1  1 )
         ( 2  3  4 )
     A = ( 3  5  2 )
         ( 4  2  5 )
         ( 5  4  3 )
    and the 5-by-2 matrix

         ( -10 -3 )
         (  12 14 )
     B = (  14 12 )
         (  16 16 )
         (  18 16 )
    We will first store the input matrix as a static C two-dimensional array,
    which is stored in row-major order, and let LAPACKE handle the work space
    array allocation. The LAPACK base name for this function is gels, and we 
    will use double precision (d), so the LAPACKE function name is LAPACKE_dgels.

    thus lda=3 and ldb=2. The output for each right hand side is stored in b as
    consecutive vectors of length 3. The correct answer for this problem is 
    the 3-by-2 matrix

         ( 2 1 )
         ( 1 1 )
         ( 1 2 )

    A complete C program for this example is given below. Note that when the arrays
     are passed to the LAPACK routine, they must be dereferenced, since LAPACK is
      expecting arrays of type double *, not double **.


   LAPACKE Interface
   =================

   LAPACKE_dgels (row-major, high-level) Example Program Results

  -- LAPACKE Example routine (version 3.5.0) --
  -- LAPACK is a software package provided by Univ. of Tennessee,    --
  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
     February 2012

*/
/* Calling DGELS using row-major order */

/* Includes */
import static org.bytedeco.openblas.global.openblas.*;

public class ExampleDGELSrowmajor {
    /* Auxiliary routine: printing a matrix */
    static void print_matrix_rowmajor(String desc, int m, int n, double[] mat, int ldm) {
        int i, j;
        System.out.printf("\n %s\n", desc);

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) System.out.printf(" %6.2f", mat[i*ldm+j]);
            System.out.printf("\n");
        }
    }


    /* Auxiliary routine: printing a matrix */
    static void print_matrix_colmajor(String desc, int m, int n, double[] mat, int ldm) {
        int i, j;
        System.out.printf("\n %s\n", desc);

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) System.out.printf(" %6.2f", mat[i+j*ldm]);
            System.out.printf("\n");
        }
    }

    /* Auxiliary routine: printing a vector of integers */
    static void print_vector(String desc, int n, int[] vec) {
        int j;
        System.out.printf("\n %s\n", desc);
        for (j = 0; j < n; j++) System.out.printf(" %6i", vec[j]);
        System.out.printf("\n");
    }

    /* Main program */
    public static void main(String[] args) {
        blas_set_num_threads(4);
        System.out.println("vendor = " + blas_get_vendor() + ", num_threads = " + blas_get_num_threads());

        /* Locals */
        double[] A = {1, 1, 1, 2, 3, 4, 3, 5, 2, 4, 2, 5, 5, 4, 3};
        double[] b = {-10, -3, 12, 14, 14, 12, 16, 16, 18, 16};
        int info, m, n, lda, ldb, nrhs;
        int i, j;

        /* Initialization */
        m = 5;
        n = 3;
        nrhs = 2;
        lda = 3;
        ldb = 2;

        /* Print Entry Matrix */
        print_matrix_rowmajor("Entry Matrix A", m, n, A, lda);
        /* Print Right Rand Side */
        print_matrix_rowmajor("Right Hand Side b", n, nrhs, b, ldb);
        System.out.println();

        /* Executable statements */
        System.out.println("LAPACKE_dgels (row-major, high-level) Example Program Results");
        /* Solve least squares problem*/
        info = LAPACKE_dgels(LAPACK_ROW_MAJOR, (byte)'N', m, n, nrhs, A, lda, b, ldb);

        /* Print Solution */
        print_matrix_rowmajor("Solution", n, nrhs, b, ldb);
        System.out.println();
        System.exit(0);
    }
} /* End of LAPACKE_dgels Example */
