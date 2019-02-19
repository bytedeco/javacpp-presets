//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example computes real matrix C=alpha*A*B+beta*C using Intel(R) MKL
*   function dgemm, where A, B, and C are matrices and alpha and beta are
*   scalars in double precision.
*
*   In this simple example, practices such as memory management, data alignment,
*   and I/O that are necessary for good programming style and high MKL
*   performance are omitted to improve readability.
********************************************************************************/

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.mkl.global.mkl_rt.*;

public class DGEMMExample {
    public static void main(String[] args) throws Exception {
        System.out.println("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
                         + " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
                         + " alpha and beta are double precision scalars\n");

        int m = 2000, p = 200, n = 1000;
        System.out.printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
                        + " A(%dx%d) and matrix B(%dx%d)\n\n", m, p, p, n);
        double alpha = 1.0, beta = 0.0;

        System.out.println(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
                         + " performance \n");
        DoublePointer A = new DoublePointer(MKL_malloc(m * p * Double.BYTES, 64));
        DoublePointer B = new DoublePointer(MKL_malloc(p * n * Double.BYTES, 64));
        DoublePointer C = new DoublePointer(MKL_malloc(m * n * Double.BYTES, 64));
        if (A.isNull() || B.isNull() || C.isNull()) {
            System.out.println( "\n ERROR: Can't allocate memory for matrices. Aborting... \n");
            MKL_free(A);
            MKL_free(B);
            MKL_free(C);
            System.exit(1);
        }

        System.out.println(" Intializing matrix data \n");
        DoubleIndexer Aidx = DoubleIndexer.create(A.capacity(m * p));
        for (int i = 0; i < m * p; i++) {
            A.put(i, (double)(i + 1));
        }

        DoubleIndexer Bidx = DoubleIndexer.create(B.capacity(p * n));
        for (int i = 0; i < p * n; i++) {
            B.put(i, (double)(-i - 1));
        }

        DoubleIndexer Cidx = DoubleIndexer.create(C.capacity(m * n));
        for (int i = 0; i < m * n; i++) {
            C.put(i, 0.0);
        }

        System.out.println(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, p, alpha, A, p, B, n, beta, C, n);
        System.out.println("\n Computations completed.\n");

        System.out.println(" Top left corner of matrix A: ");
        for (int i = 0; i < Math.min(m, 6); i++) {
            for (int j = 0; j< Math.min(p, 6); j++) {
                System.out.printf("%12.0f", Aidx.get(j + i * p));
            }
            System.out.println();
        }

        System.out.println("\n Top left corner of matrix B: ");
        for (int i = 0; i < Math.min(p, 6); i++) {
            for (int j = 0; j < Math.min(n, 6); j++) {
                System.out.printf("%12.0f", Bidx.get(j + i * n));
            }
            System.out.println();
        }

        System.out.println("\n Top left corner of matrix C: ");
        for (int i = 0; i < Math.min(m, 6); i++) {
            for (int j = 0; j < Math.min(n, 6); j++) {
                System.out.printf("%12.5G", Cidx.get(j + i * n));
            }
            System.out.println();
        }

        System.out.println("\n Deallocating memory \n");
        MKL_free(A);
        MKL_free(B);
        MKL_free(C);

        System.out.println(" Example completed. \n");
        System.exit(0);
    }
}
