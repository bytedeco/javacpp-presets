JavaCPP Presets for Accelerate BLAS
============================

Introduction
------------
This directory contains the JavaCPP Presets module for Accelerate BLAS.


Documentation
-------------
macOS has built-in BLAS library in Accelerate Framework.

As a result, unlike other JavaCPP Presets, one does not need to build the native library using cppbuild.sh.


Sample Usage
------------
Here is a simple example of BLAS matrix multiplication function `cblas_dgemm`:

```
import org.bytedeco.javacpp.accelerateblas;

public class ExampleDGEMMrowmajor {
    
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
    
        /* Locals */
        double[] A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        double[] B = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        double[] C = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        int M, N;
        
        /* Initialization */
        M = 3;
        N = 3;
        
        /* Print Matrix */
        print_matrix_rowmajor("Matrix A", M, N, A, M);
        
        /* Print Matrix */
        print_matrix_rowmajor("Matrix B", M, N, B, M);
        
        /* Print Matrix */
        print_matrix_rowmajor("Matrix C", M, N, C, M);
        System.out.println();
        
        /* Executable statements */
        System.out.println("cblas_dgemm Results: A * B = C");
        accelerateblas.cblas_dgemm(accelerateblas.CblasRowMajor, accelerateblas.CblasNoTrans, accelerateblas.CblasNoTrans, M, N, M, 1, A, M, B, M, 1, C, M);
        
        /* Print Result Matrix */
        print_matrix_rowmajor("Matrix C", M, N, C, M);

        System.out.println();
        System.exit(0);
    }
}
```

Using Gradle

`build.gradle`

```
buildscript {
    ext {
        javacpp_version = "1.3.3-SNAPSHOT"
    }
}

apply plugin: "application"

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    compile("org.bytedeco:javacpp:${javacpp_version}")
    compile("org.bytedeco.javacpp-presets:accelerateblas:10.12.5-${javacpp_version}")
    compile("org.bytedeco.javacpp-presets:accelerateblas:10.12.5-${javacpp_version}:macosx-x86_64")
}

run {
    mainClassName = "ExampleDGEMMrowmajor"
}
```

Run the example:
```
./gradlew run
```

will print out
```
 Matrix A
   1.00   2.00   3.00
   4.00   5.00   6.00
   7.00   8.00   9.00
   
 Matrix B
   1.00   2.00   3.00
   4.00   5.00   6.00
   7.00   8.00   9.00
   
 Matrix C
   0.00   0.00   0.00
   0.00   0.00   0.00
   0.00   0.00   0.00
   
 cblas_dgemm Results: A * B = C
 
 Matrix C
  30.00  36.00  42.00
  66.00  81.00  96.00
 102.00 126.00 150.00
```