// Based on code found in https://github.com/deeplearning4j/libnd4j/blob/master/blas/cpu/NativeBlas.cpp

#include <cblas.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

static int maxThreads = -1;
static int vendor = 0;

static void blas_set_num_threads(int num) {
    typedef void* (*void_int)(int);
    typedef int* (*int_int)(int);
    typedef int* (*int_int_int)(int, int);

    maxThreads = num;
#ifdef __MKL
    // if we're linked against mkl - just go for it
    MKL_Set_Num_Threads(num);
    MKL_Domain_Set_Num_Threads(num, 0); // MKL_DOMAIN_ALL
    MKL_Domain_Set_Num_Threads(num, 1); // MKL_DOMAIN_BLAS
    MKL_Set_Num_Threads_Local(num);
#elif __OPENBLAS
#ifdef _WIN32
    // for win32 we just check for mkl_rt.dll
    HMODULE handle = LoadLibrary("mkl_rt.dll");
    if (handle != NULL) {
        void_int mkl_global = (void_int) GetProcAddress(handle, "MKL_Set_Num_Threads");
        if (mkl_global != NULL) {
            mkl_global(num);

            vendor = 3;

            int_int_int mkl_domain = (int_int_int) GetProcAddress(handle, "MKL_Domain_Set_Num_Threads");
            if (mkl_domain != NULL) {
                mkl_domain(num, 0); // DOMAIN_ALL
                mkl_domain(num, 1); // DOMAIN_BLAS
            }

            int_int mkl_local = (int_int) GetProcAddress(handle, "MKL_Set_Num_Threads_Local");
            if (mkl_local != NULL) {
                mkl_local(num);
            }
        } else {
            printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
        }
        //FreeLibrary(handle);
    } else {
      // OpenBLAS path
      handle = LoadLibrary("libopenblas.dll");
      if (handle != NULL) {
        void_int oblas = (void_int) GetProcAddress(handle, "openblas_set_num_threads");
        if (oblas != NULL) {
            vendor = 2;
            oblas(num);
        } else {
            printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
        }
        //FreeLibrary(handle);
      } else {
        printf("Unable to guess runtime. Please set OMP_NUM_THREADS manually.\n");
      }
    }
#else
    // it's possible to have MKL being loaded at runtime
    void *handle = dlopen("libmkl_rt.so", RTLD_NOW|RTLD_GLOBAL);
    if (handle == NULL) {
        handle = dlopen("libmkl_rt.dylib", RTLD_NOW|RTLD_GLOBAL);
    }
    if (handle != NULL) {

        // we call for openblas only if libmkl isn't loaded, and openblas_set_num_threads exists
        void_int mkl_global = (void_int) dlsym(handle, "MKL_Set_Num_Threads");
        if (mkl_global != NULL) {
            // we're running against mkl
            mkl_global((int) num);

            vendor = 3;

            int_int_int mkl_domain = (int_int_int) dlsym(handle, "MKL_Domain_Set_Num_Threads");
            if (mkl_domain != NULL) {
                mkl_domain(num, 0); // DOMAIN_ALL
                mkl_domain(num, 1); // DOMAIN_BLAS
            }

            int_int mkl_local = (int_int) dlsym(handle, "MKL_Set_Num_Threads_Local");
            if (mkl_local != NULL) {
                mkl_local(num);
            }
        } else {
            printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
        }
        dlclose(handle);
    } else {
        // we're falling back to bundled OpenBLAS opening libopenblas.so.0
        handle = dlopen("libopenblas.so.0", RTLD_NOW|RTLD_GLOBAL);
        if (handle == NULL) {
            handle = dlopen("libopenblas.so", RTLD_NOW|RTLD_GLOBAL);
        }
        if (handle == NULL) {
            handle = dlopen("libopenblas.dylib", RTLD_NOW|RTLD_GLOBAL);
        }

        if (handle != NULL) {
            void_int oblas = (void_int) dlsym(handle, "openblas_set_num_threads");
            if (oblas != NULL) {
                vendor = 2;
                // we're running against openblas
                oblas((int) num);
            } else {
                printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
            }

            dlclose(handle);
        } else printf("Unable to guess runtime. Please set OMP_NUM_THREADS manually.\n");
    }
#endif

#else
    printf("Unable to guess runtime. Please set OMP_NUM_THREADS or equivalent manually.\n");
#endif
    fflush(stdout);
}


static int blas_get_num_threads() {
    return maxThreads;
}

/**
 *  0 - Unknown
 *  1 - cuBLAS
 *  2 - OpenBLAS
 *  3 - MKL
 */
static int blas_get_vendor() {
    return vendor;
}
