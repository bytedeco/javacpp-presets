/*
 * Copyright (C) 2019-2021 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.numpy.presets;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.cpython.presets.*;
import org.bytedeco.openblas.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = {openblas.class, python.class},
    value = {
        @Platform(
            cinclude = {
                "_numpyconfig.h",
                "numpyconfig.h",
//                "npy_config.h",
                "npy_common.h",
                "npy_os.h",
                "npy_cpu.h",
                "npy_endian.h",
                "npy_interrupt.h",
                "npy_math.h",
//                "npy_math_internal.h",
                "halffloat.h",
                "utils.h",
                "arrayobject.h",
                "arrayscalars.h",
                "ndarraytypes.h",
                "ndarrayobject.h",
                "__multiarray_api.h",
                "_neighborhood_iterator_imp.h",
//                "noprefix.h",
                "__ufunc_api.h",
                "ufuncobject.h",
            },
            exclude = {
                "__multiarray_api.h",
                "_neighborhood_iterator_imp.h",
                "__ufunc_api.h"
            },
            link = "npymath",
            resource = {"bin", "python", "scripts"}
        )
    },
    target = "org.bytedeco.numpy",
    global = "org.bytedeco.numpy.global.numpy"
)
@NoException
public class numpy implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "numpy"); }

    /** Returns {@code Loader.cacheResource("/org/bytedeco/numpy/" + Loader.getPlatform() + "/python/")}. */
    public static File cachePackage() throws IOException {
        return Loader.cacheResource("/org/bytedeco/numpy/" + Loader.getPlatform() + "/python/");
    }

    /** Returns {@code {python.cachePackages(), numpy.cachePackage()}}. */
    public static File[] cachePackages() throws IOException {
        File[] path = org.bytedeco.cpython.global.python.cachePackages();
        path = Arrays.copyOf(path, path.length + 1);
        path[path.length - 1] = cachePackage();
        return path;
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__multiarray_api.h").linePatterns("#define PyArray_GetNDArrayCVersion .*",
                                                                "         PyArray_API\\[303\\]\\)").skip())
               .put(new Info("__ufunc_api.h").linePatterns("#define PyUFunc_Type .*",
                                                           "         PyUFunc_API\\[42\\]\\)").skip())

               .put(new Info("NPY_VISIBILITY_HIDDEN", "NPY_GCC_UNROLL_LOOPS", "NPY_GCC_OPT_3",
                             "NPY_GCC_TARGET_AVX", "NPY_GCC_TARGET_FMA", "NPY_GCC_TARGET_AVX2", "NPY_GCC_TARGET_AVX512F", "NPY_GCC_TARGET_AVX512_SKX",
                             "NPY_INLINE", "NPY_TLS", "NPY_RETURNS_BORROWED_REF", "constchar",
                             "NPY_SIGINT_ON", "NPY_SIGINT_OFF", "NPY_INPLACE", "NPY_FINLINE",
                             "npy_fseek", "npy_ftell", "npy_lseek", "npy_off_t", "PyArrayDescr_Type",
                             "PyIntpArrType_Type", "PyUIntpArrType_Type", "MyPyLong_FromInt64", "MyPyLong_AsInt64",
                             "PyInt64ScalarObject", "PyInt64ArrType_Type", "PyUInt64ScalarObject", "PyUInt64ArrType_Type",
                             "PyInt32ScalarObject", "PyInt32ArrType_Type", "PyUInt32ScalarObject", "PyUInt32ArrType_Type",
                             "PyInt16ScalarObject", "PyInt16ArrType_Type", "PyUInt16ScalarObject", "PyUInt16ArrType_Type",
                             "PyInt8ScalarObject", "PyInt8ArrType_Type", "PyUInt8ScalarObject", "PyUInt8ArrType_Type",
                             "PyInt128ScalarObject", "PyInt128ArrType_Type", "PyUInt128ScalarObject", "PyUInt128ArrType_Type",
                             "PyFloat32ScalarObject", "PyComplex64ScalarObject", "PyFloat32ArrType_Type", "PyComplex64ArrType_Type",
                             "PyFloat64ScalarObject", "PyComplex128ScalarObject", "PyFloat64ArrType_Type", "PyComplex128ArrType_Type",
                             "PyFloat80ScalarObject", "PyComplex160ScalarObject", "PyFloat80ArrType_Type", "PyComplex160ArrType_Type",
                             "PyFloat96ScalarObject", "PyComplex192ScalarObject", "PyFloat96ArrType_Type", "PyComplex192ArrType_Type",
                             "PyFloat128ScalarObject", "PyComplex256ScalarObject", "PyFloat128ArrType_Type", "PyComplex256ArrType_Type",
                             "PyFloat256ScalarObject", "PyComplex512ScalarObject", "PyFloat256ArrType_Type", "PyComplex512ArrType_Type",
                             "NPY_MAX_INT128", "NPY_MIN_INT128", "NPY_MAX_UINT128", "NPY_MAX_INT256", "NPY_MIN_INT256", "NPY_MAX_UINT256",
                             "NPY_FLOAT32", "NPY_COMPLEX64", "NPY_COMPLEX64_FMT",
                             "NPY_FLOAT64", "NPY_COMPLEX128", "NPY_COMPLEX128_FMT",
                             "NPY_FLOAT80", "NPY_COMPLEX160", "NPY_COMPLEX160_FMT",
                             "NPY_FLOAT96", "NPY_COMPLEX192", "NPY_COMPLEX192_FMT",
                             "NPY_FLOAT128", "NPY_COMPLEX256", "NPY_COMPLEX256_FMT",
                             "NPY_FLOAT256", "NPY_COMPLEX512", "NPY_COMPLEX512_FMT",
                             "NPY_SIGJMP_BUF", "_npy_signbit_f", "_npy_signbit_d", "_npy_signbit_ld",
                             "npy_degrees", "npy_degreesf", "npy_degreesl", "npy_radians", "npy_radiansf", "npy_radiansl",
                             "PyStringScalarObject", /*"PyUnicodeScalarObject",*/ "__COMP_NPY_UNUSED",
                             "PyArrayScalar_False", "PyArrayScalar_True", "PyArrayScalar_RETURN_FALSE", "PyArrayScalar_RETURN_TRUE", "NPY_NO_EXPORT",
                             "PyArray_malloc", "PyArray_free", "PyArray_realloc",
                             "NPY_BEGIN_THREADS_DEF", "NPY_BEGIN_ALLOW_THREADS", "NPY_END_ALLOW_THREADS", "NPY_BEGIN_THREADS", "NPY_END_THREADS",
                             "NPY_ALLOW_C_API_DEF", "NPY_ALLOW_C_API", "NPY_DISABLE_C_API",
                             "PyArray_IsNativeByteOrder", "NPY_REFCOUNT", "NUMPY_IMPORT_ARRAY_RETVAL",
                             "NPY_LOOP_BEGIN_THREADS", "NPY_LOOP_END_THREADS", "NUMPY_IMPORT_UMATH_RETVAL", "UFUNC_NOFPE").cppTypes().annotations())

               .put(new Info("defined(_MSC_VER) && defined(_WIN64) && (_MSC_VER > 1400) ||"
                           + "    defined(__MINGW32__) || defined(__MINGW64__)",
                             "defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD",
                             "NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_INT",
                             "NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_LONG",
                             "NPY_BITSOF_LONG == 8",   "NPY_BITSOF_LONGLONG == 8",
                             "NPY_BITSOF_LONG == 16",  "NPY_BITSOF_LONGLONG == 16",
                             "NPY_BITSOF_LONG == 32",  "NPY_BITSOF_LONGLONG == 32",
                             "NPY_BITSOF_LONG == 128", "NPY_BITSOF_LONGLONG == 128",
                             "NPY_BITSOF_INT == 8",    "NPY_BITSOF_SHORT == 8",
                             "NPY_BITSOF_INT == 16",   "NPY_BITSOF_SHORT == 32",
                             "NPY_BITSOF_INT == 64",   "NPY_BITSOF_SHORT == 64",
                             "NPY_BITSOF_INT == 128",  "NPY_BITSOF_SHORT == 128", "NPY_BITSOF_CHAR == 128",
                             "defined(PY_ARRAY_UNIQUE_SYMBOL)", "defined(PY_UFUNC_UNIQUE_SYMBOL)").define(false))

               .put(new Info("NPY_BITSOF_LONG == 64", "NPY_BITSOF_LONGLONG == 64",
                             "NPY_BITSOF_INT == 32", "NPY_BITSOF_SHORT == 16").define(true))

               .put(new Info("NPY_MAX_INT", "INT_MIN", "NPY_MIN_INT", "NPY_MAX_UINT", "NPY_MAX_LONG", "NPY_MIN_LONG", "NPY_MAX_ULONG",
                             "NPY_INTP", "NPY_UINTP", "NPY_MAX_INTP", "NPY_MIN_INTP", "NPY_MAX_UINTP").translate(false).cppTypes("long"))

               .put(new Info("NPY_SIZEOF_SHORT", "NPY_SIZEOF_INT", "NPY_SIZEOF_LONG",
                             "NPY_MAX_BYTE", "NPY_MIN_BYTE", "NPY_MAX_UBYTE", "NPY_MAX_SHORT", "NPY_MIN_SHORT", "NPY_MAX_USHORT",
                             "NPY_BITSOF_CHAR", "NPY_BITSOF_BYTE", "NPY_BITSOF_SHORT", "NPY_BITSOF_INT", "NPY_BITSOF_LONG", "NPY_BITSOF_LONGLONG",
                             "NPY_BITSOF_INTP", "NPY_BITSOF_HALF", "NPY_BITSOF_FLOAT", "NPY_BITSOF_DOUBLE", "NPY_BITSOF_LONGDOUBLE",
                             "NPY_BITSOF_CFLOAT", "NPY_BITSOF_CDOUBLE", "NPY_BITSOF_CLONGDOUBLE", "NPY_BITSOF_DATETIME", "NPY_BITSOF_TIMEDELTA",
                             "NPY_INT64", "NPY_UINT64", "NPY_INT32", "NPY_UINT32", "NPY_INT16", "NPY_UINT16", "NPY_INT8", "NPY_UINT8",
                             "NPY_FLOAT32", "NPY_COMPLEX64", "NPY_FLOAT64", "NPY_COMPLEX128", "NPY_FLOAT80", "NPY_COMPLEX160",
                             "NPY_FLOAT96", "NPY_COMPLEX192", "NPY_FLOAT128", "NPY_COMPLEX256", "NPY_FLOAT16", "NPY_FLOAT256", "NPY_COMPLEX512",
                             "NPY_BYTE_ORDER", "NPY_LITTLE_ENDIAN", "NPY_BIG_ENDIAN").translate(false))

               .put(new Info("NPY_ATTR_DEPRECATE").cppText("#define NPY_ATTR_DEPRECATE()"))
               .put(new Info("NpyAuxData_tag").base("PyObject").pointerTypes("NpyAuxData"))
               .put(new Info("PyArrayObject_fields").base("PyObject").pointerTypes("PyArrayObject"))
               .put(new Info("PyArrayIterObject_tag").base("PyObject").pointerTypes("PyArrayIterObject"))
               .put(new Info("_arr_descr").pointerTypes("PyArray_ArrayDescr"))
               .put(new Info("_loop1d_info").pointerTypes("PyUFunc_Loop1d"))
               .put(new Info("_tagPyUFuncObject").base("PyObject").pointerTypes("PyUFuncObject"))
               .put(new Info("PyUFuncGenericFunction").valueTypes("PyUFuncGenericFunction").pointerTypes("@ByPtrPtr PyUFuncGenericFunction"))
               .put(new Info("PyArrayDescr_TypeFull").javaText(
                       "public static native @ByRef PyTypeObject PyArrayDescr_Type(); public static native void PyArrayDescr_Type(PyTypeObject setter);"))

               .put(new Info("PyArrayMapIter_Type", "PyArrayNeighborhoodIter_Type").skip())
        ;
    }
}
