/*
 * Copyright (C) 2018 Samuel Audet
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

package org.bytedeco.cpython.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    value = {
        @Platform(
            includepath = "/usr/include/python3.6m/",
            linkpath = "/usr/lib64/",
            cinclude = {
                "Python.h",
                "patchlevel.h",
                "pyconfig.h",
                "pymacconfig.h",
                "pyport.h",
                "pymacro.h",
//                "pyatomic.h",
                "pymath.h",
                "pytime.h",
                "pymem.h",

                "object.h",
                "objimpl.h",
                "typeslots.h",
                "pyhash.h",
                "pydebug.h",

                "descrobject.h",
                "bytearrayobject.h",
                "bytesobject.h",
                "unicodeobject.h",
                "longobject.h",
                "longintrepr.h",
                "boolobject.h",
                "floatobject.h",
                "complexobject.h",
                "rangeobject.h",
                "memoryobject.h",
                "tupleobject.h",
                "listobject.h",
                "dictobject.h",
                "structmember.h",
                "odictobject.h",
                "enumobject.h",
                "setobject.h",
                "methodobject.h",
                "moduleobject.h",
                "funcobject.h",
                "classobject.h",
                "fileobject.h",
                "pycapsule.h",
                "traceback.h",
                "sliceobject.h",
                "cellobject.h",
                "iterobject.h",
                "genobject.h",
                "warnings.h",
                "weakrefobject.h",
                "structseq.h",
                "namespaceobject.h",

                "codecs.h",
                "pyerrors.h",
                "pyarena.h",
                "pystate.h",
                "modsupport.h",
                "ceval.h",
                "sysmodule.h",
                "osmodule.h",
                "intrcheck.h",
                "import.h",

                "abstract.h",
                "bltinmodule.h",
                "asdl.h",
                "Python-ast.h",
                "node.h",
                "code.h",
                "compile.h",
                "symtable.h",
                "pythonrun.h",
                "pylifecycle.h",
                "eval.h",

                "pyctype.h",
                "pystrtod.h",
                "pystrcmp.h",
                "dtoa.h",
                "fileutils.h",
//                "pyfpe.h",
            },
            link = "python3.6m@.1.0"
        ),
        @Platform(
            value = "macosx",
            includepath = "/Library/Frameworks/Python.framework/Versions/3.6/Headers/",
            linkpath = "/Library/Frameworks/Python.framework/Versions/3.6/lib/",
            link = "python3.6"
        ),
        @Platform(
            value = "windows",
            includepath = "C:/Program Files/Python36/include/",
            linkpath = "C:/Program Files/Python36/libs/",
            preloadpath = "C:/Program Files/Python36/",
            link = "python36",
            preload = "python36"
        ),
    },
    target = "org.bytedeco.cpython.python",
    global = "org.bytedeco.cpython.global.python"
)
@NoException
public class python implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("Python-ast.h").linePatterns("#define Module.*",
                                                          "int PyAST_Check.*").skip())

               .put(new Info("COMPILER", "NTDDI_VERSION", "Py_NTDDI", "Py_IS_NAN",
                             "copysign", "hypot", "timezone", "daylight", "tzname",
                             "RETSIGTYPE", "_Py_COUNT_ALLOCS_COMMA", "Py_None", "Py_NotImplemented",
                             "PY_LONG_LONG", "PY_UINT32_T", "PY_UINT64_T", "PY_INT32_T", "PY_INT64_T",
                             "PY_FORMAT_SIZE_T", "Py_MEMCPY", "PyMODINIT_FUNC", "Py_VA_COPY",
                             "__inline__", "Py_HUGE_VAL", "Py_FORCE_DOUBLE", "Py_NAN",
                             "PyMem_Del", "PyMem_DEL", "PyDescr_COMMON", "PY_UNICODE_TYPE",
                             "PyObject_MALLOC", "PyObject_REALLOC", "PyObject_FREE", "PyObject_Del", "PyObject_DEL",
                             "_PyUnicode_AsStringAndSize", "_PyUnicode_AsString",
                             "PyLong_FromPid", "PyLong_AsPid", "PyLong_AsLong",
                             "Py_False", "Py_True", "Py_RETURN_TRUE", "Py_RETURN_FALSE", "Py_RETURN_NAN",
                             "PyObject_HEAD", "PyObject_VAR_HEAD", "Py_RETURN_NONE", "Py_RETURN_NOTIMPLEMENTED",
                             "PyModuleDef_HEAD_INIT", "_Py_atomic_address", "__declspec",
                             "PyException_HEAD", "_Py_NO_RETURN", "Py_Ellipsis",
                             "PyObject_Length", "PySequence_Length", "PySequence_In", "PyMapping_Length",
                             "Py_ALLOW_RECURSION", "Py_END_ALLOW_RECURSION",
                             "Py_BEGIN_ALLOW_THREADS", "Py_END_ALLOW_THREADS",
                             "Py_BLOCK_THREADS", "Py_UNBLOCK_THREADS", "PyOS_strnicmp", "PyOS_stricmp").cppTypes().annotations())

               .put(new Info("HAVE_GCC_ASM_FOR_X87",
                             "defined(_MSC_VER) && !defined(_WIN64)",
                             "HAVE_GCC_ASM_FOR_MC68881",
                             "HAVE__GETPTY",
                             "defined(HAVE_OPENPTY) || defined(HAVE_FORKPTY)",
                             "defined _MSC_VER && _MSC_VER >= 1900",
                             "COUNT_ALLOCS",
                             "HAVE_DLOPEN",
                             "SOLARIS",
                             "MS_WINDOWS",
                             "defined(HAVE_CLOCK_GETTIME) || defined(HAVE_KQUEUE)",
                             "X87_DOUBLE_ROUNDING",
                             "Py_DEBUG",
                             "defined(MS_WIN32) && !defined(HAVE_SNPRINTF)",
                             "defined(MS_WINDOWS) && !defined(Py_LIMITED_API)",
                             "!defined(PY_SSIZE_T_CLEAN) || !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000",
                             "PY_SSIZE_T_CLEAN").cppTypes().define(false))

               .put(new Info("!defined(__INTEL_COMPILER)").define(true))

               .put(new Info("COMPILER", "PY_LLONG_MIN", "PY_LLONG_MAX", "PY_ULLONG_MAX",
                             "SIZEOF_PY_HASH_T", "SIZEOF_PY_UHASH_T",
                             "PY_SIZE_MAX", "PY_SSIZE_T_MAX", "PY_SSIZE_T_MIN",
                             "LONG_BIT", "PyLong_BASE", "PyLong_MASK", "Py_UNICODE_SIZE").translate(false))

               .put(new Info("PyHash_FuncDef").purify())

               .put(new Info("wchar_t").cast().valueTypes("char", "int").pointerTypes("Pointer"))
               .put(new Info("timeval", "timespec", "tm").cast().pointerTypes("Pointer"))
               .put(new Info("atomic_uintptr_t", "atomic_int").cast().pointerTypes("Pointer"))
               .put(new Info("Py_ssize_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("_typeobject").cast().pointerTypes("PyTypeObject"))
               .put(new Info("_dictkeysobject").cast().pointerTypes("PyDictKeysObject"))
               .put(new Info("_gc_head").cast().pointerTypes("PyGC_Head"))
               .put(new Info("_traceback").cast().pointerTypes("PyTracebackObject"))
               .put(new Info("_co_extra_state").cast().pointerTypes("__PyCodeExtraState"))
               .put(new Info("_node").cast().pointerTypes("node"))

               .put(new Info("PyThreadFrameGetter", "jmp_buf").cast().pointerTypes("Pointer"))

               .put(new Info("_Py_memory_order", "PyThreadState::_preserve_36_ABI_1", "PyThreadState::_preserve_36_ABI_2",
                             "_PyGC_generation0", "_PyBytes_InsertThousandsGroupingLocale",
                             "_PyBytes_InsertThousandsGrouping", "_PyUnicode_DecodeUnicodeInternal",
                             "_PyFloat_Repr", "_PyFloat_Digits", "_PyFloat_DigitsInit",
                             "PySortWrapper_Type", "PyCmpWrapper_Type", "_PyGen_yf", "_PyAIterWrapper_New",
                             "_PyCoro_GetAwaitableIter", "_PyAsyncGenValueWrapperNew", "PyAsyncGen_ClearFreeLists",
                             "PyStructSequence_UnnamedField", "PySignal_SetWakeupFd",
                             "_PyArg_Fini", "PyInit_imp", "PyNullImporter_Type", "PyBuffer_SizeFromFormat",
                             "__PyCodeExtraState::co_extra_freefuncs", "_PyDict_NewKeysForClass", "_PyDictView_New",
                             "_PyDict_KeysSize", "_PyDict_SizeOf", "_PyDict_Pop_KnownHash", "_PyDict_FromKeys",
                             "_PyObjectDict_SetItem", "_PyDict_LoadGlobal", "__PyCodeExtraState_Get",
                             "_Py_asdl_seq_new", "_Py_asdl_int_seq_new").skip())

               .put(new Info("mod_ty").valueTypes("_mod").pointerTypes("@ByPtrPtr _mod"))
               .put(new Info("stmt_ty").valueTypes("_stmt").pointerTypes("@ByPtrPtr _stmt"))
               .put(new Info("expr_ty").valueTypes("_expr").pointerTypes("@ByPtrPtr _expr"))
               .put(new Info("slice_ty").valueTypes("_slice").pointerTypes("@ByPtrPtr _slice"))
               .put(new Info("comprehension_ty").valueTypes("_comprehension").pointerTypes("@ByPtrPtr _comprehension"))
               .put(new Info("excepthandler_ty").valueTypes("_excepthandler").pointerTypes("@ByPtrPtr _excepthandler"))
               .put(new Info("arguments_ty").valueTypes("_arguments").pointerTypes("@ByPtrPtr _arguments"))
               .put(new Info("arg_ty").valueTypes("_arg").pointerTypes("@ByPtrPtr _arg"))
               .put(new Info("keyword_ty").valueTypes("_keyword").pointerTypes("@ByPtrPtr _keyword"))
               .put(new Info("alias_ty").valueTypes("_alias").pointerTypes("@ByPtrPtr _alias"))
               .put(new Info("withitem_ty").valueTypes("_withitem").pointerTypes("@ByPtrPtr _withitem"))

               .put(new Info("fileutils.h").linePatterns("#  define _Py_stat_struct stat").skip())
               .put(new Info("_Py_stat_struct").pointerTypes("@Cast(\"struct _Py_stat_struct*\") Pointer"))
               .put(new Info("stat").pointerTypes("@Cast(\"struct stat*\") Pointer"))
               .put(new Info("_Py_wreadlink", "_Py_wrealpath", "_Py_get_blocking", "_Py_set_blocking").skip())
        ;
    }
}
