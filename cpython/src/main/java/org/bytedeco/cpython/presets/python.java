/*
 * Copyright (C) 2018-2021 Samuel Audet
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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            cinclude = {
                "Python.h",
                "patchlevel.h",
                "pyconfig.h",
                "pymacconfig.h",
                "pyport.h",
                "pymacro.h",
//                "pyatomic.h",
                "pymath.h",
                "cpython/pytime.h",
                "pymem.h",
                "cpython/pymem.h",

                "object.h",
                "cpython/object.h",
                "objimpl.h",
                "cpython/objimpl.h",
                "typeslots.h",
                "pyhash.h",
                "cpython/pydebug.h",

                "descrobject.h",
                "bytearrayobject.h",
                "cpython/bytearrayobject.h",
                "bytesobject.h",
                "cpython/bytesobject.h",
                "unicodeobject.h",
                "cpython/unicodeobject.h",
                "longobject.h",
                "longintrepr.h",
                "boolobject.h",
                "floatobject.h",
                "complexobject.h",
                "rangeobject.h",
                "memoryobject.h",
                "tupleobject.h",
                "cpython/tupleobject.h",
                "listobject.h",
                "cpython/listobject.h",
                "dictobject.h",
                "cpython/dictobject.h",
                "structmember.h",
                "cpython/odictobject.h",
                "enumobject.h",
                "setobject.h",
                "methodobject.h",
                "cpython/methodobject.h",
                "moduleobject.h",
                "funcobject.h",
                "classobject.h",
                "fileobject.h",
                "cpython/fileobject.h",
                "frameobject.h",
                "cpython/frameobject.h",
                "pycapsule.h",
                "pyframe.h",
                "traceback.h",
                "cpython/traceback.h",
                "sliceobject.h",
                "cellobject.h",
                "iterobject.h",
                "genobject.h",
                "genericaliasobject.h",
                "warnings.h",
                "weakrefobject.h",
                "structseq.h",
                "namespaceobject.h",
                "cpython/picklebufobject.h",

                "codecs.h",
                "pyerrors.h",
                "cpython/pyerrors.h",
//                "internal/pycore_pyarena.h",
                "pythread.h",
                "pystate.h",
                "context.h",
                "cpython/initconfig.h",
                "cpython/pystate.h",
                "modsupport.h",
                "ceval.h",
                "cpython/ceval.h",
                "sysmodule.h",
                "cpython/sysmodule.h",
                "osmodule.h",
                "intrcheck.h",
                "import.h",
                "cpython/import.h",

                "abstract.h",
                "cpython/abstract.h",
                "bltinmodule.h",
//                "internal/pycore_asdl.h",
//                "internal/pycore_ast.h",
//                "node.h",
                "code.h",
                "cpython/code.h",
                "compile.h",
                "cpython/compile.h",
//                "internal/pycore_symtable.h",
                "pythonrun.h",
                "cpython/pythonrun.h",
                "pylifecycle.h",
                "cpython/pylifecycle.h",
                "eval.h",

                "cpython/pyctype.h",
                "pystrtod.h",
                "pystrcmp.h",
//                "dtoa.h",
                "fileutils.h",
                "cpython/fileutils.h",
                "cpython/pyfpe.h",
                "tracemalloc.h",

                "datetime.h",
            },
            exclude = {
                "cpython/pymem.h",
                "cpython/object.h",
                "cpython/objimpl.h",
                "cpython/bytearrayobject.h",
                "cpython/bytesobject.h",
                "cpython/unicodeobject.h",
                "cpython/tupleobject.h",
                "cpython/listobject.h",
                "cpython/dictobject.h",
                "cpython/methodobject.h",
                "cpython/fileobject.h",
                "cpython/frameobject.h",
                "cpython/traceback.h",
                "cpython/pyerrors.h",
                "cpython/initconfig.h",
                "cpython/pystate.h",
                "cpython/ceval.h",
                "cpython/sysmodule.h",
                "cpython/import.h",
                "cpython/abstract.h",
                "cpython/code.h",
                "cpython/compile.h",
                "cpython/pythonrun.h",
                "cpython/pylifecycle.h",
                "cpython/fileutils.h",
            },
            link = "python3.10@.1.0!",
            preload = {"ffi@.6", "ffi@.5", "libcrypto-1_1", "libssl-1_1"/*, "sqlite3", "tcl86t", "tk86t"*/},
            resource = {"include", "lib", "libs", "bin", "share"}
        ),
        @Platform(value = "linux",        preloadpath = {"/usr/lib/", "/usr/lib32/", "/usr/lib64/"}),
        @Platform(value = "linux-armhf",  preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
        @Platform(value = "linux-arm64",  preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}),
        @Platform(value = "linux-x86",    preloadpath = {"/usr/lib32/", "/usr/lib/"}),
        @Platform(value = "linux-x86_64", preloadpath = {"/usr/lib64/", "/usr/lib/"}),
        @Platform(value = "linux-ppc64",  preloadpath = {"/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
        @Platform(value = "macosx",  link = "python3.10!"),
        @Platform(value = "windows", link = "python310"),
    },
    target = "org.bytedeco.cpython",
    global = "org.bytedeco.cpython.global.python",
    helper = "org.bytedeco.cpython.helper.python"
)
@NoException
public class python implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "cpython"); }

    private static File packageFile = null;

    /** Returns {@code Loader.cacheResource("/org/bytedeco/cpython/" + Loader.getPlatform())} and monkey patches files accordingly. */
    public static synchronized File cachePackage() throws IOException {
        if (packageFile != null) {
            return packageFile;
        }
        File pythonFile = Loader.cacheResource("/org/bytedeco/cpython/" + Loader.getPlatform());
        File configDir = new File(pythonFile, "lib/python3.10/");
        if (configDir.exists()) {
            String pythonPath = pythonFile.getAbsolutePath();
            Pattern pattern = Pattern.compile("'prefix': '(.*)'");
            for (File f : configDir.listFiles()) {
                if (f.getName().startsWith("_sysconfigdata")) {
                    Path path = f.toPath();
                    List<String> lines = Files.readAllLines(path);
                    String prefix = null;
                    for (String l : lines) {
                        Matcher m = pattern.matcher(l);
                        if (m.find()) {
                            prefix = m.group(1);
                            break;
                        }
                    }
                    ArrayList<String> newLines = new ArrayList<String>(lines.size());
                    if (prefix != null) {
                        for (String l : lines) {
                            newLines.add(l.replace(prefix, pythonPath));
                        }
                        Files.write(path, newLines);
                    }
                }
            }

            // Also create symbolic links for native libraries found there.
            File[] files = pythonFile.listFiles();
            if (files != null) {
                ClassProperties p = Loader.loadProperties((Class)null, Loader.loadProperties(), false);
                for (File file : files) {
                    Loader.createLibraryLink(file.getAbsolutePath(), p, null, pythonPath + "/lib");
                }
            }
        }
        packageFile = pythonFile;
        return pythonFile;
    }

    /** Returns {@code {f, new File(f, "site-packages"), new File(f, "python3.10"), new File(f, "python3.10/lib-dynload"), new File(f, "python3.10/site-packages")}} where {@code File f = new File(cachePackage(), "lib")}. */
    public static File[] cachePackages() throws IOException {
        File f = new File(cachePackage(), "lib");
        return new File[] {f, new File(f, "site-packages"), new File(f, "python3.10"), new File(f, "python3.10/lib-dynload"), new File(f, "python3.10/site-packages")};
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("Python-ast.h").linePatterns("#define Module.*",
                                                          "int PyAST_Check.*").skip())

               .put(new Info("COMPILER", "TIMEMODULE_LIB", "NTDDI_VERSION", "Py_NTDDI", "Py_IS_NAN",
                             "copysign", "hypot", "timezone", "daylight", "tzname",
                             "RETSIGTYPE", "_Py_COUNT_ALLOCS_COMMA", "Py_None", "Py_NotImplemented",
                             "PY_LONG_LONG", "PY_UINT32_T", "PY_UINT64_T", "PY_INT32_T", "PY_INT64_T", "PY_SIZE_MAX",
                             "PY_FORMAT_SIZE_T", "Py_MEMCPY", "_Py_HOT_FUNCTION", "_Py_NO_INLINE", "PyMODINIT_FUNC", "Py_VA_COPY",
                             "_Py_SET_53BIT_PRECISION_HEADER", "_Py_SET_53BIT_PRECISION_START", "_Py_SET_53BIT_PRECISION_END",
                             "__inline__", "Py_HUGE_VAL", "Py_FORCE_DOUBLE", "Py_NAN", "Py_TRASHCAN_END",
                             "PyMem_Del", "PyMem_DEL", "PyDescr_COMMON", "PY_UNICODE_TYPE", "_PyObject_EXTRA_INIT",
                             "PyObject_MALLOC", "PyObject_REALLOC", "PyObject_FREE", "PyObject_Del", "PyObject_DEL",
                             "_PyTraceMalloc_Config_INIT", "_PyUnicode_AsStringAndSize", "_PyUnicode_AsString",
                             "PyLong_FromPid", "PyLong_AsPid", "PyLong_AS_LONG",
                             "Py_False", "Py_True", "Py_RETURN_TRUE", "Py_RETURN_FALSE", "Py_RETURN_NAN",
                             "PyObject_HEAD", "PyObject_VAR_HEAD", "Py_RETURN_NONE", "Py_RETURN_NOTIMPLEMENTED",
                             "PyModuleDef_HEAD_INIT", "_Py_atomic_address", "__declspec",
                             "PyException_HEAD", "_Py_NO_RETURN", "Py_Ellipsis",
                             "PyObject_Length", "PySequence_Length", "PySequence_In", "PyMapping_Length",
                             "PY_TIMEOUT_T", "_PyCoreConfig_INIT", "_PyMainInterpreterConfig_INIT", "_PyThreadState_Current",
                             "Py_ALLOW_RECURSION", "Py_END_ALLOW_RECURSION", "NATIVE_TSS_KEY_T",
                             "PYTHREAD_INVALID_THREAD_ID", "Py_BEGIN_ALLOW_THREADS", "Py_END_ALLOW_THREADS", "Py_BLOCK_THREADS", "Py_UNBLOCK_THREADS",
                             "_Py_COMP_DIAG_PUSH", "_Py_COMP_DIAG_IGNORE_DEPR_DECLS", "_Py_COMP_DIAG_POP", "Py_EXPORTED_SYMBOL",
                             "_PyCompilerFlags_INIT", "_PyInterpreterState_Get", "PyOS_strnicmp", "PyOS_stricmp", "Py_FrozenMain",
                             "_PyObject_Vectorcall", "_PyObject_VectorcallMethod", "_PyObject_FastCallDict", "_PyVectorcall_Function",
                             "_PyObject_CallOneArg", "_PyObject_CallMethodNoArgs", "_PyObject_CallMethodOneArg", "_PyTZINFO_HEAD",
                             "_PyDateTime_TIMEHEAD", "_PyDateTime_DATETIMEHEAD", "PyDateTime_IMPORT", "PyDateTime_TimeZone_UTC").cppTypes().annotations())

               .put(new Info("PyAPI_DATA").cppText("#define PyAPI_DATA(RTYPE) RTYPE"))
               .put(new Info("Py_DEPRECATED").cppText("#define Py_DEPRECATED() deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("Py_BUILD_CORE", "defined(Py_BUILD_CORE)",
                             "HAVE_FORK",
                             "HAVE_GCC_ASM_FOR_X87",
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
                             "Py_DEBUG", "Py_TRACE_REFS",
                             "defined(MS_WIN32) && !defined(HAVE_SNPRINTF)",
                             "defined(MS_WINDOWS) && !defined(Py_LIMITED_API)",
                             "defined(Py_REF_DEBUG) && !(defined(Py_LIMITED_API) && Py_LIMITED_API+0 >= 0x030A0000)",
                             "PY_SSIZE_T_CLEAN").cppTypes().define(false))

               .put(new Info("!defined(__INTEL_COMPILER)", "WITH_THREAD", "PY_NO_SHORT_FLOAT_REPR").cppTypes().define(true))

               .put(new Info("COMPILER", "PY_LLONG_MIN", "PY_LLONG_MAX", "PY_ULLONG_MAX", "PY_TIMEOUT_MAX",
                             "SIZEOF_PY_HASH_T", "SIZEOF_PY_UHASH_T", "PY_SSIZE_T_MAX", "PY_SSIZE_T_MIN",
                             "LONG_BIT", "PyLong_BASE", "PyLong_MASK", "Py_UNICODE_SIZE").cppTypes("long long").translate(false))

               .put(new Info("PyHash_FuncDef").purify())

               .put(new Info("wchar_t").cast().valueTypes("char", "int").pointerTypes("Pointer"))
               .put(new Info("timeval", "timespec", "tm").cast().pointerTypes("Pointer"))
               .put(new Info("atomic_uintptr_t", "atomic_int").cast().pointerTypes("Pointer"))
               .put(new Info("Py_ssize_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("_typeobject").cast().pointerTypes("PyTypeObject"))
               .put(new Info("_dictkeysobject").cast().pointerTypes("PyDictKeysObject"))
               .put(new Info("_gc_head").cast().pointerTypes("PyGC_Head"))
               .put(new Info("_traceback").cast().pointerTypes("PyTracebackObject"))
               .put(new Info("_err_stackitem").cast().pointerTypes("_PyErr_StackItem"))
               .put(new Info("_co_extra_state").cast().pointerTypes("__PyCodeExtraState"))
               .put(new Info("_cframe").cast().pointerTypes("CFrame"))
               .put(new Info("_node").cast().pointerTypes("node"))
               .put(new Info("_is").cast().pointerTypes("PyInterpreterState"))
               .put(new Info("_ts").cast().pointerTypes("PyThreadState"))

               .put(new Info("PyThreadFrameGetter", "jmp_buf").cast().pointerTypes("Pointer"))

               .put(new Info("_Py_memory_order", "PyThreadState::_preserve_36_ABI_1", "PyThreadState::_preserve_36_ABI_2",
                             "_PyGC_generation0", "_PyBytes_InsertThousandsGroupingLocale",
                             "_PyBytes_InsertThousandsGrouping", "_PyUnicode_DecodeUnicodeInternal",
                             "_PyFloat_Repr", "_PyFloat_Digits", "_PyFloat_DigitsInit",
                             "PySortWrapper_Type", "PyCmpWrapper_Type", "_PyGen_yf", "_PyAIterWrapper_New",
                             "_PyTime_FromTimeval", "_PyAIterWrapper_Type", "_PyErr_WarnUnawaitedCoroutine", "_PyErr_GetTopmostException",
                             "PyInit__imp", "_PyCoro_GetAwaitableIter", "_PyAsyncGenValueWrapperNew", "PyAsyncGen_ClearFreeLists",
                             "_PyFrame_New_NoTrack", "PyStructSequence_UnnamedField", "PySignal_SetWakeupFd",
                             "_PyArg_Fini", "PyInit_imp", "PyNullImporter_Type", "PyBuffer_SizeFromFormat",
                             "__PyCodeExtraState::co_extra_freefuncs", "_PyDict_NewKeysForClass", "_PyDictView_New",
                             "_PyDict_KeysSize", "_PyDict_SizeOf", "_PyDict_Pop_KnownHash", "_PyDict_FromKeys",
                             "_PyObjectDict_SetItem", "_PyDict_LoadGlobal", "__PyCodeExtraState_Get",
                             "_Py_asdl_seq_new", "_Py_asdl_int_seq_new", "_Py_isabs", "_PyTime_MIN", "_PyTime_MAX",
                             "_Py_InitializeFromWideArgs", "_Py_InitializeFromArgs", "_PyNode_FinalizeEndPos",
                             "_PyCodec_Forget", "_PyCode_InitAddressRange", "_PyDict_GetItemHint", "PyLineTable_InitAddressRange",
                             "PyLineTable_NextAddressRange", "PyLineTable_PreviousAddressRange").skip())

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
               .put(new Info("vectorcallfunc").valueTypes("vectorcallfunc").pointerTypes("@ByPtrPtr vectorcallfunc").javaText(
                       "public class vectorcallfunc extends FunctionPointer {\n"
                     + "    static { Loader.load(); }\n"
                     + "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n"
                     + "    public    vectorcallfunc(Pointer p) { super(p); }\n"
                     + "    protected vectorcallfunc() { allocate(); }\n"
                     + "    private native void allocate();\n"
                     + "    public native PyObject call(PyObject callable, @Cast(\"PyObject *const *\") PointerPointer args,\n"
                     + "                                    @Cast(\"size_t\") long nargsf, PyObject kwnames);\n"
                     + "}\n"))
               .put(new Info("_PyArg_UnpackKeywords").javaText(
                       "@NoException public static native @Cast(\"PyObject *const *\") PointerPointer _PyArg_UnpackKeywords(\n"
                     + "        @Cast(\"PyObject *const *\") PointerPointer args, @Cast(\"Py_ssize_t\") long nargs,\n"
                     + "        PyObject kwargs, PyObject kwnames,\n"
                     + "        _PyArg_Parser parser,\n"
                     + "        int minpos, int maxpos, int minkw,\n"
                     + "        @Cast(\"PyObject**\") PointerPointer buf);\n"))

               .put(new Info("fileutils.h").linePatterns("#  define _Py_stat_struct stat").skip())
               .put(new Info("_Py_stat_struct").pointerTypes("@Cast(\"struct _Py_stat_struct*\") Pointer"))
               .put(new Info("stat").pointerTypes("@Cast(\"struct stat*\") Pointer"))
               .put(new Info("_Py_wreadlink", "_Py_wrealpath", "_Py_get_blocking", "_Py_set_blocking").skip())
        ;

        String PyArg_Parse = "public static native int PyArg_Parse(PyObject arg0, String arg1",
               PyArg_ParseTuple = "public static native int PyArg_ParseTuple(PyObject arg0, String arg1",
               PyArg_ParseTupleAndKeywords = "public static native int PyArg_ParseTupleAndKeywords(PyObject arg0, PyObject arg1,\n"
                                           + "                                                  String arg2, @Cast(\"char**\") PointerPointer arg3";

        String PyArg_ParseText = "", PyArg_ParseTupleText = "", PyArg_ParseTupleAndKeywordsText = "";
        for (int i = 0; i < 10; i++) {
            PyArg_ParseText += PyArg_Parse;
            PyArg_ParseTupleText += PyArg_ParseTuple;
            PyArg_ParseTupleAndKeywordsText += PyArg_ParseTupleAndKeywords;
            for (int j = 0; j <= i; j++) {
                PyArg_ParseText += ", Pointer vararg" + j;
                PyArg_ParseTupleText += ", Pointer vararg" + j;
                PyArg_ParseTupleAndKeywordsText += ", Pointer vararg" + j;
            }
            PyArg_ParseText += ");\n";
            PyArg_ParseTupleText += ");\n";
            PyArg_ParseTupleAndKeywordsText += ");\n";
        }
        infoMap.put(new Info("PyArg_Parse").javaText(PyArg_ParseText))
               .put(new Info("PyArg_ParseTuple").javaText(PyArg_ParseTupleText))
               .put(new Info("PyArg_ParseTupleAndKeywords").javaText(PyArg_ParseTupleAndKeywordsText));
    }
}
