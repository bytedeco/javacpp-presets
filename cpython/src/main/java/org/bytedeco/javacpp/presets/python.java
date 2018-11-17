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

package org.bytedeco.javacpp.presets;

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
                "pyport.h",
                "pymem.h",
                "descrobject.h",
                "dictobject.h",
                "structmember.h",
                "object.h",
                "methodobject.h",
                "moduleobject.h",
                "pyarena.h",
                "pystate.h",
                "ceval.h",
                "asdl.h",
                "Python-ast.h",
                "node.h",
                "code.h",
                "compile.h",
                "symtable.h",
                "pythonrun.h",
                "pylifecycle.h",
                "fileutils.h",
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
    target = "org.bytedeco.javacpp.python"
)
@NoException
public class python implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("Python-ast.h").linePatterns("#define Module.*",
                                                          "int PyAST_Check.*").skip())

               .put(new Info("_Py_COUNT_ALLOCS_COMMA", "Py_None", "Py_NotImplemented",
                             "PY_LONG_LONG", "PY_UINT32_T", "PY_UINT64_T", "PY_INT32_T", "PY_INT64_T",
                             "PY_FORMAT_SIZE_T", "Py_MEMCPY", "PyMODINIT_FUNC", "Py_VA_COPY",
                             "PyMem_Del", "PyMem_DEL", "PyDescr_COMMON",
                             "PyObject_HEAD", "PyObject_VAR_HEAD", "Py_RETURN_NONE", "Py_RETURN_NOTIMPLEMENTED",
                             "PyModuleDef_HEAD_INIT", "_Py_atomic_address", "__declspec",
                             "Py_ALLOW_RECURSION", "Py_END_ALLOW_RECURSION",
                             "Py_BEGIN_ALLOW_THREADS", "Py_END_ALLOW_THREADS",
                             "Py_BLOCK_THREADS", "Py_UNBLOCK_THREADS").cppTypes().annotations())

               .put(new Info("HAVE_GCC_ASM_FOR_X87",
                             "defined(_MSC_VER) && !defined(_WIN64)",
                             "HAVE_GCC_ASM_FOR_MC68881",
                             "HAVE__GETPTY",
                             "defined(HAVE_OPENPTY) || defined(HAVE_FORKPTY)",
                             "defined _MSC_VER && _MSC_VER >= 1900",
                             "COUNT_ALLOCS",
                             "HAVE_DLOPEN",
                             "SOLARIS",
                             "MS_WINDOWS").define(false))

               .put(new Info("PY_LLONG_MIN", "PY_LLONG_MAX", "PY_ULLONG_MAX",
                             "SIZEOF_PY_HASH_T", "SIZEOF_PY_UHASH_T",
                             "PY_SIZE_MAX", "PY_SSIZE_T_MAX", "PY_SSIZE_T_MIN",
                             "LONG_BIT").translate(false))

               .put(new Info("wchar_t").cast().valueTypes("char", "int").pointerTypes("Pointer"))
               .put(new Info("Py_ssize_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("_typeobject").cast().pointerTypes("PyTypeObject"))
               .put(new Info("_dictkeysobject").cast().pointerTypes("PyDictKeysObject"))
               .put(new Info("_co_extra_state").cast().pointerTypes("__PyCodeExtraState"))
               .put(new Info("_node").cast().pointerTypes("node"))

               .put(new Info("PyThreadFrameGetter").cast().pointerTypes("Pointer"))

               .put(new Info("PyThreadState::_preserve_36_ABI_1", "PyThreadState::_preserve_36_ABI_2",
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
