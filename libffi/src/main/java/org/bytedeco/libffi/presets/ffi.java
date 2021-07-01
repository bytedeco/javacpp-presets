/*
 * Copyright (C) 2021 Samuel Audet
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

package org.bytedeco.libffi.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.MemberGetter;
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
            include = {"ffitarget.h", "ffi.h"},
            exclude = "ffitarget.h",
            link = "ffi@.8",
            resource = {"include", "lib"}
        ),
        @Platform(
            value = "windows",
            link = "libffi-8"
        ),
    },
    target = "org.bytedeco.libffi",
    global = "org.bytedeco.libffi.global.ffi"
)
@NoException
public class ffi implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "libffi"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("FFI_EXTRA_CIF_FIELDS", "FFI_NATIVE_RAW_API", "_CET_NOTRACK",
                             "FFI_LONG_LONG_MAX", "FFI_API", "FFI_EXTERN").cppTypes().annotations())
               .put(new Info("defined (POWERPC64)", "LONG_LONG_MAX", "1").define(true))
               .put(new Info("!FFI_NATIVE_RAW_API", "FFI_TARGET_HAS_COMPLEX_TYPE",
                             "defined(_WIN32)", "defined(X86_WIN32)", "defined(X86_WIN64)", "X86_WIN64",
                             "defined __x86_64__ && defined __ILP32__",
                             "defined(X86_64) || (defined (__x86_64__) && defined (X86_DARWIN))",
                             "defined(__ARM_PCS_VFP) || defined(_M_ARM)",
                             "defined(__ARM_PCS_VFP) || defined(_WIN32)",
                             "defined(POWERPC_DARWIN) || defined(POWERPC_AIX)",
                             "defined (POWERPC_AIX)", "defined (POWERPC_DARWIN)").define(false))
               .put(new Info("FFI_TYPE_SMALL_STRUCT_1B", "FFI_TYPE_SMALL_STRUCT_2B",
                             "FFI_TYPE_SMALL_STRUCT_4B", "FFI_TYPE_MS_STRUCT", "FFI_TRAMPOLINE_SIZE",
                             "FFI_PPC_TYPE_LAST", "FFI_SIZEOF_ARG", "FFI_SIZEOF_JAVA_RAW").translate(false))
               .put(new Info("ffi_type_uchar", "ffi_type_schar", "ffi_type_ushort", "ffi_type_sshort",
                             "ffi_type_uint", "ffi_type_sint", "ffi_type_ulong", "ffi_type_slong",
                             "ffi_type_longdouble").cppTypes("ffi_type").translate(false))
               .put(new Info("_ffi_type").pointerTypes("ffi_type"))
               .put(new Info("void (*)(void)").cast().pointerTypes("Pointer"))
               .put(new Info("__attribute__((deprecated))").annotations("@Deprecated"))
        ;
    }

    public static native @MemberGetter int FFI_FIRST_ABI();
    public static native @MemberGetter @Platform(not = "windows", pattern = ".*-x86_64") int FFI_UNIX64();
    public static native @MemberGetter @Platform(not = "windows", pattern = ".*-x86_64") int FFI_EFI64();
    public static native @MemberGetter @Platform(pattern = ".*-x86_64") int FFI_WIN64();
    public static native @MemberGetter @Platform(pattern = ".*-x86_64") int FFI_GNUW64();
    public static native @MemberGetter @Platform(pattern = ".*-x86") int FFI_THISCALL();
    public static native @MemberGetter @Platform(pattern = ".*-x86") int FFI_FASTCALL();
    public static native @MemberGetter @Platform(pattern = ".*-x86") int FFI_STDCALL();
    public static native @MemberGetter @Platform(pattern = ".*-x86") int FFI_PASCAL();
    public static native @MemberGetter @Platform(pattern = ".*-x86") int FFI_REGISTER();
    public static native @MemberGetter @Platform(pattern = ".*-x86") int FFI_MS_CDECL();
    public static native @MemberGetter @Platform(pattern = {".*-x86", ".*-arm.*"}) int FFI_SYSV();
    public static native @MemberGetter @Platform(pattern = "(?!.*-arm64).*-arm.*") int FFI_VFP();
    public static native @MemberGetter @Platform(pattern = ".*-ppc64.*") int FFI_LINUX();
    public static native @MemberGetter @Platform(pattern = ".*-ppc64.*") int FFI_LINUX_STRUCT_ALIGN();
    public static native @MemberGetter @Platform(pattern = ".*-ppc64.*") int FFI_LINUX_LONG_DOUBLE_128();
    public static native @MemberGetter @Platform(pattern = ".*-ppc64.*") int FFI_LINUX_LONG_DOUBLE_IEEE128();
    public static native @MemberGetter int FFI_LAST_ABI();
    public static native @MemberGetter int FFI_DEFAULT_ABI();
}
