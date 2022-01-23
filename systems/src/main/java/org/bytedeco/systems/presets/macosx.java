/*
 * Copyright (C) 2017-2022 Samuel Audet, Eduardo Gonzalez
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

package org.bytedeco.systems.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.BuildEnabled;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.javacpp.tools.Logger;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = javacpp.class, value = {@Platform(value = "macosx-x86", define = "__STDC_WANT_LIB_EXT1__ 1",
    include = {"cpuid.h", "dlfcn.h", "nl_types.h", "_xlocale.h", "xlocale.h", "_locale.h", "langinfo.h", "locale.h",
               "sys/uio.h", "sys/_types/_iovec_t.h", "sys/socket.h", "sys/errno.h", "string.h", "stdlib.h", /*"sys/types.h",*/
               "sys/_types/_timespec.h", "sys/_types/_timeval.h", "sys/time.h", "time.h", "utime.h",
               "sys/_types/_s_ifmt.h", "sys/_types/_filesec_t.h", "sys/stat.h", "fcntl.h", "sys/file.h", "grp.h", "pwd.h",
               "sys/_types/_sigaltstack.h", "sys/signal.h", "signal.h", /*"sys/_types/_ucontext.h", "sys/ucontext.h", "ucontext.h",*/
               "sched.h", "mach/machine.h", "spawn.h", "sys/_types/_seek_set.h", "sys/unistd.h", "unistd.h",
               "sys/poll.h", "sys/reboot.h", "sys/resource.h", "sys/sysctl.h", "sys/wait.h",
               "sys/_types/_uid_t.h", "sys/_types/_gid_t.h", "sys/_types/_mode_t.h", "sys/_types/_key_t.h", "sys/ipc.h",
               "sys/_types/_pid_t.h", "sys/_types/_time_t.h", "sys/_types/_size_t.h", "sys/shm.h"},
    includepath = {"/usr/include/", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/",
                   "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/"})},
    target = "org.bytedeco.systems.macosx", global = "org.bytedeco.systems.global.macosx")
@NoException
public class macosx implements BuildEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "systems"); }

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;
    private boolean is64bits;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
        this.is64bits = properties.getProperty("platform").contains("64");
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("stat.h").linePatterns("#define[ \t]st_.*").skip())
               .put(new Info("signal.h").linePatterns("#define[ \t]sa_.*").skip())
               .put(new Info("wait.h").linePatterns("#define[ \t]w_.*").skip())

               .put(new Info("__BEGIN_DECLS").cppText("#define __BEGIN_DECLS"))
               .put(new Info("__END_DECLS").cppText("#define __END_DECLS"))

               .put(new Info("__LP64__", "__x86_64__").define(is64bits))

               .put(new Info("!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)",
                             "__APPLE__", "__DARWIN_UNIX03").define(true))

               .put(new Info("__BLOCKS__", "!__DARWIN_UNIX03",
                             "__DARWIN_C_LEVEL < __DARWIN_C_FULL").define(false))

               .put(new Info("__deprecated").annotations("@Deprecated").cppTypes())

               .put(new Info("__DARWIN_ALIAS", "__DARWIN_STRUCT_STAT64_TIMES", "__DARWIN_STRUCT_STAT64", "_NLS_PRIVATE",
                             "_STRUCT_TIMESPEC", "_STRUCT_TIMEVAL", "_STRUCT_SIGALTSTACK", "_STRUCT_UCONTEXT",
                             "__extension__", "__header_always_inline", "__inline", "__mode__",
                             "__nonnull", "_Nullable", "__restrict", "__CLOCK_AVAILABILITY", "__OS_AVAILABILITY_MSG",
                             "__DYLDDL_DRIVERKIT_UNAVAILABLE", "__IOS_PROHIBITED", "__TVOS_PROHIBITED", "__WATCHOS_PROHIBITED",
                             "ru_first", "ru_last", "sv_onstack").annotations().cppTypes())

               .put(new Info("_POSIX2_VERSION", "_POSIX2_C_VERSION", "_POSIX2_C_BIND",
                             "_POSIX2_C_DEV", "_POSIX2_SW_DEV", "_POSIX2_LOCALEDEF").cppTypes("long"))

               .put(new Info("sa_family_t", "__uint8_t")
                       .cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))

               .put(new Info("nlink_t", "mode_t", "__int16_t", "__uint16_t", "u_int16_t")
                       .cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))

               .put(new Info("nl_item", "socklen_t", "useconds_t", "errno_t", "blksize_t",
                             "dev_t", "id_t", "gid_t", "uid_t", "pid_t", "cpu_type_t", "cpu_subtype_t", "sigset_t",
                             "__darwin_suseconds_t", "__darwin_sigset_t", "exception_behavior_t", "exception_mask_t",
                             "mach_port_t", "thread_state_flavor_t", "integer_t", "__int32_t", "__uint32_t", "u_int32_t")
                       .cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))

               .put(new Info("clock_t", "ino_t", "intptr_t", "rlim_t", "rsize_t", "ssize_t",
                             "__darwin_size_t", "__darwin_time_t")
                       .cast().valueTypes("long").pointerTypes("SizeTPointer"))

               .put(new Info("greg_t", "blkcnt_t", "blkcnt64_t", "off_t", "off64_t", "rlim64_t",
                             "__darwin_ino64_t", "__int64_t", "__uint64_t", "u_int64_t")
                       .cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))

               .put(new Info("pthread_t", "pthread_attr_t").cast().pointerTypes("Pointer"))
               .put(new Info("posix_spawnattr_t").cast().valueTypes("posix_spawnattr_t")
                       .pointerTypes("@ByPtrPtr posix_spawnattr_t", "PointerPointer"))
               .put(new Info("posix_spawn_file_actions_t").cast().valueTypes("posix_spawn_file_actions_t")
                       .pointerTypes("@ByPtrPtr posix_spawn_file_actions_t", "PointerPointer"))
               .put(new Info("uuid_t").annotations("@Cast(\"unsigned char*\")").valueTypes("BytePointer", "ByteBuffer", "byte[]"))

               .put(new Info("nl_catd").valueTypes("__nl_cat_d"))
               .put(new Info("locale_t").valueTypes("_xlocale"))
               .put(new Info("filesec_t").valueTypes("_filesec"))
               .put(new Info("struct stat").pointerTypes("stat"))
               .put(new Info("struct stat64").pointerTypes("stat64"))
               .put(new Info("struct timezone").pointerTypes("timezone"))
               .put(new Info("struct sigaction").pointerTypes("sigaction"))
               .put(new Info("struct sigvec").pointerTypes("sigvec"))
               .put(new Info("__siginfo").pointerTypes("siginfo_t"))
               .put(new Info("__darwin_sigaltstack").pointerTypes("stack_t"))
               .put(new Info("__darwin_ucontext").valueTypes("ucontext_t"))
               .put(new Info("union wait").pointerTypes("wait"))

               .put(new Info("LC_ALL_MASK").cppTypes("int").translate(false))
               .put(new Info("LC_GLOBAL_LOCALE", "LC_C_LOCALE").cppTypes("locale_t").translate(false))
               .put(new Info("PF_VLAN", "PF_BOND").cppTypes("int").translate(false))
               .put(new Info("RTLD_NEXT", "RTLD_DEFAULT", "RTLD_SELF", "RTLD_MAIN_ONLY").cppTypes("void*").translate(false))
               .put(new Info("SAE_ASSOCID_ANY", "SAE_ASSOCID_ALL", "SAE_CONNID_ANY", "SAE_CONNID_ALL").cppTypes("long").translate(false))
               .put(new Info("RLIM_INFINITY", "RLIM_SAVED_MAX", "RLIM_SAVED_CUR").cppTypes("long").translate(false))
               .put(new Info("CLOCKS_PER_SEC").cppTypes("long").translate(false))
               .put(new Info("NSIG").cppTypes("long").translate(false))
               .put(new Info("SIG_DFL", "SIG_IGN", "SIG_HOLD", "SIG_ERR", "BADSIG").annotations("@Cast(\"void*\")").cppTypes("void*").translate(false))
               .put(new Info("_SS_PAD2SIZE", "WCOREFLAG").cppTypes("int").translate(false))

               .put(new Info("__cpuid").cppTypes("void", "int", "int&", "int&", "int&", "int&"))
               .put(new Info("__cpuid_count").cppTypes("void", "int", "int", "int&", "int&", "int&", "int&"))

               .put(new Info("memchr").javaText("public static native Pointer memchr(Pointer __s, int __c, @Cast(\"size_t\") long __n);"))

               .put(new Info("getwd", "mkstemp_dprotected_np", "posix_spawnattr_setsuidcredport_np",
                             "__ipc_perm_new", "__shmid_ds_new", "shmid_ds").skip());
    }
}
