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

import java.io.File;
import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
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
@Properties(inherit = javacpp.class, value = {@Platform(value = "linux",
    exclude = {"xlocale.h", "bits/types/__locale_t.h", "bits/types/locale_t.h", "bits/locale.h", "bits/types/struct_iovec.h",
               "bits/types/struct_tm.h", "bits/types/struct_timeval.h", "bits/types/struct_timespec.h", "bits/types/struct_itimerspec.h", "bits/types/timer_t.h",
               "bits/uio.h", "bits/socket_type.h", "bits/socket.h", "bits/errno.h", "bits/types/siginfo_t.h", "bits/types/__sigset_t.h", "bits/types/sigset_t.h",
               "bits/types/__sigval_t.h", "bits/types/sigval_t.h", "bits/types/stack_t.h", "bits/siginfo.h", "bits/sigset.h", "bits/signum.h",
               "bits/sigaction.h", "bits/sigcontext.h", "bits/sigstack.h", "bits/cpu-set.h", "bits/types/struct_sched_param.h", "bits/sched.h",
               "bits/confname.h", "bits/resource.h", "bits/struct_stat.h", "bits/ipc-perm.h", "bits/ipc.h", "bits/shm.h", "bits/types/struct_shmid_ds.h"},
    include = {"cpuid.h", "dlfcn.h", "nl_types.h", "xlocale.h", "bits/types/__locale_t.h", "bits/types/locale_t.h", "bits/locale.h", "langinfo.h", "locale.h",
               "bits/types/struct_tm.h", "bits/types/struct_timeval.h", "bits/types/struct_timespec.h", "bits/types/struct_itimerspec.h", "bits/types/timer_t.h",
               "bits/types/struct_iovec.h", "bits/uio.h", "sys/uio.h", "bits/sockaddr.h", "bits/socket_type.h", "bits/socket.h", "sys/socket.h",
               "asm-generic/errno-base.h", "asm-generic/errno.h", "bits/errno.h", "errno.h", "string.h", "stdlib.h", /*"sys/types.h", "bits/timex.h",*/
               "bits/time.h", "sys/time.h", "time.h", "utime.h", "bits/stat.h", "sys/stat.h", "fcntl.h", "sys/file.h", "grp.h", "pwd.h",
               "bits/types/siginfo_t.h", "bits/types/__sigset_t.h", "bits/types/sigset_t.h", "bits/types/__sigval_t.h", "bits/types/sigval_t.h", "bits/types/stack_t.h",
               "bits/siginfo.h", "bits/sigset.h", "bits/signum.h", "bits/sigaction.h", "bits/sigcontext.h", "bits/sigstack.h", "signal.h",
               "bits/cpu-set.h", "bits/types/struct_sched_param.h", "sys/ucontext.h", "ucontext.h", "bits/sched.h", "sched.h", "spawn.h", "bits/posix_opt.h",
               "bits/confname.h", "unistd.h", "sys/poll.h", "sys/reboot.h", "bits/resource.h", "sys/resource.h", /*"sys/sysctl.h",*/ "bits/waitflags.h", "sys/wait.h",
               "bits/struct_stat.h", "bits/ipc-perm.h", "bits/ipc.h", "sys/ipc.h", "bits/shm.h", "bits/types/struct_shmid_ds.h", "sys/shm.h"},
    link = "dl")}, target = "org.bytedeco.systems.linux", global = "org.bytedeco.systems.global.linux")
@NoException
public class linux implements BuildEnabled, LoadEnabled, InfoMapper {
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

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        if (platform.startsWith("linux")) {
            List<String> includePaths = properties.get("platform.includepath");
            List<String> includes = properties.get("platform.include");
            for (String path : includePaths) {
                if (new File(path, "linux/sysinfo.h").exists()) {
                    includes.add("linux/sysinfo.h");
                }
            }
            includes.add("linux/kernel.h");
            includes.add("sys/sysinfo.h");
        }
    }

    public void map(InfoMap infoMap) {

        infoMap.put(new Info("stat.h").linePatterns("# *define st_.*").skip())
               .put(new Info("siginfo.h").linePatterns("# define si_.*").skip())
               .put(new Info("sigaction.h").linePatterns("    union", "      \\{", "      \\}", ".*sa_sigaction").skip())
               .put(new Info("signal.h").linePatterns("#ifndef\t_SIGNAL_H", "#endif").skip())
               .put(new Info("ptrace.h").linePatterns("#define .*regs\\[.*").skip())
               .put(new Info("sysinfo.h").linePatterns(".*char _f.*").skip())
               .put(new Info("siginfo_t.h").linePatterns(".*X/Open.*", "#endif").skip())

               .put(new Info("__BEGIN_DECLS").cppText("#define __BEGIN_DECLS"))
               .put(new Info("__END_DECLS").cppText("#define __END_DECLS"))
               .put(new Info("__NTH").cppText("#define __NTH(a) a"))
               .put(new Info("__BEGIN_NAMESPACE_C99").cppText("#define __BEGIN_NAMESPACE_C99"))
               .put(new Info("__END_NAMESPACE_C99").cppText("#define __END_NAMESPACE_C99"))
               .put(new Info("__USING_NAMESPACE_C99").cppText("#define __USING_NAMESPACE_C99(name)"))
               .put(new Info("__BEGIN_NAMESPACE_STD").cppText("#define __BEGIN_NAMESPACE_STD"))
               .put(new Info("__END_NAMESPACE_STD").cppText("#define __END_NAMESPACE_STD"))
               .put(new Info("__USING_NAMESPACE_STD").cppText("#define __USING_NAMESPACE_STD(name)"))

               .put(new Info("__WORDSIZE == 32", "__i386__").define(!is64bits))
               .put(new Info("__WORDSIZE == 64", "__aarch64__", "__powerpc64__", "__x86_64__").define(is64bits))

               .put(new Info("defined __cplusplus || !__GNUC_PREREQ (2, 7) || !defined __USE_GNU",
                             "__WORDSIZE == 64 || !defined __USE_FILE_OFFSET64",
                             "defined __x86_64__ || !defined __USE_FILE_OFFSET64",
                             "defined __USE_MISC || defined __USE_XOPEN2K8",
                             "!defined __GNUC__ || __GNUC__ < 2 || defined __cplusplus",
                             "__CORRECT_ISO_CPP_STRING_H_PROTO", "__GNUC__", "__USE_BSD",
                             "__USE_POSIX199309", "defined __USE_POSIX199309 || defined __USE_XOPEN_EXTENDED",
                             "__USE_ISOC99", "__USE_MISC", "__USE_XOPEN2K", "__USE_XOPEN2K8").define(true))

               .put(new Info("defined __USE_ISOC11 || defined __USE_ISOCXX11",
                             "defined __x86_64__ && __WORDSIZE == 32",
                             "defined __USE_XOPEN2K && !defined __USE_GNU",
                             "defined __GNUC__ && __GNUC__ >= 2 && defined __USE_EXTERN_INLINES",
                             "defined __USE_XOPEN_EXTENDED && !defined __USE_XOPEN2K8",
                             "__GNUC_PREREQ (3, 0)",  "__SI_CLOCK_T", "__TIMESIZE == 32",
                             "__USE_EXTERN_INLINES", "__USE_FILE_OFFSET64", "_LINUX_KERNEL_H",
                             "__HAVE_FLOAT16", "__HAVE_FLOAT16 && __GLIBC_USE (IEC_60559_TYPES_EXT)",
                             "__HAVE_FLOAT32", "__HAVE_FLOAT32 && __GLIBC_USE (IEC_60559_TYPES_EXT)",
                             "__HAVE_FLOAT64", "__HAVE_FLOAT64 && __GLIBC_USE (IEC_60559_TYPES_EXT)",
                             "__HAVE_FLOAT128", "__HAVE_FLOAT128 && __GLIBC_USE (IEC_60559_TYPES_EXT)",
                             "__HAVE_FLOAT32X", "__HAVE_FLOAT32X && __GLIBC_USE (IEC_60559_TYPES_EXT)",
                             "__HAVE_FLOAT64X", "__HAVE_FLOAT64X && __GLIBC_USE (IEC_60559_TYPES_EXT)",
                             "__HAVE_FLOAT128X", "__HAVE_FLOAT128X && __GLIBC_USE (IEC_60559_TYPES_EXT)").define(false).cppTypes())

               .put(new Info("__SOCKADDR_ARG", "__CONST_SOCKADDR_ARG", "__SOCKADDR_ALLTYPES", "CLK_TCK", "__SI_BAND_TYPE",
                             "error_t", "__extern_always_inline", "__extern_inline", "_EXTERN_INLINE", "__inline",
                             "__ext", "__extension__", "__mode__", "__nonnull", "__ss_aligntype", "__sysconf",
                             "__REDIRECT_NTH", "__REDIRECT", "__THROW", "__restrict", "__wur", "UIO_MAXIOV",
                             "__WAIT_STATUS", "__WAIT_STATUS_DEFN", "sched_priority", "__sched_priority", "sigcontext_struct",
                             "sigev_notify_function", "sigev_notify_attributes", "sv_onstack", "__FUNCTION__",
                             "st_atime", "st_mtime", "st_ctime").annotations().cppTypes())

               .put(new Info("_POSIX2_VERSION", "_POSIX2_C_BIND",
                             "_POSIX2_C_DEV", "_POSIX2_SW_DEV", "_POSIX2_LOCALEDEF").cppTypes("long"))
               .put(new Info("_POSIX2_C_VERSION").skip())

               .put(new Info("__u16", "__uint16_t")
                       .cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))

               .put(new Info("socklen_t", "clockid_t", "useconds_t", "id_t", "gid_t", "uid_t", "pid_t", "mode_t",
                             "__socklen_t", "__clockid_t", "__useconds_t", "__id_t", "__gid_t", "__uid_t", "__pid_t", "__mode_t",
                             "error_t", "__u32", "__uint32_t", "key_t", "__key_t")
                       .cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))

               .put(new Info("clock_t", "dev_t", "off_t", "intptr_t", "rlim_t", "ssize_t",
                             "__clock_t", "__dev_t", "__off_t", "__intptr_t", "__rlim_t", "__ssize_t",
                             "__blkcnt_t", "__blksize_t", "__ino_t", "__nlink_t", "__time_t", "__timer_t", "__suseconds_t",
                             "__syscall_slong_t", "__syscall_ulong_t", "__CPU_MASK_TYPE", "__kernel_long_t", "__kernel_ulong_t")
                       .cast().valueTypes("long").pointerTypes("SizeTPointer"))

               .put(new Info("off64_t", "rlim64_t", "__off64_t", "__rlim64_t", "__blkcnt64_t", "greg_t", "__ino64_t", "__u64", "__uint64_t")
                       .cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))

               .put(new Info("__locale_data", "module", "sigcontext", "__spawn_action", "timex", "pt_regs",
                             "_fpreg", "_fpstate", "_fpxreg", "_libc_fpstate", "_libc_fpxreg", "_libc_xmmreg", "_xmmreg").cast().pointerTypes("Pointer"))
               .put(new Info("__timezone_ptr_t").cast().pointerTypes("timezone"))
               .put(new Info("gregset_t", "fpregset_t").cppTypes("void* const"))

               .put(new Info("__locale_struct").pointerTypes("locale_t"))
               .put(new Info("__locale_t").valueTypes("locale_t"))
               .put(new Info("struct stat").pointerTypes("stat"))
               .put(new Info("struct stat64").pointerTypes("stat64"))
               .put(new Info("struct sysinfo").pointerTypes("sysinfo"))
               .put(new Info("struct timezone").pointerTypes("timezone"))
               .put(new Info("struct sigaction").pointerTypes("sigaction"))
               .put(new Info("struct sigvec").pointerTypes("sigvec").skip())
               .put(new Info("__sigset_t").pointerTypes("sigset_t"))
               .put(new Info("sigval_t", "__sigval_t").pointerTypes("sigval"))
               .put(new Info("struct sigevent").pointerTypes("sigevent_t"))
               .put(new Info("struct sigstack").pointerTypes("sigstack"))
               .put(new Info("sigaltstack").pointerTypes("stack_t"))
               .put(new Info("ucontext").valueTypes("ucontext_t"))
               .put(new Info("__sighandler_t").valueTypes("__sighandler_t"))
               .put(new Info("struct shminfo").pointerTypes("shminfo"))
               .put(new Info("struct shmid_ds").pointerTypes("shmid_ds"))

               .put(new Info("siginfo_t").javaText("\n"
                     + "public static class siginfo_t extends Pointer {\n"
                     + "     static { Loader.load(); }\n"
                     + "      /** Default native constructor. */\n"
                     + "      public siginfo_t() { super((Pointer)null); allocate(); }\n"
                     + "      /** Native array allocator. Access with {@link Pointer#position(long)}. */\n"
                     + "      public siginfo_t(long size) { super((Pointer)null); allocateArray(size); }\n"
                     + "      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n"
                     + "      public siginfo_t(Pointer p) { super(p); }\n"
                     + "      private native void allocate();\n"
                     + "      private native void allocateArray(long size);\n"
                     + "      @Override public siginfo_t position(long position) {\n"
                     + "          return (siginfo_t)super.position(position);\n"
                     + "      }\n"
                     + "\n"
                     + "      public native int si_signo(); public native siginfo_t si_signo(int si_signo);		/* Signal number.  */\n"
                     + "      public native int si_errno(); public native siginfo_t si_errno(int si_errno);		/* If non-zero, an errno value associated with\n"
                     + "                     this signal, as defined in <errno.h>.  */\n"
                     + "      public native int si_code(); public native siginfo_t si_code(int si_code);		/* Signal code.  */\n"
                     + "\n"
                     + "      public native @Cast(\"__pid_t\") int si_pid(); public native siginfo_t si_pid(int si_pid);	/* Sending process ID.  */\n"
                     + "      public native @Cast(\"__uid_t\") int si_uid(); public native siginfo_t si_uid(int si_uid);	/* Real user ID of sending process.  */\n"
                     + "      public native int si_timerid(); public native siginfo_t si_timerid(int si_timerid);		/* Timer ID.  */\n"
                     + "      public native int si_overrun(); public native siginfo_t si_overrun(int si_overrun);	/* Overrun count.  */\n"
                     + "      public native @ByRef sigval si_value(); public native siginfo_t si_value(sigval si_value);	/* Signal value.  */\n"
                     + "      public native int si_int(); public native siginfo_t si_int(int si_int);	/* Signal value.  */\n"
                     + "      public native Pointer si_ptr(); public native siginfo_t si_ptr(Pointer si_ptr);	/* Signal value.  */\n"
                     + "      public native int si_status(); public native siginfo_t si_status(int si_status);	/* Exit value or signal.  */\n"
                     + "      public native @Cast(\"__clock_t\") long si_utime(); public native siginfo_t si_utime(long si_utime);\n"
                     + "      public native @Cast(\"__clock_t\") long si_stime(); public native siginfo_t si_stime(long si_stime);\n"
                     + "      public native Pointer si_addr(); public native siginfo_t si_addr(Pointer si_addr);	/* Faulting insn/memory ref.  */\n"
//                     + "      public native short si_addr_lsb(); public native siginfo_t si_addr_lsb(short si_addr_lsb);	/* Valid LSB of the reported address.  */\n"
//                     + "      public native Pointer si_lower(); public native siginfo_t si_lower(Pointer si_lower);\n"
//                     + "      public native Pointer si_upper(); public native siginfo_t si_upper(Pointer si_upper);\n"
                     + "      public native long si_band(); public native siginfo_t si_band(long si_band);	/* Band event for SIGPOLL.  */\n"
                     + "      public native int si_fd(); public native siginfo_t si_fd(int si_fd);\n"
                     + "      public native Pointer si_call_addr(); public native siginfo_t si_call_addr(Pointer si_call_addr);	/* Calling user insn.  */\n"
                     + "      public native int si_syscall(); public native siginfo_t si_syscall(int si_syscall);	/* Triggering system call number.  */\n"
                     + "      public native @Cast(\"unsigned int\") int si_arch(); public native siginfo_t si_arch(int si_arch); /* AUDIT_ARCH_* of syscall.  */\n"
                     + "  }\n"))

               .put(new Info("sysinfo::_f").javaText("@MemberGetter public native @Cast(\"char*\") BytePointer _f();    /* Padding: libc5 uses this.. */"))

               .put(new Info("LC_GLOBAL_LOCALE").cppTypes("__locale_t").translate(false))
               .put(new Info("RTLD_NEXT", "RTLD_DEFAULT", "RTLD_SELF", "RTLD_MAIN_ONLY").cppTypes("void*").translate(false))
               .put(new Info("RLIM_INFINITY", "RLIM_SAVED_MAX", "RLIM_SAVED_CUR").cppTypes("long").translate(false))
               .put(new Info("CLOCKS_PER_SEC", "NSIG").cppTypes("long").translate(false))
               .put(new Info("SIG_ERR", "SIG_DFL", "SIG_IGN", "SIG_HOLD").cppTypes("__sighandler_t*").translate(false))
               .put(new Info("_SS_PADSIZE", "__SI_PAD_SIZE", "__SIGEV_PAD_SIZE", "SIGCLD", "SIGPOLL",
                             "FP_XSTATE_MAGIC2_SIZE", "WCOREFLAG").cppTypes("int").translate(false))

               .put(new Info("__cpuid").cppTypes("void", "int", "int&", "int&", "int&", "int&"))
               .put(new Info("__cpuid_count").cppTypes("void", "int", "int", "int&", "int&", "int&", "int&"))

               .put(new Info("cmsghdr").purify())
               .put(new Info("cmsghdr::__flexarr", "getwd", "getpw", "lchmod", "mktemp", "revoke", "setlogin",
                             "sigblock", "siggetmask", "sigsetmask", "sigreturn", "sigstack(sigstack*, sigstack*)",
                             "__sched_param", "_fpx_sw_bytes", "_xsave_hdr", "_xstate", "_ymmh_state", "__key").skip());
    }
}
