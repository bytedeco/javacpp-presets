/*
 * Copyright (C) 2017-2020 Samuel Audet
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
@Properties(inherit = javacpp.class, value = {@Platform(value = "windows", define = {"WINVER 0x0601", "_WIN32_WINNT 0x0601"},
    include = {"minwindef.h", "guiddef.h", "winnt.h", "minwinbase.h", "processenv.h", "fileapi.h", "debugapi.h", "utilapiset.h",
               "handleapi.h", "errhandlingapi.h", "fibersapi.h", "namedpipeapi.h", "profileapi.h", "heapapi.h", "ioapiset.h",
               "synchapi.h", "interlockedapi.h", "processthreadsapi.h", "sysinfoapi.h", "memoryapi.h", "threadpoollegacyapiset.h",
               "threadpoolapiset.h", /*"bemapiset.h",*/ "jobapi.h", "wow64apiset.h", "libloaderapi.h", "securitybaseapi.h",
               "namespaceapi.h", "systemtopologyapi.h", "processtopologyapi.h", "securityappcontainer.h", "realtimeapiset.h",
               "WinBase.h", "timezoneapi.h", "Psapi.h", "TlHelp32.h", "mmsyscom.h", "timeapi.h"},
    includepath = {"C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared/",
                   "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/um/"},
    link = {"ntdll", "AdvAPI32", "mincore", "synchronization", "User32", "Psapi", "winmm"},
    linkpath = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x86/"),
@Platform(value = "windows-x86_64",
    linkpath = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64/"),
@Platform(value = "windows-arm64",
    linkpath = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/arm64/")},
        target = "org.bytedeco.systems.windows", global = "org.bytedeco.systems.global.windows")
@NoException
public class windows implements BuildEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "systems"); }

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;
    private boolean isArm;
    private boolean is64bits;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
        this.isArm = properties.getProperty("platform").contains("arm");
        this.is64bits = properties.getProperty("platform").contains("64");
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("winnt.h")
                       .linePatterns("#define [a-zA-Z0-9]+ +_[a-zA-Z0-9_]+").skip())
               .put(new Info("__dmb", "__dsb", "__dsb").skip())
               .put(new Info("processenv.h", "fileapi.h", "debugapi.h", "namedpipeapi.h", "synchapi.h",
                             "processthreadsapi.h", "sysinfoapi.h", "memoryapi.h", "libloaderapi.h",
                             "securitybaseapi.h", "WinBase.h", "Psapi.h", "TlHelp32.h")
                       .linePatterns("#define [a-zA-Z0-9]+ +[a-zA-Z0-9_]+W").skip())

               .put(new Info("_X86_", "defined(_X86_)", "_M_IX86", "defined(_M_IX86)",
                             "defined(_X86_) && !defined(_M_HYBRID_X86_ARM64)").define(!isArm && !is64bits))
               .put(new Info("_AMD64_", "defined(_AMD64_)", "_M_AMD64", "defined(_M_AMD64)",
                             "defined(_AMD64_) && !defined(_ARM64EC_)").define(!isArm && is64bits))
               .put(new Info("_ARM_", "defined(_ARM_)", "_M_ARM", "defined(_M_ARM)").define(isArm && !is64bits))
               .put(new Info("_ARM64_", "defined(_ARM64_)", "_M_ARM64", "defined(_M_ARM64)",
                             "__ARM64_COMPILER_BITTEST64_WORKAROUND", "!defined(__ARM64_COMPILER_BITTEST64_WORKAROUND)",
                             "defined(_ARM64_) || defined(_CHPE_X86_ARM64_) || defined(_ARM64EC_)").define(isArm && is64bits))
               .put(new Info("_WIN64", "defined(_WIN64)", "(_WIN32_WINNT >= 0x0601) && !defined(MIDL_PASS)").define(is64bits))

               .put(new Info("__cplusplus", "defined(_MSC_EXTENSIONS)", "(_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)", "UNICODE",
                             "_WIN32_WINNT >= 0x0601", "(_WIN32_WINNT >= _WIN32_WINNT_WIN7)", "(_WIN32_WINNT < _WIN32_WINNT_WIN8)",
                             "(NTDDI_VERSION < NTDDI_WINTHRESHOLD)", "NTDDI_VERSION < NTDDI_WINTHRESHOLD", "!defined(_M_ARM64EC)").define(true))

               .put(new Info("defined(NONAMELESSUNION) || !defined(_MSC_EXTENSIONS)", "defined(VOLATILE_ACCESSOR_LIB)",
                             "_ARM64EC_", "defined(_ARM64EC_)", "_M_ARM64EC", "defined(_M_ARM64EC)", "_IA64_", "defined(_IA64_)", "_M_IA64", "defined(_M_IA64)",
                             "_M_CEE_PURE", "defined(_M_CEE_PURE)", "_MAC", "__midl", "defined(__midl)", "MIDL_PASS", "defined(MIDL_PASS)", "_PREFAST_", "defined(_PREFAST_)",
                             "defined(_M_IA64) && !defined(RC_INVOKED) && !defined(MIDL_PASS)",
                             "!defined(__midl) && !defined(GENUTIL) && !defined(_GENIA64_) && defined(_IA64_)",
                             "(NTDDI_VERSION < NTDDI_WINXP)", "(NTDDI_VERSION >= NTDDI_WIN8)", "NTDDI_VERSION >= NTDDI_WIN8",
                             "(defined(_M_ARM64) || defined(_M_ARM64EC)) && !defined(__midl) && !defined(_M_CEE_PURE)",
                             "!defined(MIDL_PASS) && !defined(SORTPP_PASS) && !defined(RC_INVOKED)",
                             "defined(_KERNEL_MODE) && defined(__SANITIZE_ADDRESS__) && defined(CSAN_ON_ASAN)",
                             "(NTDDI_VERSION >= NTDDI_WINBLUE)", "NTDDI_VERSION >= NTDDI_WINBLUE", "(_WIN32_WINNT >= _WIN32_WINNT_WINBLUE)",
                             "(NTDDI_VERSION >= NTDDI_WINTHRESHOLD)", "NTDDI_VERSION >= NTDDI_WINTHRESHOLD", "(_WIN32_WINNT >= _WIN32_WINNT_WINTHRESHOLD)",
                             "(NTDDI_VERSION >= NTDDI_WIN10)", "NTDDI_VERSION >= NTDDI_WIN10", "(_WIN32_WINNT >= 0x0A00)", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_TH2)", "NTDDI_VERSION >= NTDDI_WIN10_TH2", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_TH2)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_RS1)", "NTDDI_VERSION >= NTDDI_WIN10_RS1", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_RS1)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_RS2)", "NTDDI_VERSION >= NTDDI_WIN10_RS2", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_RS2)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_RS3)", "NTDDI_VERSION >= NTDDI_WIN10_RS3", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_RS3)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_RS4)", "NTDDI_VERSION >= NTDDI_WIN10_RS4", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_RS4)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_RS5)", "NTDDI_VERSION >= NTDDI_WIN10_RS5", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_RS5)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_19H1)", "NTDDI_VERSION >= NTDDI_WIN10_19H1", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_19H1)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_VB)", "NTDDI_VERSION >= NTDDI_WIN10_VB", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_VB)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_MN)", "NTDDI_VERSION >= NTDDI_WIN10_MN", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_MN)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_FE)", "NTDDI_VERSION >= NTDDI_WIN10_FE", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_FE)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_CO)", "NTDDI_VERSION >= NTDDI_WIN10_CO", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_CO)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_NI)", "NTDDI_VERSION >= NTDDI_WIN10_NI", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_NI)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_CU)", "NTDDI_VERSION >= NTDDI_WIN10_CU", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_CU)",
                             "(NTDDI_VERSION >= NTDDI_WIN11_ZN)", "NTDDI_VERSION >= NTDDI_WIN11_ZN", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_ZN)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_GA)", "NTDDI_VERSION >= NTDDI_WIN10_GA", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_GA)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_GE)", "NTDDI_VERSION >= NTDDI_WIN10_GE", "(_WIN32_WINNT >= _WIN32_WINNT_WIN10_GE)",
                             "(NTDDI_VERSION >= NTDDI_WIN11_GA)", "NTDDI_VERSION >= NTDDI_WIN11_GA", "(_WIN32_WINNT >= _WIN32_WINNT_WIN11_GA)",
                             "(NTDDI_VERSION >= NTDDI_WIN11_GE)", "NTDDI_VERSION >= NTDDI_WIN11_GE", "(_WIN32_WINNT >= _WIN32_WINNT_WIN11_GE)",
                             "(NTDDI_VERSION >= NTDDI_WIN10_RS5) || (NTDDI_VERSION >= NTDDI_WIN8)", "defined(_M_HYBRID_X86_ARM64)", "defined(_CHPE_X86_ARM64_)",
                             "defined(NTDDI_WIN11_GA) && (NTDDI_VERSION >= NTDDI_WIN11_GA)", "defined(NTDDI_WIN11_GE) && (NTDDI_VERSION >= NTDDI_WIN11_GE)",
                             "(_WIN32_WINNT >= 0x0602)", "(_WIN32_WINNT >= 0x0603)", "(_WIN32_WINNT >= _WIN32_WINNT_WIN8)",
                             "((NTDDI_VERSION >= NTDDI_WIN8) && !defined(_CONTRACT_GEN)) || (_APISET_RTLSUPPORT_VER > 0x0100)",
                             "defined(_DBG_MEMCPY_INLINE_) && !defined(MIDL_PASS) && !defined(_MEMCPY_INLINE_) && !defined(_CRTBLD)",
                             "((NTDDI_VERSION > NTDDI_WINBLUE) ||     (NTDDI_VERSION == NTDDI_WINBLUE && defined(WINBLUE_KBSPRING14)))",
                             "((NTDDI_VERSION >= NTDDI_WIN8) && !defined(_CONTRACT_GEN)) || (_APISET_INTERLOCKED_VER > 0x0100)",
                             "((!defined(_CONTRACT_GEN) && (_WIN32_WINNT >= _WIN32_WINNT_WIN8)) || (_APISET_SECURITYBASE_VER > 0x0100))",
                             "(_WIN32_WINNT >= _WIN32_WINNT_WINBLUE)", "defined(WINBASE_DECLARE_RESTORE_LAST_ERROR)", "(PSAPI_VERSION > 1)").define(false))

               .put(new Info("__declspec(deprecated)", "DECLSPEC_DEPRECATED", "DECLSPEC_DEPRECATED_DDK", "NOT_BUILD_WINDOWS_DEPRECATE").annotations("@Deprecated").cppTypes())

               .put(new Info("far", "near", "pascal", "cdecl", "_cdecl", "__cdecl", "__stdcall", "APIENTRY", "APIPRIVATE", "CALLBACK",
                             "FAR", "NEAR", "PASCAL", "CDECL", "CDECL_NON_WVMPURE", "CONST", "WINAPI", "WINAPIV", "WINAPI_INLINE",
                             "WINBASEAPI", "WINADVAPI", "WINUSERAPI", "REFGUID", "REFIID", "REFCLSID", "REFFMTID",

                             "DECLSPEC_ADDRSAFE", "DECLSPEC_ALIGN", "DECLSPEC_CACHEALIGN", "DECLSPEC_GUARDNOCF", "DECLSPEC_IMPORT", "DECLSPEC_NOINLINE",
                             "DECLSPEC_NORETURN", "DECLSPEC_NOTHROW", "DECLSPEC_NOVTABLE", "DECLSPEC_SAFEBUFFERS","DECLSPEC_SELECTANY", "NOP_FUNCTION",
                             "DECLSPEC_NOINITALL", "DECLSPEC_GUARD_SUPPRESS", "DECLSPEC_NOSANITIZEADDRESS", "DECLSPEC_CHPE_GUEST", "DECLSPEC_CHPE_PATCHABLE", "DECLSPEC_RESTRICT",
                             "EXTERN_C_START", "EXTERN_C_END", "WIN_NOEXCEPT", "WIN_NOEXCEPT_PFN", "PFORCEINLINE", "NONVOL_INT_SIZE_ARM64", "NONVOL_FP_SIZE_ARM64", "_ENUM_FLAG_CONSTEXPR",
                             "MEM_EXTENDED_PARAMETER_NUMA_NODE_MANDATORY", "GUID_HUPR_ADAPTIVE_DISPLAY_TIMEOUT", "GUID_HUPR_ADAPTIVE_DIM_TIMEOUT",
                             "IMAGE_POLICY_METADATA_VERSION", "IMAGE_POLICY_SECTION_NAME", "IMAGE_POLICY_METADATA_NAME", "STDAPI_CHPE_PATCHABLE",
                             "ASAN_WARNING_DISABLE_4714_PUSH", "ASAN_WARNING_DISABLE_4714_POP", "DEPRECATED_NO_MESSAGE_STDAPI", "DEPRECATED_NO_MESSAGE_STDAPIV",
                             "__export", "__override", "_Null_terminated_", "_NullNull_terminated_", "EXTERN_C", "FORCEINLINE", "CFORCEINLINE", "ICEFORCEINLINE",
                             "STKFORCEINLINE", "RESTRICTED_POINTER", "UNALIGNED", "UNALIGNED64", "NTAPI", "NTAPI_INLINE", "NTSYSAPI", "NTSYSCALLAPI",
                             "STDMETHODCALLTYPE", "STDMETHODVCALLTYPE", "STDAPICALLTYPE", "STDAPIVCALLTYPE", "STDAPI", "STDMETHODIMP", "STDOVERRIDEMETHODIMP",
                             "IFACEMETHODIMP", "STDAPIV", "STDMETHODIMPV", "STDOVERRIDEMETHODIMPV", "STDOVERRIDEMETHODIMPV", "IFACEMETHODIMPV", "DEFAULT_UNREACHABLE",

                             "__analysis_noreturn", "_Check_return_", "_Must_inspect_result_", "_Ret_maybenull_", "_Ret_writes_", "_Success_", "_When_",
                             "__callback", "__inline", "_Field_z_", "_Frees_ptr_opt_", "_Pre_", "_Pre_notnull_", "_Pre_valid_", "_Reserved_", "_Always_",
                             "_Post_", "_Post_equals_last_error_", "_Post_invalid_", "_Post_readable_byte_size_", "_Post_writable_byte_size_", "_Post_satisfies_", "_Post_ptr_invalid_",
                             "_IRQL_requires_same_", "_IRQL_requires_max_", "_Function_class_", "__forceinline", "_Interlocked_operand_", "_Struct_size_bytes_",
                             "_Maybe_raises_SEH_exception_", "__drv_aliasesMem", "__drv_freesMem", "__drv_preferredFunction", "__out_data_source",
                             "_In_", "_In_reads_", "_In_reads_bytes_", "_In_reads_opt_", "_In_opt_", "_In_reads_bytes_opt_", "_In_range_", "_In_z_", "_Ret_range_",
                             "_Out_", "_Outptr_", "_Out_writes_", "_Out_writes_all_", "_Out_writes_bytes_", "_Out_writes_bytes_to_", "_Out_writes_to_",
                             "_Out_opt_", "_Outptr_opt_", "_Out_writes_opt_", "_Out_writes_to_opt_", "_Out_writes_bytes_opt_", "_Out_writes_bytes_to_opt_", "_Outptr_result_z_",
                             "_Outptr_result_buffer_maybenull_", "_Out_writes_bytes_all_", "_Outptr_opt_result_bytebuffer_all_", "_Outptr_opt_result_maybenull_",
                             "_Inout_", "_At_", "_Inout_opt_", "_Inout_updates_", "_Inout_updates_z_", "_Inout_updates_bytes_", "_Inout_updates_opt_",

                             "InterlockedExchangeNoFence8", "InterlockedExchangeAcquire8", "InterlockedExchangeNoFence16", "InterlockedExchangeAcquire16",
                             "InterlockedAndAcquire16", "InterlockedAndRelease16", "InterlockedAndNoFence16", "InterlockedOrAcquire16", "InterlockedOrRelease16",
                             "InterlockedOrNoFence16", "InterlockedXorAcquire16", "InterlockedXorRelease16", "InterlockedXorNoFence16",
                             "InterlockedAndAffinity", "InterlockedOr64", "InterlockedOrAffinity", "InterlockedExchangeAcquire64", "InterlockedExchangeNoFence64",
                             "InterlockedCompareExchangeAcquire64", "InterlockedCompareExchangeRelease64", "InterlockedCompareExchangeNoFence64",
                             "InterlockedPushListSList", "InterlockedCompareExchangePointer", "InterlockedCompareExchangePointerAcquire",
                             "InterlockedCompareExchangePointerRelease", "InterlockedCompareExchangePointerNoFence", "InterlockedIncrementAcquire64",
                             "ReadSizeTAcquire", "ReadSizeTNoFence", "ReadSizeTRaw", "WriteSizeTRelease", "WriteSizeTNoFence", "WriteSizeTRaw",
                             "ReadLongPtrAcquire", "ReadLongPtrNoFence", "ReadLongPtrRaw", "WriteLongPtrRelease", "WriteLongPtrNoFence", "WriteLongPtrRaw",
                             "ReadULongPtrAcquire", "ReadULongPtrNoFence", "ReadULongPtrRaw", "WriteULongPtrRelease", "WriteULongPtrNoFence", "WriteULongPtrRaw",
                             "ACTIVATIONCONTEXTINFOCLASS", "CaptureStackBackTrace", "CopyMemory", "FillMemory", "MoveMemory", "SecureZeroMemory", "ZeroMemory",
                             "CopyVolatileMemory", "MoveVolatileMemory", "FillVolatileMemory", "SecureZeroMemory2", "ZeroVolatileMemory",
                             "CopyDeviceMemory", "FillDeviceMemory", "ZeroDeviceMemory", "EncodeRemotePointer", "DecodeRemotePointer",
                             "GetSystemWow64Directory", "SpeculationFence", "FatalAppExitW", "FatalAppExit",
                             "_ASSEMBLY_DLL_REDIRECTION_DETAILED_INFORMATION", "ASSEMBLY_DLL_REDIRECTION_DETAILED_INFORMATION",
                             "PASSEMBLY_DLL_REDIRECTION_DETAILED_INFORMATION", "PCASSEMBLY_DLL_REDIRECTION_DETAILED_INFORMATION",
                             "PGET_MODULE_HANDLE_EX", "EXCEPTION_POSSIBLE_DEADLOCK", "MICROSOFT_WINDOWS_WINBASE_H_DEFINE_INTERLOCKED_CPLUSPLUS_OVERLOADS",
                             "PENUM_PAGE_FILE_CALLBACK").annotations().cppTypes())

               .put(new Info("BOOLEAN")
                        .cast().valueTypes("boolean").pointerTypes("BoolPointer", "boolean[]"))

               .put(new Info("PBOOLEAN")
                        .cast().valueTypes("BoolPointer", "boolean[]").pointerTypes("PointerPointer"))

               .put(new Info("BOOL", "WINBOOL")
                        .cast().valueTypes("boolean").pointerTypes("IntPointer", "IntBuffer", "int[]"))

               .put(new Info("CHAR", "UCHAR", "CCHAR", "BYTE", "INT8", "UINT8")
                       .cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))

               .put(new Info("WCHAR")
                       .cast().valueTypes("char").pointerTypes("CharPointer", "CharBuffer", "char[]"))

               .put(new Info("PCHAR", "LPCHAR", "PUCHAR", "LPUCHAR", "PBYTE", "LPBYTE", "PCH", "LPCH", "PSTR", "LPSTR")
                       .cast().valueTypes("BytePointer", "ByteBuffer", "byte[]").pointerTypes("PointerPointer"))

               .put(new Info("LPCCH", "PCCH", "LPCSTR", "PCSTR")
                       .cast().valueTypes("BytePointer", "ByteBuffer", "byte[]", "String").pointerTypes("PointerPointer"))

               .put(new Info("PWCHAR", "LPWCHAR", "PWCH", "LPWCH", "PWSTR", "LPWSTR",
                             "PCWCH", "LPCWCH", "PCWSTR", "LPCWSTR",
                             "PTSTR", "LPTSTR", "PCTSTR", "LPCTSTR")
                       .cast().valueTypes("CharPointer", "CharBuffer", "char[]").pointerTypes("PointerPointer"))

               .put(new Info("WORD", "LANGID", "INT16", "UINT16")
                       .cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))

               .put(new Info("DWORD", "INT", "LONG", "INT32", "UINT32",
                             "ACCESS_MASK", "EXCEPTION_DISPOSITION", "EXECUTION_STATE", "HRESULT", "SECURITY_INFORMATION")
                       .cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))

               .put(new Info("PDWORD", "LPDWORD", "PINT", "LPINT", "PLONG", "LPLONG", "PACCESS_MASK", "PEXECUTION_STATE")
                       .cast().valueTypes("IntPointer", "IntBuffer", "int[]").pointerTypes("PointerPointer"))

               .put(new Info("DWORD_PTR", "INT_PTR", "LONG_PTR", "UINT_PTR", "ULONG_PTR", "KAFFINITY", "SIZE_T")
                       .cast().valueTypes("long").pointerTypes("SizeTPointer"))

               .put(new Info("PDWORD_PTR", "PULONG_PTR", "PSIZE_T")
                       .cast().valueTypes("SizeTPointer").pointerTypes("PointerPointer"))

               .put(new Info("DWORD64", "LONG64", "ULONG64", "DWORDLONG", "LONGLONG", "ULONGLONG", "INT64", "UINT64")
                       .cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))

               .put(new Info("PDWORD64", "PLONG64", "PULONG64", "PDWORDLONG", "PLONGLONG", "PULONGLONG", "PLARGE_INTEGER", "PULARGE_INTEGER", "PLUID",
                             "PSYSTEM_PROCESSOR_CYCLE_TIME_INFORMATION")
                       .cast().valueTypes("LongPointer", "LongBuffer", "long[]").pointerTypes("PointerPointer"))

               .put(new Info("VOID").cppTypes().valueTypes("void").pointerTypes("Pointer"))

               .put(new Info("PVOID", "LPVOID", "PVOID64", "PSID", "PLUID",
                             "HANDLE", "HINSTANCE", "HMODULE", "HWND", "HHOOK", "HRSRC")
                       .cast().valueTypes("Pointer").pointerTypes("PointerPointer"))

               .put(new Info("PHANDLE").cast().valueTypes("PointerPointer"))
               .put(new Info("_CONTEXT").pointerTypes("CONTEXT"))
               .put(new Info("ARM64_NT_CONTEXT").pointerTypes("CONTEXT"))
               .put(new Info("_LIST_ENTRY").pointerTypes("LIST_ENTRY"))
               .put(new Info("_SINGLE_LIST_ENTRY").pointerTypes("SINGLE_LIST_ENTRY"))
               .put(new Info("_EXCEPTION_RECORD").pointerTypes("EXCEPTION_RECORD"))
               .put(new Info("_EXCEPTION_POINTERS").pointerTypes("EXCEPTION_POINTERS"))
               .put(new Info("_EXCEPTION_REGISTRATION_RECORD").pointerTypes("EXCEPTION_REGISTRATION_RECORD"))
               .put(new Info("_NT_TIB").pointerTypes("NT_TIB"))
               .put(new Info("_SLIST_ENTRY").pointerTypes("SLIST_ENTRY"))
               .put(new Info("_RTL_CRITICAL_SECTION").pointerTypes("RTL_CRITICAL_SECTION"))
               .put(new Info("_RTL_CRITICAL_SECTION_DEBUG").pointerTypes("RTL_CRITICAL_SECTION_DEBUG"))
               .put(new Info("LPTOP_LEVEL_EXCEPTION_FILTER").valueTypes("PTOP_LEVEL_EXCEPTION_FILTER"))
               .put(new Info("LPPROC_THREAD_ATTRIBUTE_LIST").valueTypes("_PROC_THREAD_ATTRIBUTE_LIST"))
               .put(new Info("PBAD_MEMORY_CALLBACK_ROUTINE").valueTypes("BAD_MEMORY_CALLBACK_ROUTINE"))
               .put(new Info("_ACTIVATION_CONTEXT", "DISPATCHER_CONTEXT", "DISPATCHER_CONTEXT_ARM64", "GET_RUNTIME_FUNCTION_CALLBACK", "FLOATING_SAVE_AREA",
                             "KNONVOLATILE_CONTEXT_POINTERS", "PKNONVOLATILE_CONTEXT_POINTERS", "OUT_OF_PROCESS_FUNCTION_TABLE_CALLBACK",
                             "PEXCEPTION_FILTER", "PTERMINATION_HANDLER", "RUNTIME_FUNCTION", "UMS_COMPLETION_LIST", "PUMS_COMPLETION_LIST",
                             "UMS_CONTEXT", "PUMS_CONTEXT", "UMS_SCHEDULER_STARTUP_INFO", "PUMS_SCHEDULER_STARTUP_INFO",
                             "UMS_SYSTEM_THREAD_INFORMATION", "PUMS_SYSTEM_THREAD_INFORMATION", "UNWIND_HISTORY_TABLE",
                             "PUNWIND_HISTORY_TABLE", "UNWIND_HISTORY_TABLE_ENTRY", "_TEB").cast().pointerTypes("Pointer"))
               .put(new Info("UOW")./*cast().*/pointerTypes("GUID"))

               .put(new Info("IID_NULL", "CLSID_NULL", "FMTID_NULL").cppTypes("GUID").translate(false))
               .put(new Info("RLIM_INFINITY", "RLIM_SAVED_MAX", "RLIM_SAVED_CUR").cppTypes("long").translate(false))
               .put(new Info("CLK_TCK", "CLOCKS_PER_SEC").cppTypes("long").translate(false))
               .put(new Info("NSIG").cppTypes("long").translate(false))
               .put(new Info("SECURITY_MAX_SID_SIZE", "MIN_ACL_REVISION").cppTypes("int").translate(false))
               .put(new Info("SECURITY_NULL_SID_AUTHORITY", "SECURITY_WORLD_SID_AUTHORITY", "SECURITY_LOCAL_SID_AUTHORITY",
                             "SECURITY_CREATOR_SID_AUTHORITY", "SECURITY_NON_UNIQUE_AUTHORITY", "SECURITY_RESOURCE_MANAGER_AUTHORITY",
                             "SECURITY_NT_AUTHORITY", "SECURITY_APP_PACKAGE_AUTHORITY", "SECURITY_MANDATORY_LABEL_AUTHORITY",
                             "SECURITY_SCOPED_POLICY_ID_AUTHORITY", "SECURITY_AUTHENTICATION_AUTHORITY", "SECURITY_PROCESS_TRUST_AUTHORITY")
                       .cppTypes("std::vector<BYTE>").translate(false))
               .put(new Info("SYSTEM_LUID", "ANONYMOUS_LOGON_LUID", "LOCALSERVICE_LUID", "NETWORKSERVICE_LUID", "IUSER_LUID", "PROTECTED_TO_SYSTEM_LUID")
                       .cppTypes("LUID").translate(false))
               .put(new Info("MEMORY_CURRENT_PARTITION_HANDLE", "MEMORY_SYSTEM_PARTITION_HANDLE", "MEMORY_EXISTING_VAD_PARTITION_HANDLE",
                             "INVALID_HANDLE_VALUE").cppTypes("HANDLE").pointerTypes("Pointer").translate(false))

               .put(new Info("Int32x32To64", "UInt32x32To64").cppTypes("long", "int", "int"))
               .put(new Info("Int64ShllMod32", "Int64ShraMod32", "Int64ShrlMod32").cppTypes("long", "long", "int"))
               .put(new Info("MAKELANGID", "MAKELCID").cppTypes("int", "int", "int"))
               .put(new Info("MAKESORTLCID").cppTypes("int", "int", "int", "int"))
               .put(new Info("PRIMARYLANGID", "SUBLANGID", "LANGIDFROMLCID", "SORTIDFROMLCID", "SORTVERSIONFROMLCID").cppTypes("int", "int"))
               .put(new Info("ProcThreadAttributeValue").cppTypes("int", "int", "int", "int", "int"))

               .put(new Info("_PACKEDEVENTINFO", "_EVENTSFORLOGFILE", "ACTIVATION_CONTEXT_COMPATIBILITY_INFORMATION").purify())
               .put(new Info("_mm_prefetch", "GUID_AUDIO_PLAYBACK", "GUID_CS_BATTERY_SAVER_ACTION", "GUID_CS_BATTERY_SAVER_THRESHOLD",
                             "GUID_CS_BATTERY_SAVER_TIMEOUT", "GUID_IDLE_RESILIENCY_PLATFORM_STATE", "GUID_VIDEO_FULLSCREEN_PLAYBACK",
                             "LookupAccountSidLocalA", "LookupAccountSidLocalW", "LookupAccountNameLocalA", "LookupAccountNameLocalW",
                             "MEM_EXTENDED_PARAMETER", "_ARM64_NT_CONTEXT", "_DISPATCHER_CONTEXT_ARM64", "IMAGE_POLICY_METADATA", "_IMAGE_POLICY_METADATA",
                             "__shiftleft128", "__shiftright128", "__break",
                             "WinMain", "wWinMain").skip())

               .put(new Info("MMRESULT").cppTypes("UINT").translate(false))
               .put(new Info("WINMMAPI").cppTypes().annotations())
               .put(new Info("HDRVR").cast().valueTypes("Pointer"));
    }
}
