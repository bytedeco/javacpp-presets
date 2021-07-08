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
@Properties(inherit = javacpp.class, value = {@Platform(value = "windows-x86", define = {"WINVER 0x0601", "_WIN32_WINNT 0x0601"},
    include = {"minwindef.h", "guiddef.h", "winnt.h", "minwinbase.h", "processenv.h", "fileapi.h", "debugapi.h", "utilapiset.h",
               "handleapi.h", "errhandlingapi.h", "fibersapi.h", "namedpipeapi.h", "profileapi.h", "heapapi.h", "ioapiset.h",
               "synchapi.h", "interlockedapi.h", "processthreadsapi.h", "sysinfoapi.h", "memoryapi.h", "threadpoollegacyapiset.h",
               "threadpoolapiset.h", /*"bemapiset.h",*/ "jobapi.h", "wow64apiset.h", "libloaderapi.h", "securitybaseapi.h",
               "namespaceapi.h", "systemtopologyapi.h", "processtopologyapi.h", "securityappcontainer.h", "realtimeapiset.h",
               "WinBase.h", "timezoneapi.h", "Psapi.h", "TlHelp32.h"},
    includepath = {"C:/Program Files (x86)/Windows Kits/8.1/Include/shared/",
                   "C:/Program Files (x86)/Windows Kits/8.1/Include/um/"},
    link = {"ntdll", "AdvAPI32", "mincore", "synchronization", "User32", "Psapi"},
    linkpath = "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x86/"),
@Platform(value = "windows-x86_64",
    linkpath = "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/")},
        target = "org.bytedeco.systems.windows", global = "org.bytedeco.systems.global.windows")
@NoException
public class windows implements BuildEnabled, InfoMapper {
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
        infoMap.put(new Info("winnt.h")
                       .linePatterns("#define [a-zA-Z0-9]+ +_[a-zA-Z0-9_]+").skip())
               .put(new Info("processenv.h", "fileapi.h", "debugapi.h", "namedpipeapi.h", "synchapi.h",
                             "processthreadsapi.h", "sysinfoapi.h", "memoryapi.h", "libloaderapi.h",
                             "securitybaseapi.h", "WinBase.h", "Psapi.h", "TlHelp32.h")
                       .linePatterns("#define [a-zA-Z0-9]+ +[a-zA-Z0-9_]+W").skip())

               .put(new Info("_X86_", "defined(_X86_)", "_M_IX86", "defined(_M_IX86)").define(!is64bits))
               .put(new Info("_AMD64_", "defined(_AMD64_)", "_M_AMD64", "defined(_M_AMD64)",
                             "_WIN64", "defined(_WIN64)", "(_WIN32_WINNT >= 0x0601) && !defined(MIDL_PASS)").define(is64bits))

               .put(new Info("__cplusplus", "defined(_MSC_EXTENSIONS)", "(_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)", "UNICODE",
                             "_WIN32_WINNT >= 0x0601", "(_WIN32_WINNT >= _WIN32_WINNT_WIN7)", "(_WIN32_WINNT < _WIN32_WINNT_WIN8)").define(true))

               .put(new Info("defined(NONAMELESSUNION) || !defined(_MSC_EXTENSIONS)",
                             "_ARM_", "defined(_ARM_)", "_M_ARM", "defined(_M_ARM)", "_IA64_", "defined(_IA64_)", "_M_IA64", "defined(_M_IA64)",
                             "_M_CEE_PURE", "defined(_M_CEE_PURE)", "_MAC", "__midl", "defined(__midl)", "MIDL_PASS", "defined(MIDL_PASS)",
                             "defined(_M_IA64) && !defined(RC_INVOKED) && !defined(MIDL_PASS)",
                             "!defined(__midl) && !defined(GENUTIL) && !defined(_GENIA64_) && defined(_IA64_)",
                             "(NTDDI_VERSION < NTDDI_WINXP)", "(NTDDI_VERSION >= NTDDI_WIN8)", "NTDDI_VERSION >= NTDDI_WIN8",
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
                             "__export", "__override", "_Null_terminated_", "_NullNull_terminated_", "EXTERN_C", "FORCEINLINE", "CFORCEINLINE", "ICEFORCEINLINE",
                             "STKFORCEINLINE", "RESTRICTED_POINTER", "UNALIGNED", "UNALIGNED64", "NTAPI", "NTAPI_INLINE", "NTSYSAPI", "NTSYSCALLAPI",
                             "STDMETHODCALLTYPE", "STDMETHODVCALLTYPE", "STDAPICALLTYPE", "STDAPIVCALLTYPE", "STDAPI", "STDMETHODIMP", "STDOVERRIDEMETHODIMP",
                             "IFACEMETHODIMP", "STDAPIV", "STDMETHODIMPV", "STDOVERRIDEMETHODIMPV", "STDOVERRIDEMETHODIMPV", "IFACEMETHODIMPV", "DEFAULT_UNREACHABLE",

                             "__analysis_noreturn", "_Check_return_", "_Must_inspect_result_", "_Ret_maybenull_", "_Ret_writes_", "_Success_", "_When_",
                             "__callback", "__inline", "_Field_z_", "_Frees_ptr_opt_", "_Pre_", "_Pre_notnull_", "_Pre_valid_", "_Reserved_",
                             "_Post_", "_Post_equals_last_error_", "_Post_invalid_", "_Post_readable_byte_size_", "_Post_writable_byte_size_", "_Post_satisfies_",
                             "_IRQL_requires_same_", "_IRQL_requires_max_", "_Function_class_", "__forceinline", "_Interlocked_operand_", "_Struct_size_bytes_",
                             "_Maybe_raises_SEH_exception_", "__drv_aliasesMem", "__drv_freesMem", "__drv_preferredFunction", "__out_data_source",
                             "_In_", "_In_reads_", "_In_reads_bytes_", "_In_reads_opt_", "_In_opt_", "_In_reads_bytes_opt_", "_Ret_range_",
                             "_Out_", "_Outptr_", "_Out_writes_", "_Out_writes_all_", "_Out_writes_bytes_", "_Out_writes_bytes_to_", "_Out_writes_to_",
                             "_Out_opt_", "_Outptr_opt_", "_Out_writes_opt_", "_Out_writes_to_opt_", "_Out_writes_bytes_opt_", "_Out_writes_bytes_to_opt_",
                             "_Out_writes_bytes_all_", "_Outptr_opt_result_bytebuffer_all_", "_Outptr_opt_result_maybenull_",
                             "_Inout_", "_At_", "_Inout_opt_", "_Inout_updates_", "_Inout_updates_z_", "_Inout_updates_bytes_", "_Inout_updates_opt_",

                             "InterlockedAndAffinity", "InterlockedOr64", "InterlockedOrAffinity", "InterlockedExchangeAcquire64", "InterlockedExchangeNoFence64",
                             "InterlockedCompareExchangeAcquire64", "InterlockedCompareExchangeRelease64", "InterlockedCompareExchangeNoFence64",
                             "InterlockedPushListSList", "InterlockedCompareExchangePointer", "InterlockedCompareExchangePointerAcquire",
                             "InterlockedCompareExchangePointerRelease", "InterlockedCompareExchangePointerNoFence", "InterlockedIncrementAcquire64",
                             "ReadSizeTAcquire", "ReadSizeTNoFence", "ReadSizeTRaw", "WriteSizeTRelease", "WriteSizeTNoFence", "WriteSizeTRaw",
                             "ReadLongPtrAcquire", "ReadLongPtrNoFence", "ReadLongPtrRaw", "WriteLongPtrRelease", "WriteLongPtrNoFence", "WriteLongPtrRaw",
                             "ReadULongPtrAcquire", "ReadULongPtrNoFence", "ReadULongPtrRaw", "WriteULongPtrRelease", "WriteULongPtrNoFence", "WriteULongPtrRaw",
                             "ACTIVATIONCONTEXTINFOCLASS", "CaptureStackBackTrace", "CopyMemory", "FillMemory", "MoveMemory", "SecureZeroMemory", "ZeroMemory",
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

               .put(new Info("CHAR", "UCHAR", "CCHAR", "BYTE")
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

               .put(new Info("WORD", "LANGID")
                       .cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))

               .put(new Info("DWORD", "INT", "LONG", "ACCESS_MASK", "EXCEPTION_DISPOSITION", "EXECUTION_STATE", "HRESULT", "SECURITY_INFORMATION")
                       .cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))

               .put(new Info("PDWORD", "LPDWORD", "PINT", "LPINT", "PLONG", "LPLONG", "PACCESS_MASK", "PEXECUTION_STATE")
                       .cast().valueTypes("IntPointer", "IntBuffer", "int[]").pointerTypes("PointerPointer"))

               .put(new Info("DWORD_PTR", "INT_PTR", "LONG_PTR", "UINT_PTR", "ULONG_PTR", "KAFFINITY", "SIZE_T")
                       .cast().valueTypes("long").pointerTypes("SizeTPointer"))

               .put(new Info("PDWORD_PTR", "PULONG_PTR", "PSIZE_T")
                       .cast().valueTypes("SizeTPointer").pointerTypes("PointerPointer"))

               .put(new Info("DWORD64", "LONG64", "ULONG64", "DWORDLONG", "LONGLONG", "ULONGLONG")
                       .cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))

               .put(new Info("PDWORD64", "PLONG64", "PULONG64", "PDWORDLONG", "PLONGLONG", "PULONGLONG", "PLUID",
                             "PSYSTEM_PROCESSOR_CYCLE_TIME_INFORMATION")
                       .cast().valueTypes("LongPointer", "LongBuffer", "long[]").pointerTypes("PointerPointer"))

               .put(new Info("VOID").cppTypes().valueTypes("void").pointerTypes("Pointer"))

               .put(new Info("PVOID", "LPVOID", "PVOID64", "PSID", "PLUID",
                             "HANDLE", "HINSTANCE", "HMODULE", "HWND", "HHOOK", "HRSRC")
                       .cast().valueTypes("Pointer").pointerTypes("PointerPointer"))

               .put(new Info("PHANDLE").cast().valueTypes("PointerPointer"))
               .put(new Info("_CONTEXT").pointerTypes("CONTEXT"))
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
               .put(new Info("_ACTIVATION_CONTEXT", "DISPATCHER_CONTEXT", "GET_RUNTIME_FUNCTION_CALLBACK", "FLOATING_SAVE_AREA",
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
               .put(new Info("SYSTEM_LUID", "ANONYMOUS_LOGON_LUID", "LOCALSERVICE_LUID", "NETWORKSERVICE_LUID", "IUSER_LUID")
                       .cppTypes("LUID").translate(false))
               .put(new Info("INVALID_HANDLE_VALUE").cppTypes("HANDLE").pointerTypes("Pointer").translate(false))

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
                             "WinMain", "wWinMain").skip());
    }
}
