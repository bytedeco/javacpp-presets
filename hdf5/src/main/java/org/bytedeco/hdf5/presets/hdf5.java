/*
 * Copyright (C) 2016-2017 Samuel Audet
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

package org.bytedeco.hdf5.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(target = "org.bytedeco.hdf5", global = "org.bytedeco.hdf5.global.hdf5", value = {
    @Platform(value = {"linux-x86", "linux-ppc64le", "macosx", "windows"},
            define = {"GENERIC_EXCEPTION_CLASS H5::Exception", "GENERIC_EXCEPTION_TOSTRING getCDetailMsg()"},
            include = {"H5pubconf.h", /* "H5version.h", */ "H5public.h", "H5Cpublic.h", "H5Ipublic.h",
        "H5Tpublic.h", "H5Lpublic.h", "H5Opublic.h", "H5Zpublic.h",  "H5Apublic.h", "H5ACpublic.h", "H5Dpublic.h", "H5Epublic.h", "H5Fpublic.h",
        "H5FDpublic.h", "H5Gpublic.h", "H5MMpublic.h", "H5Ppublic.h", "H5PLpublic.h", "H5Rpublic.h", "H5Spublic.h", "H5FDcore.h", "H5FDdirect.h",
        "H5FDfamily.h", "H5FDlog.h", "H5FDmpi.h", "H5FDmulti.h", "H5FDsec2.h", "H5FDstdio.h", /* "H5FDwindows.h", */ "H5DOpublic.h", "H5DSpublic.h",
        "H5LTpublic.h", "H5IMpublic.h", "H5TBpublic.h", "H5PTpublic.h", "H5LDpublic.h", "H5PacketTable.h",

        "H5Cpp.h", "H5Include.h", "H5Exception.h", "H5IdComponent.h", "H5DataSpace.h", "H5PropList.h", "H5AbstractDs.h", "H5Attribute.h",
        "H5OcreatProp.h", "H5DcreatProp.h", "H5LaccProp.h", "H5DaccProp.h", "H5LcreatProp.h", "H5Location.h", "H5Object.h", "H5CommonFG.h", "H5DataType.h", "H5DxferProp.h",
        "H5FaccProp.h", "H5FcreatProp.h", "H5AtomType.h", "H5PredType.h", "H5EnumType.h", "H5IntType.h", "H5FloatType.h", "H5StrType.h", "H5CompType.h",
        "H5ArrayType.h", "H5VarLenType.h", "H5DataSet.h", "H5Group.h", "H5File.h", "H5Library.h"},
            link = {"hdf5@.103", "hdf5_cpp@.103", "hdf5_hl@.100", "hdf5_hl_cpp@.100"}, resource = {"include", "lib"}),
    @Platform(value = "windows", link = {"zlibstatic", "libhdf5", "libhdf5_cpp", "libhdf5_hl", "libhdf5_hl_cpp"}, preload = {
        "api-ms-win-crt-locale-l1-1-0", "api-ms-win-crt-string-l1-1-0", "api-ms-win-crt-stdio-l1-1-0", "api-ms-win-crt-math-l1-1-0",
        "api-ms-win-crt-heap-l1-1-0", "api-ms-win-crt-runtime-l1-1-0", "api-ms-win-crt-convert-l1-1-0", "api-ms-win-crt-environment-l1-1-0",
        "api-ms-win-crt-time-l1-1-0", "api-ms-win-crt-filesystem-l1-1-0", "api-ms-win-crt-utility-l1-1-0", "api-ms-win-crt-multibyte-l1-1-0",
        "api-ms-win-core-string-l1-1-0", "api-ms-win-core-errorhandling-l1-1-0", "api-ms-win-core-timezone-l1-1-0", "api-ms-win-core-file-l1-1-0",
        "api-ms-win-core-namedpipe-l1-1-0", "api-ms-win-core-handle-l1-1-0", "api-ms-win-core-file-l2-1-0", "api-ms-win-core-heap-l1-1-0",
        "api-ms-win-core-libraryloader-l1-1-0", "api-ms-win-core-synch-l1-1-0", "api-ms-win-core-processthreads-l1-1-0",
        "api-ms-win-core-processenvironment-l1-1-0", "api-ms-win-core-datetime-l1-1-0", "api-ms-win-core-localization-l1-2-0",
        "api-ms-win-core-sysinfo-l1-1-0", "api-ms-win-core-synch-l1-2-0", "api-ms-win-core-console-l1-1-0", "api-ms-win-core-debug-l1-1-0",
        "api-ms-win-core-rtlsupport-l1-1-0", "api-ms-win-core-processthreads-l1-1-1", "api-ms-win-core-file-l1-2-0", "api-ms-win-core-profile-l1-1-0",
        "api-ms-win-core-memory-l1-1-0", "api-ms-win-core-util-l1-1-0", "api-ms-win-core-interlocked-l1-1-0", "ucrtbase",
        "vcruntime140", "msvcp140", "concrt140"}),
    @Platform(value = "windows-x86",    preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x86/Microsoft.VC140.CRT/",
                                                       "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x86/"}),
    @Platform(value = "windows-x86_64", preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x64/Microsoft.VC140.CRT/",
                                                       "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x64/"}) })
public class hdf5 implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("H5_DLL", "H5_DLLVAR", "H5_HLDLL", "H5_DLLCPP", "H5CHECK", "H5OPEN", "H5E_ERR_CLS", "H5E_BEGIN_TRY", "H5E_END_TRY",
                             "H5G_link_t", "H5std_string").cppTypes().annotations())
               .put(new Info("((__GNUC__ * 100) + __GNUC_MINOR__) >= 406", "H5_HAVE_PARALLEL", "NEW_HYPERSLAB_API", "H5_HAVE_DIRECT",
                             "BOOL_NOTDEFINED", "H5_NO_STD").define(false))
               .put(new Info("H5_NO_DEPRECATED_SYMBOLS", "H5_HAVE_STDBOOL_H", "H5_SIZEOF_UINT32_T>=4", "H5_SIZEOF_INT64_T>=8", "H5_SIZEOF_UINT64_T>=8").define(true))
               .put(new Info("HSIZE_UNDEF", "HADDR_UNDEF", "HADDR_AS_MPI_TYPE", "H5L_MAX_LINK_NAME_LEN", "H5L_SAME_LOC", "H5O_SHMESG_SDSPACE_FLAG",
                             "H5O_SHMESG_DTYPE_FLAG", "H5O_SHMESG_FILL_FLAG", "H5O_SHMESG_PLINE_FLAG", "H5O_SHMESG_ATTR_FLAG", "H5T_VARIABLE",
                             "H5T_NATIVE_CHAR", "H5D_CHUNK_CACHE_NSLOTS_DEFAULT", "H5D_CHUNK_CACHE_NBYTES_DEFAULT", "H5E_DEFAULT",
                             "H5F_ACC_SWMR_WRITE", "H5F_ACC_SWMR_READ", "H5F_FAMILY_DEFAULT", "H5F_UNLIMITED", "H5P_DEFAULT", "H5S_ALL").translate(false))
               .put(new Info("ssize_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("hsize_t", "hssize_t", "haddr_t", "hid_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("H5FD_FLMAP_SINGLE", "H5FD_FLMAP_DICHOTOMY", "H5FD_FLMAP_DEFAULT", "H5E_ERR_CLS_g",
                             "H5::Attribute::getName(size_t, std::string&)", "H5::FileAccPropList::getFileAccDirect", "H5::FileAccPropList::setFileAccDirect").skip())

               .put(new Info("H5::H5Location").purify())
               .put(new Info("H5::attr_operator_t").valueTypes("attr_operator_t").pointerTypes("@ByPtrPtr attr_operator_t").javaText(
                       "public static class attr_operator_t extends FunctionPointer {\n"
                     + "    static { Loader.load(); }\n"
                     + "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n"
                     + "    public    attr_operator_t(Pointer p) { super(p); }\n"
                     + "    protected attr_operator_t() { allocate(); }\n"
                     + "    private native void allocate();\n"
                     + "    public native void call( @ByRef H5Object loc/*in*/,\n"
                     + "            @Cast({\"\", \"std::string\", \"std::string&\"}) @Adapter(\"StringAdapter<char>\") BytePointer attr_name/*in*/,\n"
                     + "            Pointer operator_data/*in,out*/);\n"
                     + "}\n"))

               .put(new Info("H5::FileAccPropList::setSplit").skip())
               .put(new Info("H5::FileAccPropList::setSplit(const H5::FileAccPropList&, const H5::FileAccPropList&, const char*, const char*)").javaText(
                       "public native void setSplit(@Const @ByRef FileAccPropList meta_plist,\n"
                     + "              @Const @ByRef FileAccPropList raw_plist,\n"
                     + "              @Cast(\"const char*\") BytePointer meta_ext/*=\".meta\"*/,\n"
                     + "              @Cast(\"const char*\") BytePointer raw_ext/*=\".raw\"*/ );\n"
                     + "public native void setSplit(@Const @ByRef FileAccPropList meta_plist,\n"
                     + "              @Const @ByRef FileAccPropList raw_plist,\n"
                     + "              String meta_ext/*=\".meta\"*/,\n"
                     + "              String raw_ext/*=\".raw\"*/ );\n"));
    }
}
