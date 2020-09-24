package org.bytedeco.libecl.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        target = "org.bytedeco.libecl",
        global = "org.bytedeco.libecl.global.libecl",
        value = {
            @Platform(
                    value = {
                        "linux-x86",
                        "linux-x86_64",
                        "macosx-x86_64",
                        "windows-x86",
                        "windows-x86_64"
                    },
                    includepath = { /* XXX: added at runtime "$INSTALL_PATH/libecl-$LIBECL_VERSION/lib/private-include" */},
                    include = {
                        //<editor-fold defaultstate="collapsed" desc=" ... ">
                        // "wchar.h",
                        // "vector",
                        // "util_unlink.h",
                        // "unistd.h",
                        // "type_traits",
                        // "time.h",
                        // "sys/types.h",
                        // "sys/stat.h",
                        // "string.h",
                        // "string",
                        // "stdlib.h",
                        // "stdio.h",
                        // "stdint.h",
                        // "stdexcept",
                        // "stdbool.h",
                        // "stdarg.h",
                        // "setjmp.h",
                        // "memory",
                        // "math.h",
                        // "limits.h",
                        // "fstream",
                        // "algorithm",
                        "ert/util/vector_util.hpp",
                        "ert/util/node_data.hpp",
                        "ert/util/ert_api_config.h",
                        "ert/util/ert_api_config.hpp",
                        "ert/util/util.h",
                        "ert/util/type_macros.hpp",
                        "ert/util/perm_vector.hpp",
                        "ert/util/perm_vector.h",
                        "ert/util/type_macros.h",
                        "ert/util/int_vector.h",
                        "ert/util/int_vector.hpp",
                        "ert/util/vector.hpp",
                        "ert/util/vector.h",
                        "ert/util/util_unlink.h",
                        "ert/util/util_endian.h",
                        "ert/util/util.hpp",
                        "ert/util/double_vector.h",
                        "ert/util/double_vector.hpp",
                        "ert/util/bool_vector.h",
                        "ert/util/bool_vector.hpp",
                        "ert/util/type_vector_functions.hpp",
                        "ert/util/type_vector_functions.h",
                        "ert/util/timer.hpp",
                        "ert/util/timer.h",
                        "ert/util/time_t_vector.h",
                        "ert/util/time_t_vector.hpp",
                        "ert/util/time_interval.hpp",
                        "ert/util/thread_pool1.h",
                        "ert/util/test_work_area.hpp",
                        "ert/util/test_work_area.h",
                        "ert/util/test_util.hpp",
                        "ert/util/test_util.h",
                        "ert/util/stringlist.hpp",
                        "ert/util/stringlist.h",
                        "ert/util/string_util.hpp",
                        "ert/util/string_util.h",
                        "ert/util/statistics.hpp",
                        "ert/util/statistics.h",
                        "ert/util/ssize_t.hpp",
                        "ert/util/ssize_t.h",
                        "ert/util/size_t_vector.h",
                        "ert/util/size_t_vector.hpp",
                        "ert/util/rng.hpp",
                        "ert/util/rng.h",
                        "ert/util/path_stack.hpp",
                        "ert/util/path_stack.h",
                        "ert/util/parser.hpp",
                        "ert/util/parser.h",
                        "ert/util/node_data.h",
                        "ert/util/node_ctype.hpp",
                        "ert/util/node_ctype.h",
                        "ert/util/mzran.hpp",
                        "ert/util/mzran.h",
                        "ert/util/msvc_stdbool.h",
                        "ert/util/lookup_table.hpp",
                        "ert/util/lookup_table.h",
                        "ert/util/long_vector.h",
                        "ert/util/long_vector.hpp",
                        "ert/util/hash_node.hpp",
                        "ert/util/hash_sll.hpp",
                        "ert/util/hash_sll.h",
                        "ert/util/hash_node.h",
                        "ert/util/hash.hpp",
                        "ert/util/hash.h",
                        "ert/util/float_vector.h",
                        "ert/util/float_vector.hpp",
                        "ert/util/ert_unique_ptr.hpp",
                        "ert/util/ecl_version.hpp",
                        "ert/util/ecl_version.h",
                        "ert/util/build_config.h",
                        "ert/util/build_config.hpp",
                        //"ert/util/buffer_string.h",
                        // "buffer_string.h",
                        "ert/util/buffer.hpp",
                        "ert/util/buffer.h",
                        "ert/geometry/geo_util.hpp",
                        "ert/geometry/geo_util.h",
                        "ert/geometry/geo_pointset.hpp",
                        "ert/geometry/geo_surface.hpp",
                        "ert/geometry/geo_surface.h",
                        "ert/geometry/geo_polygon.hpp",
                        "ert/geometry/geo_region.hpp",
                        "ert/geometry/geo_region.h",
                        "ert/geometry/geo_polygon_collection.hpp",
                        "ert/geometry/geo_polygon_collection.h",
                        "ert/geometry/geo_polygon.h",
                        "ert/geometry/geo_pointset.h",
                        "ert/ecl/ecl_kw_grdecl.hpp",
                        "ert/ecl/ecl_type.hpp",
                        "ert/ecl/ecl_util.hpp",
                        "ert/ecl/fortio.h",
                        "ert/ecl/ecl_kw.hpp",
                        "ert/ecl/ecl_file_kw.hpp",
                        "ert/ecl/ecl_file_view.hpp",
                        "ert/ecl/ecl_file.hpp",
                        "ert/ecl/ecl_rsthead.hpp",
                        "ert/ecl_well/well_conn.hpp",
                        "ert/ecl_well/well_conn_collection.hpp",
                        "ert/ecl_well/well_const.hpp",
                        "ert/ecl_well/well_rseg_loader.hpp",
                        "ert/ecl_well/well_segment.hpp",
                        "ert/ecl_well/well_branch_collection.hpp",
                        "ert/ecl_well/well_segment_collection.hpp",
                        "ert/ecl/ecl_coarse_cell.hpp",
                        "ert/ecl/nnc_vector.hpp",
                        "ert/ecl/nnc_info.hpp",
                        "ert/ecl/grid_dims.hpp",
                        "ert/ecl/ecl_grid.hpp",
                        "ert/ecl_well/well_state.hpp",
                        "ert/ecl_well/well_ts.hpp",
                        "ert/ecl_well/well_ts.h",
                        "ert/ecl_well/well_state.h",
                        "ert/ecl_well/well_segment_collection.h",
                        "ert/ecl_well/well_segment.h",
                        "ert/ecl_well/well_rseg_loader.h",
                        "ert/ecl_well/well_info.hpp",
                        "ert/ecl_well/well_info.h",
                        "ert/ecl_well/well_const.h",
                        "ert/ecl_well/well_conn_collection.h",
                        "ert/ecl_well/well_conn.h",
                        "ert/ecl_well/well_branch_collection.h",
                        "ert/ecl/smspec_node.h",
                        // "array",
                        "ert/ecl/smspec_node.hpp",
                        "ert/ecl/nnc_vector.h",
                        "ert/ecl/nnc_info.h",
                        "ert/ecl/layer.hpp",
                        "ert/ecl/layer.h",
                        "ert/ecl/grid_dims.h",
                        "ert/ecl/fault_block.hpp",
                        "ert/ecl/fault_block_layer.hpp",
                        "ert/ecl/fault_block_layer.h",
                        "ert/ecl/fault_block.h",
                        "ert/ecl/ecl_util.h",
                        "ert/ecl/ecl_units.hpp",
                        "ert/ecl/ecl_units.h",
                        "ert/ecl/ecl_type.h",
                        "ert/ecl/ecl_smspec.hpp",
                        "ert/ecl/ecl_sum_tstep.hpp",
                        "ert/ecl/ecl_sum.hpp",
                        "ert/ecl/ecl_sum_vector.hpp",
                        "ert/ecl/ecl_sum_vector.h",
                        "ert/ecl/ecl_sum_tstep.h",
                        "ert/ecl/ecl_sum_index.hpp",
                        "ert/ecl/ecl_sum_index.h",
                        "ert/ecl/ecl_sum_data.hpp",
                        "ert/ecl/ecl_sum_data.h",
                        "ert/ecl/ecl_sum.h",
                        "ert/ecl/ecl_region.hpp",
                        "ert/ecl/ecl_subsidence.hpp",
                        "ert/ecl/ecl_subsidence.h",
                        "ert/ecl/ecl_smspec.h",
                        "ert/ecl/ecl_rsthead.h",
                        "ert/ecl/ecl_rst_file.hpp",
                        "ert/ecl/ecl_rst_file.h",
                        "ert/ecl/ecl_rft_cell.hpp",
                        "ert/ecl/ecl_rft_node.hpp",
                        "ert/ecl/ecl_rft_node.h",
                        "ert/ecl/ecl_rft_file.hpp",
                        "ert/ecl/ecl_rft_file.h",
                        "ert/ecl/ecl_rft_cell.h",
                        "ert/ecl/ecl_region.h",
                        "ert/ecl/ecl_nnc_geometry.hpp",
                        "ert/ecl/ecl_nnc_geometry.h",
                        "ert/ecl/ecl_nnc_export.hpp",
                        "ert/ecl/ecl_nnc_export.h",
                        "ert/ecl/ecl_nnc_data.hpp",
                        "ert/ecl/ecl_nnc_data.h",
                        "ert/ecl/ecl_kw_magic.hpp",
                        "ert/ecl/ecl_kw_magic.h",
                        "ert/ecl/ecl_kw_grdecl.h",
                        "ert/ecl/ecl_kw.h",
                        "ert/ecl/ecl_io_config.hpp",
                        "ert/ecl/ecl_io_config.h",
                        "ert/ecl/ecl_init_file.hpp",
                        "ert/ecl/ecl_init_file.h",
                        "ert/ecl/ecl_grid_dims.hpp",
                        "ert/ecl/ecl_grid_dims.h",
                        "ert/ecl/ecl_grid.h",
                        "detail/ecl/ecl_grid_cache.hpp",
                        "ert/ecl/ecl_grav_common.hpp",
                        "ert/ecl/ecl_grav_common.h",
                        "ert/ecl/ecl_file.h",
                        "ert/ecl/ecl_grav_calc.hpp",
                        "ert/ecl/ecl_grav_calc.h",
                        "ert/ecl/ecl_grav.hpp",
                        "ert/ecl/ecl_grav.h",
                        "ert/ecl/ecl_file_view.h",
                        "ert/ecl/ecl_file_kw.h",
                        "ert/ecl/ecl_endian_flip.hpp",
                        "ert/ecl/ecl_endian_flip.h",
                        "ert/ecl/ecl_coarse_cell.h",
                        "ert/ecl/ecl_box.hpp",
                        "ert/ecl/FortIO.hpp",
                        "ert/ecl/EclKW.hpp",
                        "ert/ecl/EclFilename.hpp"
                    //</editor-fold>
                    },
                    link = "ecl"
            )
        }
)
public class libecl implements InfoMapper, LoadEnabled {

    static {
        Loader.checkVersion("org.bytedeco", "libecl");
    }

    @Override
    public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> includePaths = properties.get("platform.includepath");
        if (!includePaths.isEmpty()) {
            String include = includePaths.get(0);
            if (include.endsWith("/" + platform + "/include/")) {
                String privateInclude = include.replace("/include/", "/libecl-2.9.1/lib/private-include/");
                if (!includePaths.contains(privateInclude)) {
                    includePaths.add(privateInclude);
                }
            }
        }
//        System.out.println("XXXXX includePaths = ");
//        includePaths.stream().forEach(System.out::println);
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("true").javaNames("TRUE").javaText("public static final int TRUE = 1;"));
        infoMap.put(new Info("false").javaNames("FALSE").javaText("public static final int FALSE = 0;"));
        infoMap.put(new Info("ECL_CHAR").cppTypes("int"));
        infoMap.put(new Info("std::array<int,3>").cast().pointerTypes("IntPointer", "IntBuffer", "int[]"));
        infoMap.put(new Info("tm").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("jmp_buf").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::ios_base::openmode").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"));
        infoMap.put(new Info("INT_MAX").javaNames("Integer.MAX_VALUE"));
        infoMap.put(new Info("HUGE_VAL").javaNames("Integer.MAX_VALUE"));
        infoMap.put(new Info("ECL_UNITS_CUBIC").cppTypes("double", "double"));
        infoMap.put(new Info("ECL_UNITS_MILLI").cppTypes("double", "double"));
        infoMap.put(new Info("ECL_UNITS_MEGA").cppTypes("double", "double"));
    }
}
