package org.bytedeco.cudf.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Properties(
    target = "org.bytedeco.cudf",
    global = "org.bytedeco.cudf.global.cudf",
    value = {
        @Platform(
            value = {"linux-x86_64"},
            define = {"NDEBUG", "UNIQUE_PTR_NAMESPACE std", "SHARED_PTR_NAMESPACE std"},
            include = {
         // "<cudf/types.h>",
// 			"<cudf/wrappers/timestamps.hpp>",
// 			"<cudf/wrappers/bool.hpp>",
// 			"<cudf/replace.hpp>",
// 			"<cudf/scalar/scalar_device_view.cuh>",
			// "<cudf/scalar/scalar.hpp>",
// 			"<cudf/scalar/scalar_factories.hpp>",
// 			"<cudf/copying.hpp>",
// 			"<cudf/quantiles.hpp>",
// 			"<cudf/transpose.hpp>",
// 			"<cudf/cudf.h>",
// 			"<cudf/strings/combine.hpp>",
// 			"<cudf/strings/replace.hpp>",
// 			"<cudf/strings/replace_re.hpp>",
// 			"<cudf/strings/copying.hpp>",
// 			"<cudf/strings/char_types/char_types.hpp>",
// 			"<cudf/strings/strings_column_view.hpp>",
// 			"<cudf/strings/strip.hpp>",
// 			"<cudf/strings/translate.hpp>",
// 			"<cudf/strings/find_multiple.hpp>",
// 			"<cudf/strings/substring.hpp>",
// 			"<cudf/strings/contains.hpp>",
//			"<cudf/strings/string_view.cuh>",
// 			"<cudf/strings/detail/fill.hpp>",
// 			"<cudf/strings/detail/utilities.cuh>",
// 			"<cudf/strings/detail/merge.cuh>",
// 			"<cudf/strings/detail/gather.cuh>",
// 			"<cudf/strings/detail/utilities.hpp>",
// 			"<cudf/strings/detail/copy_if_else.cuh>",
// 			"<cudf/strings/detail/copy_range.cuh>",
// 			"<cudf/strings/detail/concatenate.hpp>",
// 			"<cudf/strings/detail/scatter.cuh>",
// 			"<cudf/strings/findall.hpp>",
// 			"<cudf/strings/padding.hpp>",
// 			"<cudf/strings/split/partition.hpp>",
// 			"<cudf/strings/split/split.hpp>",
// 			"<cudf/strings/sorting.hpp>",
// 			"<cudf/strings/extract.hpp>",
// 			"<cudf/strings/find.hpp>",
// 			"<cudf/strings/case.hpp>",
// 			"<cudf/strings/attributes.hpp>",
// 			"<cudf/strings/convert/convert_floats.hpp>",
// 			"<cudf/strings/convert/convert_booleans.hpp>",
// 			"<cudf/strings/convert/convert_integers.hpp>",
// 			"<cudf/strings/convert/convert_urls.hpp>",
// 			"<cudf/strings/convert/convert_datetime.hpp>",
// 			"<cudf/strings/convert/convert_ipv4.hpp>",
// 			"<cudf/groupby.hpp>",
// 			"<cudf/dlpack.hpp>",
// 			"<cudf/detail/fill.hpp>",
// 			"<cudf/detail/replace.hpp>",
//			"<cudf/detail/merge.cuh>",
//			"<cudf/detail/gather.cuh>",
// 			"<cudf/detail/copy_if.cuh>",
// 			"<cudf/detail/copy.hpp>",
// 			"<cudf/detail/transpose.hpp>",
// 			"<cudf/detail/groupby.hpp>",
// 			"<cudf/detail/dlpack.hpp>",
// 			"<cudf/detail/gather.hpp>",
// 			"<cudf/detail/unary.hpp>",
// 			"<cudf/detail/groupby/sort_helper.hpp>",
//			"<cudf/detail/valid_if.cuh>",
// 			"<cudf/detail/reduction.cuh>",
// 			"<cudf/detail/transform.hpp>",
// 			"<cudf/detail/repeat.hpp>",
// 			"<cudf/detail/stream_compaction.hpp>",
//			"<cudf/detail/binaryop.hpp>",
//			"<cudf/detail/copy_if_else.cuh>",
// 			"<cudf/detail/aggregation/aggregation.cuh>",
// 			"<cudf/detail/aggregation/aggregation.hpp>",
// 			"<cudf/detail/copy_range.cuh>",
// 			"<cudf/detail/search.hpp>",
// 			"<cudf/detail/null_mask.hpp>",
// 			"<cudf/detail/hashing.hpp>",
// 			"<cudf/detail/scatter.hpp>",
// 			"<cudf/detail/sorting.hpp>",
//			"<cudf/detail/reduction_operators.cuh>",
// 			"<cudf/detail/reduction_functions.hpp>",
// 			"<cudf/detail/scatter.cuh>",
// 			"<cudf/detail/iterator.cuh>",
// 			"<cudf/detail/utilities/transform_unary_functions.cuh>",
// 			"<cudf/detail/utilities/trie.cuh>",
// 			"<cudf/detail/utilities/integer_utils.hpp>",
// 			"<cudf/detail/utilities/device_operators.cuh>",
// 			"<cudf/detail/utilities/release_assert.cuh>",
// 			"<cudf/detail/utilities/cuda.cuh>",
//			"<cudf/detail/utilities/hash_functions.cuh>",
// 			"<cudf/detail/utilities/int_fastdiv.h>",
// 			"<cudf/detail/utilities/device_atomics.cuh>",
 			"<cudf/unary.hpp>",
			"<cudf/types.hpp>",
// 			"<cudf/round_robin.hpp>",
// 			"<cudf/transform.hpp>",
// 			"<cudf/reshape.hpp>",
// 			"<cudf/rolling.hpp>",
// 			"<cudf/stream_compaction.hpp>",
// 			"<cudf/binaryop.hpp>",
// 			"<cudf/aggregation.hpp>",
// 			"<cudf/reduction.hpp>",
// 			"<cudf/types.h>",
// 			"<cudf/merge.hpp>",
// 			"<cudf/search.hpp>",
// 			"<cudf/null_mask.hpp>",
// 			"<cudf/hashing.hpp>",
//			"<cudf/legacy/io_functions.hpp>",
//			"<cudf/legacy/replace.hpp>",
//			"<cudf/legacy/copying.hpp>",
//			"<cudf/legacy/quantiles.hpp>",
//			"<cudf/legacy/groupby.hpp>",
//			"<cudf/legacy/functions.h>",
//			"<cudf/legacy/bitmask.hpp>",
//			"<cudf/legacy/io_functions.h>",
//			"<cudf/legacy/predicates.hpp>",
//			"<cudf/legacy/unary.hpp>",
//			"<cudf/legacy/transform.hpp>",
//			"<cudf/legacy/reshape.hpp>",
//			"<cudf/legacy/io_types.hpp>",
//			"<cudf/legacy/interop.hpp>",
//			"<cudf/legacy/rolling.hpp>",
//			"<cudf/legacy/table.hpp>",
//			"<cudf/legacy/stream_compaction.hpp>",
//			"<cudf/legacy/column.hpp>",
//			"<cudf/legacy/binaryop.hpp>",
//			"<cudf/legacy/reduction.hpp>",
//			"<cudf/legacy/merge.hpp>",
//			"<cudf/legacy/search.hpp>",
//			"<cudf/legacy/io_readers.hpp>",
//			"<cudf/legacy/datetime.hpp>",
//			"<cudf/legacy/io_writers.hpp>",
//			"<cudf/legacy/join.hpp>",
//			"<cudf/legacy/filling.hpp>",
//			"<cudf/legacy/io_types.h>",
// 			"<cudf/datetime.hpp>",
// 			"<cudf/ipc.hpp>",
// 			"<cudf/sorting.hpp>",
//			"<cudf/table/row_operators.cuh>",
// 			"<cudf/table/table.hpp>",
//			"<cudf/table/table_device_view.cuh>",
// 			"<cudf/table/table_view.hpp>",
			// "<cudf/column/column_factories.hpp>",
			"<cudf/column/column.hpp>",
//			"<cudf/column/column_device_view.cuh>",
			"<cudf/column/column_view.hpp>",
			// "<cudf/join.hpp>",
			// "<cudf/filling.hpp>",
			// "<cudf/io/writers.hpp>",
			// "<cudf/io/functions.hpp>",
			// "<cudf/io/types.hpp>",
			// "<cudf/io/readers.hpp>",
			// "<cudf/convert_types.h>",
			// "<cudf/utilities/nvtx_utils.hpp>",
			// "<cudf/utilities/bit.hpp>",
			// "<cudf/utilities/traits.hpp>",
			// "<cudf/utilities/error.hpp>",
//			"<cudf/utilities/legacy/wrapper_types.hpp>",
			// "<cudf/utilities/legacy/nvcategory_util.hpp>",
			// "<cudf/utilities/legacy/type_dispatcher.hpp>",
			// "<cudf/utilities/type_dispatcher.hpp>"
		},
	    compiler = "cpp11",
	    includepath = {"/usr/local/cuda/include/"},
            link = "cudf"
        )
    }
)
public class cudf implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "cudf"); }

    @Override
    public void map(InfoMap infoMap) {
        infoMap
        .put(new Info().enumerate())
        .put(new Info("std::unique_ptr<cudf::column>").valueTypes("@MoveUniquePtr column").pointerTypes("@UniquePtr column"))
        .put(new Info("CUstream_st").pointerTypes("org.bytedeco.cuda.cudart.CUstream_st").define())
        .put(new Info("std::vector<std::unique_ptr<cudf::column> >").skip())
        .put(new Info("CUDA_HOST_DEVICE_CALLABLE").cppTypes().annotations())
        .put(new Info("CUDA_DEVICE_CALLABLE").cppTypes().annotations())
        .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("SizeTPointer"));

        String qualifier = "cudf::detail::column_view_base::";
        String[] functionTemplates = {"head","data", "begin", "end"};

        for (String function: functionTemplates) {
            infoMap
            .put(new Info(qualifier + function + "<bool>").javaNames(function + "Boolean"))
            .put(new Info(qualifier + function + "<int32_t>").javaNames(function + "Int"))
            .put(new Info(qualifier + function + "<int64_t>").javaNames(function + "Long"))
            .put(new Info(qualifier + function + "<float>").javaNames(function + "Float"))
            .put(new Info(qualifier + function + "<double>").javaNames(function + "Double"));
        }

        ;
    }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"std::unique_ptr", "&&"}) @Adapter("UniquePtrAdapter")
    public @interface MoveUniquePtr {
        String value() default "";
    }
}
