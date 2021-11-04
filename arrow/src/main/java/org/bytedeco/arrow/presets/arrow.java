/*
 * Copyright (C) 2019-2021 Samuel Audet
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

package org.bytedeco.arrow.presets;

import org.bytedeco.javacpp.Loader;
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
    names = {"linux", "macosx", "windows"},
    value = {
        @Platform(
            compiler = "cpp11",
            define = {"NO_WINDOWS_H", "UNIQUE_PTR_NAMESPACE std", "SHARED_PTR_NAMESPACE std"},
            include = {
                "arrow/api.h",
                "arrow/util/config.h",
                "arrow/util/checked_cast.h",
                "arrow/util/macros.h",
                "arrow/util/compare.h",
                "arrow/util/type_fwd.h",
                "arrow/util/type_traits.h",
                "arrow/util/visibility.h",
                "arrow/util/compression.h",
                "arrow/util/functional.h",
                "arrow/util/iterator.h",
                "arrow/util/memory.h",
                "arrow/util/variant.h",
                "arrow/util/bit_util.h",
                "arrow/util/ubsan.h",
                "arrow/util/key_value_metadata.h",
                "arrow/util/string_builder.h",
//                "arrow/util/string_view.h",
//                "arrow/vendored/string_view.hpp",
//                "arrow/vendored/variant.hpp",
                "arrow/status.h",
                "arrow/memory_pool.h",
                "arrow/buffer.h",
                "arrow/buffer_builder.h",
                "arrow/compare.h",
                "arrow/result.h",
                "arrow/type_fwd.h",
                "arrow/type_traits.h",
                "arrow/util/basic_decimal.h",
                "arrow/util/decimal.h",
                "arrow/util/sort.h",
                "arrow/util/future.h",
                "arrow/util/cancel.h",
                "arrow/util/task_group.h",
                "arrow/util/thread_pool.h",
                "arrow/util/async_util.h",
                "arrow/util/async_generator.h",
//                "arrow/util/value_parsing.h",
                "arrow/type.h",
                "arrow/scalar.h",
                "arrow/visitor.h",
                "arrow/array.h",
                "arrow/array/data.h",
                "arrow/array/util.h",
                "arrow/array/array_base.h",
                "arrow/array/array_dict.h",
                "arrow/array/array_nested.h",
                "arrow/array/array_primitive.h",
                "arrow/array/concatenate.h",
                "arrow/array/builder_base.h",
//                "arrow/array/builder_adaptive.h",
                "arrow/array/builder_binary.h",
                "arrow/array/builder_decimal.h",
                "arrow/array/builder_dict.h",
                "arrow/array/builder_nested.h",
                "arrow/array/builder_primitive.h",
                "arrow/array/builder_time.h",
                "arrow/array/builder_union.h",
                "arrow/chunked_array.h",
                "arrow/config.h",
                "arrow/datum.h",
                "arrow/builder.h",
                "arrow/extension_type.h",
                "arrow/pretty_print.h",
                "arrow/record_batch.h",
                "arrow/table.h",
                "arrow/table_builder.h",
                "arrow/tensor.h",
                "arrow/io/api.h",
                "arrow/io/caching.h",
                "arrow/io/type_fwd.h",
                "arrow/io/interfaces.h",
                "arrow/io/concurrency.h",
                "arrow/io/buffered.h",
                "arrow/io/compressed.h",
                "arrow/io/file.h",
                "arrow/io/hdfs.h",
                "arrow/io/memory.h",
                "arrow/io/slow.h",
                "arrow/filesystem/api.h",
                "arrow/filesystem/type_fwd.h",
                "arrow/filesystem/filesystem.h",
                "arrow/filesystem/hdfs.h",
                "arrow/filesystem/localfs.h",
                "arrow/filesystem/mockfs.h",
//                "arrow/filesystem/path_forest.h",
//                "arrow/filesystem/s3fs.h",
                "arrow/csv/api.h",
                "arrow/csv/options.h",
                "arrow/csv/reader.h",
                "arrow/json/options.h",
                "arrow/json/reader.h",
                "arrow/compute/api.h",
                "arrow/compute/api_aggregate.h",
                "arrow/compute/api_scalar.h",
                "arrow/compute/api_vector.h",
                "arrow/compute/kernel.h",
                "arrow/compute/type_fwd.h",
                "arrow/compute/exec/expression.h",
                "arrow/compute/exec.h",
                "arrow/compute/function.h",
                "arrow/compute/cast.h",
                "arrow/compute/registry.h",
                "arrow/ipc/api.h",
                "arrow/ipc/type_fwd.h",
                "arrow/ipc/dictionary.h",
                "arrow/ipc/feather.h",
                "arrow/ipc/json_simple.h",
                "arrow/ipc/options.h",
                "arrow/ipc/message.h",
                "arrow/ipc/reader.h",
                "arrow/ipc/writer.h",
            },
            link = "arrow@.500"
        ),
    },
    target = "org.bytedeco.arrow",
    global = "org.bytedeco.arrow.global.arrow"
)
public class arrow implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "arrow"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().enumerate().javaText("import org.bytedeco.arrow.Function;"))
               .put(new Info("defined(__cplusplus)").define())
               .put(new Info("__cplusplus_cli", "ARROW_EXTRA_ERROR_CONTEXT").define(false))
               .put(new Info("ARROW_NORETURN", "ARROW_MUST_USE_RESULT", "NULLPTR", "ARROW_EXPORT", "ARROW_FORCE_INLINE",
                             "ARROW_MEMORY_POOL_DEFAULT", "ARROW_BYTE_SWAP64", "ARROW_BYTE_SWAP32", "ARROW_MUST_USE_TYPE",
                             "ARROW_NOINLINE", "ARROW_POPCOUNT64", "ARROW_POPCOUNT32", "ARROW_RESTRICT",
                             "ARROW_SUPPRESS_DEPRECATION_WARNING", "ARROW_UNSUPPRESS_DEPRECATION_WARNING").cppTypes().annotations())
               .put(new Info("ARROW_BITNESS", "ARROW_LITTLE_ENDIAN").translate(false))

               .put(new Info("ARROW_DEPRECATED").cppText("#define ARROW_DEPRECATED(...) deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("basic/containers").cppTypes("arrow::util::optional"))

               .put(new Info("arrow::internal::BitsetStack::reference").pointerTypes("BoolPointer"))
               .put(new Info("arrow::util::bytes_view", "arrow::util::string_view", "arrow::internal::Bitmap", "std::atomic<int64_t>", "std::initializer_list").skip())
               .put(new Info("arrow::Buffer").pointerTypes("ArrowBuffer"))
               .put(new Info("arrow::EqualOptions::nans_equal", "arrow::EqualOptions::atol", "arrow::EqualOptions::diff_sink").annotations("@org.bytedeco.javacpp.annotation.Function"))
               .put(new Info("arrow::detail::CTypeImpl", "arrow::detail::IntegerTypeImpl", "arrow::internal::IsOneOf", "arrow::util::internal::non_null_filler", "arrow::TypeTraits",
                             "arrow::detail::CTypeImpl<DERIVED,BASE,TYPE_ID,C_TYPE>::type_id", "arrow::detail::Empty", "arrow::detail::is_future", "arrow::util::detail::all",
                             "arrow::util::detail::delete_copy_constructor", "arrow::internal::max_size_traits", "arrow::internal::max_size", "arrow::GetPhysicalType",
                             "arrow::BasicDecimal128::LittleEndianArrayTag", "arrow::BasicDecimal256::LittleEndianArrayTag",
                             "arrow::Decimal128::ToRealConversion", "arrow::Decimal256::ToRealConversion", "arrow::internal::FnOnce", "arrow::compute::internal::Grouper",
                             "arrow::internal::Empty", "arrow::FutureToSync", "arrow::AsyncGenerator", "arrow::ipc::RecordBatchFileReader::GetRecordBatchGenerator",
                             "arrow::compute::Kernel::InitAll", "arrow::compute::Expression::Call::ComputeHash").skip())

               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("const char").valueTypes("byte").pointerTypes("String", "@Cast(\"const char*\") BytePointer"))
               .put(new Info("std::string").annotations("@StdString").valueTypes("String", "BytePointer").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::array<uint8_t,16>").pointerTypes("Byte16Array").define())
               .put(new Info("std::array<uint8_t,32>").pointerTypes("Byte32Array").define())
               .put(new Info("std::array<uint64_t,4>").pointerTypes("Long4Array").define())
               .put(new Info("std::array<uint64_t,2>").pointerTypes("Long2Array").define())
               .put(new Info("std::pair<arrow::Decimal128,arrow::Decimal128>").pointerTypes("Decimal128Pair").define())
               .put(new Info("std::pair<arrow::Decimal256,arrow::Decimal256>").pointerTypes("Decimal256Pair").define())
               .put(new Info("std::pair<std::string,std::string>").pointerTypes("StringPair").define())
               .put(new Info("std::vector<bool>").pointerTypes("BoolVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<std::pair<std::string,std::string> >").pointerTypes("StringStringPairVector").define())
               .put(new Info("std::unordered_map<std::string,std::string>").pointerTypes("StringStringMap").define())
               .put(new Info("std::vector<int>::const_iterator").cast().pointerTypes("IntPointer"))
               .put(new Info("arrow::Type::type").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))

               .put(new Info("arrow::NumericArray<arrow::Int8Type>::value_type",
                             "arrow::NumericArray<arrow::UInt8Type>::value_type",
                             "arrow::UnionArray::type_code_t", "arrow::DenseUnionArray::type_code_t", "arrow::SparseUnionArray::type_code_t",
                             "arrow::NumericScalar<arrow::Int8Type>::ValueType",
                             "arrow::NumericScalar<arrow::UInt8Type>::ValueType",
                             "arrow::NumericBuilder<arrow::Int8Type>::value_type",
                             "arrow::NumericBuilder<arrow::UInt8Type>::value_type").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("arrow::NumericBuilder<arrow::Int8Type>::ArrayType",
                             "arrow::NumericBuilder<arrow::UInt8Type>::ArrayType").cast().valueTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("arrow::NumericArray<arrow::Int16Type>::value_type",
                             "arrow::NumericArray<arrow::UInt16Type>::value_type",
                             "arrow::NumericScalar<arrow::Int16Type>::ValueType",
                             "arrow::NumericScalar<arrow::UInt16Type>::ValueType",
                             "arrow::NumericBuilder<arrow::Int16Type>::value_type",
                             "arrow::NumericBuilder<arrow::UInt16Type>::value_type").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("arrow::NumericBuilder<arrow::Int16Type>::ArrayType",
                             "arrow::NumericBuilder<arrow::UInt16Type>::ArrayType").cast().valueTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("arrow::NumericArray<arrow::Int32Type>::value_type",
                             "arrow::NumericArray<arrow::UInt32Type>::value_type",
                             "arrow::NumericScalar<arrow::Int32Type>::ValueType",
                             "arrow::NumericScalar<arrow::UInt32Type>::ValueType",
                             "arrow::DateScalar<arrow::Date32Type>::ValueType",
                             "arrow::TemporalScalar<arrow::Date32Type>::ValueType",
                             "arrow::TemporalScalar<arrow::Time32Type>::ValueType",
                             "arrow::TemporalScalar<arrow::MonthIntervalType>::ValueType",
                             "arrow::NumericArray<arrow::Date32Type>::value_type",
                             "arrow::NumericArray<arrow::Time32Type>::value_type",
                             "arrow::NumericArray<arrow::MonthIntervalType>::value_type",
                             "arrow::BaseListArray<arrow::ListType>::offset_type",
                             "arrow::BaseBinaryArray<arrow::BinaryType>::offset_type",
                             "arrow::BaseListBuilder<arrow::ListType>::offset_type",
                             "arrow::BaseBinaryBuilder<arrow::BinaryType>::offset_type",
                             "arrow::NumericBuilder<arrow::Int32Type>::value_type",
                             "arrow::NumericBuilder<arrow::UInt32Type>::value_type").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("arrow::NumericBuilder<arrow::Int32Type>::ArrayType",
                             "arrow::NumericBuilder<arrow::UInt32Type>::ArrayType").cast().valueTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("arrow::NumericArray<arrow::Int64Type>::value_type",
                             "arrow::NumericArray<arrow::UInt64Type>::value_type",
                             "arrow::DateScalar<arrow::Date64Type>::ValueType",
                             "arrow::NumericScalar<arrow::Int64Type>::ValueType",
                             "arrow::NumericScalar<arrow::UInt64Type>::ValueType",
                             "arrow::TemporalScalar<arrow::Date64Type>::ValueType",
                             "arrow::TemporalScalar<arrow::DurationType>::ValueType",
                             "arrow::TemporalScalar<arrow::Time64Type>::ValueType",
                             "arrow::TemporalScalar<arrow::TimestampType>::ValueType",
                             "arrow::NumericArray<arrow::Date64Type>::value_type",
                             "arrow::NumericArray<arrow::Time64Type>::value_type",
                             "arrow::NumericArray<arrow::DayTimeIntervalType>::value_type",
                             "arrow::NumericArray<arrow::MonthDayNanoIntervalType>::value_type",
                             "arrow::NumericArray<arrow::DurationType>::value_type",
                             "arrow::NumericArray<arrow::TimestampType>::value_type",
                             "arrow::BaseListArray<arrow::LargeListType>::offset_type",
                             "arrow::BaseBinaryArray<arrow::LargeBinaryType>::offset_type",
                             "arrow::BaseListBuilder<arrow::LargeListType>::offset_type",
                             "arrow::BaseBinaryBuilder<arrow::LargeBinaryType>::offset_type",
                             "arrow::NumericBuilder<arrow::Int64Type>::value_type",
                             "arrow::NumericBuilder<arrow::UInt64Type>::value_type").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("arrow::NumericBuilder<arrow::Int64Type>::ArrayType",
                             "arrow::NumericBuilder<arrow::UInt64Type>::ArrayType",
                             "arrow::NumericBuilder<arrow::DayTimeIntervalType>::ArrayType",
                             "arrow::NumericBuilder<arrow::MonthDayNanoIntervalType>::ArrayType").cast().valueTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("arrow::NumericArray<arrow::HalfFloatType>::value_type", 
                             "arrow::NumericScalar<arrow::HalfFloatType>::ValueType",
                             "arrow::NumericBuilder<arrow::HalfFloatType>::value_type",
                             "arrow::NumericBuilder<arrow::HalfFloatType>::value_type").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("arrow::NumericBuilder<arrow::HalfFloatType>::ArrayType",
                             "arrow::NumericBuilder<arrow::HalfFloatType>::ArrayType").cast().valueTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("arrow::NumericArray<arrow::FloatType>::value_type",
                             "arrow::NumericScalar<arrow::FloatType>::ValueType",
                             "arrow::NumericBuilder<arrow::FloatType>::value_type",
                             "arrow::NumericBuilder<arrow::FloatType>::value_type").cast().valueTypes("float").pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("arrow::NumericBuilder<arrow::FloatType>::ArrayType",
                             "arrow::NumericBuilder<arrow::FloatType>::ArrayType").cast().valueTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("arrow::NumericArray<arrow::DoubleType>::value_type",
                             "arrow::NumericScalar<arrow::DoubleType>::ValueType",
                             "arrow::NumericBuilder<arrow::DoubleType>::value_type",
                             "arrow::NumericBuilder<arrow::DoubleType>::value_type").cast().valueTypes("double").pointerTypes("DoublePointer", "DoubleBuffer", "double[]"))
               .put(new Info("arrow::NumericBuilder<arrow::DoubleType>::ArrayType",
                             "arrow::NumericBuilder<arrow::DoubleType>::ArrayType").cast().valueTypes("DoublePointer", "DoubleBuffer", "double[]"))

               .put(new Info("arrow::DayTimeIntervalArray::TypeClass::DayMilliseconds",
                             "arrow::TemporalScalar<arrow::DayTimeIntervalType>::ValueType").pointerTypes("DayTimeIntervalType.DayMilliseconds"))
               .put(new Info("arrow::MonthDayNanoIntervalArray::TypeClass::MonthDayNanos",
                             "arrow::TemporalScalar<arrow::MonthDayNanoIntervalType>::ValueType").pointerTypes("MonthDayNanoIntervalType.MonthDayNanos"))

               .put(new Info("arrow::BaseListType", "arrow::BaseBinaryType", "arrow::BaseListScalar", "arrow::NestedType", "arrow::NumberType",
                             "arrow::Date64Scalar", "arrow::DayTimeIntervalScalar", "arrow::FixedSizeListScalar", "arrow::Int64Scalar", "arrow::UInt64Scalar",
                             "arrow::PrimitiveCType", "arrow::Scalar", "arrow::StructScalar", "arrow::TemporalType", "arrow::TimestampParser",
                             "arrow::ipc::RecordBatchStreamReader", "arrow::ipc::RecordBatchStreamWriter", "arrow::ipc::RecordBatchFileWriter",
                             "arrow::csv::StreamingReader", "arrow::compute::CompareFunction", "arrow::compute::KernelInitArgs").purify())

               .put(new Info("arrow::BaseBinaryScalar<arrow::BinaryType>").pointerTypes("BaseBinaryScalar").define())
               .put(new Info("arrow::BaseBinaryScalar<arrow::LargeBinaryType>").pointerTypes("BaseLargeBinaryScalar").define())
               .put(new Info("arrow::DateScalar<arrow::Date32Type>").pointerTypes("BaseDate32Scalar").define())
               .put(new Info("arrow::DateScalar<arrow::Date64Type>").pointerTypes("BaseDate64Scalar").define())
               .put(new Info("arrow::IntervalScalar<arrow::DayTimeIntervalType>").pointerTypes("BaseDayTimeIntervalScalar").define())
               .put(new Info("arrow::IntervalScalar<arrow::MonthDayNanoIntervalType>").pointerTypes("BaseMonthDayNanoIntervalScalar").define())
               .put(new Info("arrow::IntervalScalar<arrow::MonthIntervalType>").pointerTypes("BaseMonthIntervalScalar").define())
//               .put(new Info("arrow::NumericScalar<arrow::Date32Type>").pointerTypes("BaseDate32Scalar").define())
//               .put(new Info("arrow::NumericScalar<arrow::Date64Type>").pointerTypes("BaseDate64Scalar").define())
               .put(new Info("arrow::NumericScalar<arrow::Int8Type>").pointerTypes("BaseInt8Type").define())
               .put(new Info("arrow::NumericScalar<arrow::Int16Type>").pointerTypes("BaseInt16Type").define())
               .put(new Info("arrow::NumericScalar<arrow::Int32Type>").pointerTypes("BaseInt32Type").define())
               .put(new Info("arrow::NumericScalar<arrow::Int64Type>").pointerTypes("BaseInt64Type").define())
               .put(new Info("arrow::NumericScalar<arrow::UInt8Type>").pointerTypes("BaseUInt8Type").define())
               .put(new Info("arrow::NumericScalar<arrow::UInt16Type>").pointerTypes("BaseUInt16Type").define())
               .put(new Info("arrow::NumericScalar<arrow::UInt32Type>").pointerTypes("BaseUInt32Type").define())
               .put(new Info("arrow::NumericScalar<arrow::UInt64Type>").pointerTypes("BaseUInt64Type").define())
               .put(new Info("arrow::NumericScalar<arrow::HalfFloatType>").pointerTypes("BaseHalfFloatScalar").define())
               .put(new Info("arrow::NumericScalar<arrow::FloatType>").pointerTypes("BaseFloatScalar").define())
               .put(new Info("arrow::NumericScalar<arrow::DoubleType>").pointerTypes("BaseDoubleScalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::Date32Type>").pointerTypes("BaseBaseDate32Scalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::Date64Type>").pointerTypes("BaseBaseDate64Scalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::DurationType>").pointerTypes("BaseDurationScalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::DayTimeIntervalType>").pointerTypes("BaseBaseDayTimeIntervalScalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::MonthDayNanoIntervalType>").pointerTypes("BaseMonthDayNanoIntervalScalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::MonthIntervalType>").pointerTypes("BaseBaseMonthIntervalType").define())
               .put(new Info("arrow::TemporalScalar<arrow::TimestampType>").pointerTypes("BaseTimestampScalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::Time32Type>").pointerTypes("BaseBaseTime32Scalar").define())
               .put(new Info("arrow::TemporalScalar<arrow::Time64Type>").pointerTypes("BaseBaseTime64Scalar").define())
               .put(new Info("arrow::TimeScalar<arrow::Time32Type>").pointerTypes("BaseTime32Scalar").define())
               .put(new Info("arrow::TimeScalar<arrow::Time64Type>").pointerTypes("BaseTime64Scalar").define())

               .put(new Info("arrow::NumericArray<arrow::Int8Type>").pointerTypes("Int8Array").define())
               .put(new Info("arrow::NumericArray<arrow::Int16Type>").pointerTypes("Int16Array").define())
               .put(new Info("arrow::NumericArray<arrow::Int32Type>").pointerTypes("Int32Array").define())
               .put(new Info("arrow::NumericArray<arrow::Int64Type>").pointerTypes("Int64Array").define())
               .put(new Info("arrow::NumericArray<arrow::UInt8Type>").pointerTypes("UInt8Array").define())
               .put(new Info("arrow::NumericArray<arrow::UInt16Type>").pointerTypes("UInt16Array").define())
               .put(new Info("arrow::NumericArray<arrow::UInt32Type>").pointerTypes("UInt32Array").define())
               .put(new Info("arrow::NumericArray<arrow::UInt64Type>").pointerTypes("UInt64Array").define())
               .put(new Info("arrow::NumericArray<arrow::HalfFloatType>").pointerTypes("HalfFloatArray").define())
               .put(new Info("arrow::NumericArray<arrow::FloatType>").pointerTypes("FloatArray").define())
               .put(new Info("arrow::NumericArray<arrow::DoubleType>").pointerTypes("DoubleArray").define())
               .put(new Info("arrow::NumericArray<arrow::Date64Type>").pointerTypes("Date64Array").define())
               .put(new Info("arrow::NumericArray<arrow::Date32Type>").pointerTypes("Date32Array").define())
               .put(new Info("arrow::NumericArray<arrow::Time32Type>").pointerTypes("Time32Array").define())
               .put(new Info("arrow::NumericArray<arrow::Time64Type>").pointerTypes("Time64Array").define())
               .put(new Info("arrow::NumericArray<arrow::TimestampType>").pointerTypes("TimestampArray").define())
               .put(new Info("arrow::NumericArray<arrow::MonthIntervalType>").pointerTypes("MonthIntervalArray").define())
//               .put(new Info("arrow::NumericArray<arrow::DayTimeInterval>").pointerTypes("DayTimeIntervalArray").define())
               .put(new Info("arrow::NumericArray<arrow::DurationType>").pointerTypes("DurationArray").define())
//               .put(new Info("arrow::NumericArray<arrow::ExtensionType>").pointerTypes("ExtensionArray").define())

               .put(new Info("finalize").javaNames("_finalize"))
               .put(new Info("arrow::ArrayData::GetValues<jbyte>").javaNames("GetValuesByte"))
               .put(new Info("arrow::ArrayData::GetValues<jshort>").javaNames("GetValuesShort"))
               .put(new Info("arrow::ArrayData::GetValues<jint>").javaNames("GetValuesInt"))
               .put(new Info("arrow::ArrayData::GetValues<jlong>").javaNames("GetValuesLong"))
               .put(new Info("arrow::ArrayData::GetValues<float>").javaNames("GetValuesFloat"))
               .put(new Info("arrow::ArrayData::GetValues<double>").javaNames("GetValuesDouble"))

               .put(new Info("std::shared_ptr<arrow::Scalar>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::Scalar>\"}) Scalar").pointerTypes("Scalar"))
               .put(new Info("std::shared_ptr<arrow::Field>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::Field>\"}) Field").pointerTypes("Field"))
               .put(new Info("std::shared_ptr<arrow::Array>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::Array>\"}) Array").pointerTypes("Array"))
               .put(new Info("std::shared_ptr<arrow::ArrayData>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::ArrayData>\"}) ArrayData").pointerTypes("ArrayData"))
               .put(new Info("std::shared_ptr<arrow::Buffer>").annotations("@SharedPtr").valueTypes("ArrowBuffer").pointerTypes("@Cast({\"\", \"std::shared_ptr<arrow::Buffer>*\"}) ArrowBuffer"))
               .put(new Info("std::shared_ptr<arrow::ResizableBuffer>").annotations("@SharedPtr").pointerTypes("ResizableBuffer"))
               .put(new Info("std::shared_ptr<arrow::ArrayBuilder>").annotations("@SharedPtr").pointerTypes("ArrayBuilder"))
               .put(new Info("std::shared_ptr<arrow::RecordBatch>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::RecordBatch>\"}) RecordBatch").pointerTypes("RecordBatch"))
               .put(new Info("std::shared_ptr<arrow::ChunkedArray>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::ChunkedArray>\"}) ChunkedArray").pointerTypes("ChunkedArray"))
               .put(new Info("std::shared_ptr<arrow::Schema>").annotations("@SharedPtr").pointerTypes("Schema"))
               .put(new Info("std::shared_ptr<arrow::Table>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::Table>\"}) Table").pointerTypes("Table"))
//               .put(new Info("std::shared_ptr<arrow::TimestampParser>").annotations("@SharedPtr").pointerTypes("TimestampParser"))
               .put(new Info("std::shared_ptr<arrow::ListArray>").annotations("@SharedPtr").pointerTypes("ListArray"))
               .put(new Info("std::shared_ptr<arrow::LargeListArray>").annotations("@SharedPtr").pointerTypes("LargeListArray"))
               .put(new Info("std::shared_ptr<arrow::BinaryArray>").annotations("@SharedPtr").pointerTypes("BinaryArray"))
               .put(new Info("std::shared_ptr<arrow::StructArray>").annotations("@SharedPtr").pointerTypes("StructArray"))
               .put(new Info("std::shared_ptr<arrow::DataType>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::DataType>\"}) DataType")
                                                                                          .pointerTypes("@Cast({\"\", \"std::shared_ptr<arrow::DataType>*\"}) DataType"))
               .put(new Info("std::shared_ptr<const arrow::KeyValueMetadata>").annotations("@SharedPtr")
                                                                              .valueTypes("@Cast({\"const arrow::KeyValueMetadata*\", \"std::shared_ptr<const arrow::KeyValueMetadata>\"}) KeyValueMetadata")
                                                                              .pointerTypes("@Cast(\"const arrow::KeyValueMetadata*\") KeyValueMetadata"))
               .put(new Info("std::shared_ptr<arrow::compute::KernelSignature>").annotations("@SharedPtr").valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::compute::KernelSignature>\"}) KernelSignature")
                                                                                .pointerTypes("KernelSignature"))
               .put(new Info("std::shared_ptr<arrow::compute::SelectionVector>").annotations("@SharedPtr").pointerTypes("SelectionVector"))
               .put(new Info("std::unique_ptr<arrow::compute::KernelState>").annotations("@UniquePtr")
                                                                            .valueTypes("@Cast({\"\", \"std::unique_ptr<arrow::compute::KernelState>&&\"}) KernelState")
                                                                            .pointerTypes("KernelState"))
               .put(new Info("std::shared_ptr<arrow::internal::ThreadPool>").annotations("@SharedPtr").pointerTypes("ThreadPool"))
               .put(new Info("std::shared_ptr<arrow::io::InputStream>",
                             "arrow::Future<std::shared_ptr<arrow::io::InputStream> >::ValueType>").annotations("@SharedPtr")
                                                                       .valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::io::InputStream>\"}) InputStream")
                                                                       .pointerTypes("InputStream"))
               .put(new Info("std::shared_ptr<arrow::io::OutputStream>").annotations("@SharedPtr").valueTypes("OutputStream")
                                                                        .pointerTypes("@Cast({\"\", \"std::shared_ptr<arrow::io::OutputStream>*\"}) OutputStream"))
               .put(new Info("std::shared_ptr<arrow::io::FileOutputStream>").annotations("@SharedPtr").valueTypes("FileOutputStream")
                                                                            .pointerTypes("@Cast({\"\", \"std::shared_ptr<arrow::io::FileOutputStream>*\"}) FileOutputStream"))
               .put(new Info("std::shared_ptr<arrow::io::RandomAccessFile>",
                             "arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >::ValueType>").annotations("@SharedPtr")
                                                                            .valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::io::RandomAccessFile>\"}) RandomAccessFile")
                                                                            .pointerTypes("RandomAccessFile"))
               .put(new Info("std::shared_ptr<arrow::csv::StreamingReader>").annotations("@SharedPtr")
                                                                            .valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::csv::StreamingReader>\"}) StreamingReader")
                                                                            .pointerTypes("StreamingReader"))
               .put(new Info("std::shared_ptr<arrow::ipc::Message>").annotations("@SharedPtr")
                                                                    .valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::ipc::Message>\"}) Message")
                                                                    .pointerTypes("Message"))
               .put(new Info("std::shared_ptr<arrow::ipc::RecordBatchFileReader>").annotations("@SharedPtr")
                                                                                  .valueTypes("@Cast({\"\", \"std::shared_ptr<arrow::ipc::RecordBatchFileReader>\"}) RecordBatchFileReader")
                                                                                  .pointerTypes("RecordBatchFileReader"))

               .put(new Info("std::vector<std::shared_ptr<arrow::DataType> >").pointerTypes("DataTypeVector").define())
               .put(new Info("std::vector<std::shared_ptr<const arrow::KeyValueMetadata> >").pointerTypes("KeyValueMetadataVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::Scalar> >").pointerTypes("ScalarVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::Field> >").pointerTypes("FieldVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::Array> >").pointerTypes("ArrayVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::ArrayData> >").pointerTypes("ArrayDataVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::Buffer> >").pointerTypes("BufferVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::ArrayBuilder> >").pointerTypes("ArrayBuilderVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::RecordBatch> >").pointerTypes("RecordBatchVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::ChunkedArray> >").pointerTypes("ChunkedArrayVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::Schema> >").pointerTypes("SchemaVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::Table> >").pointerTypes("TableVector").define())
//               .put(new Info("std::vector<std::shared_ptr<arrow::TimestampParser> >").pointerTypes("TimestampParserVector").define())
               .put(new Info("const std::vector<std::unique_ptr<arrow::compute::KernelState> >",
                                   "std::vector<std::unique_ptr<arrow::compute::KernelState> >").pointerTypes("KernelStateVector").define())
               .put(new Info("std::vector<std::vector<std::shared_ptr<arrow::Array> > >", "std::vector<arrow::ArrayVector>").pointerTypes("ArrayVectorVector").define())
               .put(new Info("std::vector<std::pair<int64_t,std::shared_ptr<arrow::Array> > >").pointerTypes("DictionaryVector").define())
               .put(new Info("std::vector<arrow::Datum>").pointerTypes("DatumVector").define())
//               .put(new Info("std::vector<arrow::FutureImpl*>").pointerTypes("FutureImplVector").define())
               .put(new Info("std::vector<arrow::ValueDescr>").pointerTypes("ValueDescrVector").define())
               .put(new Info("std::vector<arrow::fs::FileInfo>", "arrow::fs::FileInfoVector",
                             "arrow::Future<std::vector<arrow::fs::FileInfo> >::ValueType").pointerTypes("FileInfoVector").define())
//               .put(new Info("std::vector<arrow::fs::FileStats>").pointerTypes("FileStatsVector").define())
               .put(new Info("arrow::BaseListArray<arrow::ListType>").pointerTypes("BaseListArray").define())
               .put(new Info("arrow::BaseBinaryArray<arrow::BinaryType>").pointerTypes("BaseBinaryArray").define())
               .put(new Info("arrow::BaseListArray<arrow::LargeListType>").pointerTypes("BaseLargeListArray").define())
               .put(new Info("arrow::BaseBinaryArray<arrow::LargeBinaryType>").pointerTypes("BaseLargeBinaryArray").define())

               .put(new Info("arrow::Enumerated<std::shared_ptr<arrow::RecordBatch> >").pointerTypes("RecordBatchEnumerated").define())
               .put(new Info("arrow::Future<>").pointerTypes("Future").define().purify())
               .put(new Info("arrow::Future<>::result", "arrow::Future<>::MoveResult", "arrow::Future<>::MarkFinished", "arrow::Future<>::MakeFinished",
                             "arrow::Future<>::internal::Empty",
                             "arrow::Future<>::WrapResultyOnComplete",
                             "arrow::Future<>::WrapStatusyOnComplete",
                             "arrow::Future<>::ThenOnComplete",
                             "arrow::Future<>::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<bool>").pointerTypes("BoolResult").define())
               .put(new Info("arrow::Result<int>", "arrow::Result<int32_t>").pointerTypes("IntResult").define())
               .put(new Info("arrow::Future<int64_t>",
                             "arrow::Future<arrow::Future<int64_t>::ValueType>").pointerTypes("LongFuture").define())
               .put(new Info("arrow::Future<int64_t>::internal::Empty",
                             "arrow::Future<int64_t>::WrapResultyOnComplete",
                             "arrow::Future<int64_t>::WrapStatusyOnComplete",
                             "arrow::Future<int64_t>::ThenOnComplete",
                             "arrow::Future<int64_t>::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<int64_t>",
                             "arrow::Result<arrow::Future<int64_t>::ValueType>").pointerTypes("LongResult").define())
               .put(new Info("arrow::util::optional<int64_t>").pointerTypes("LongOptional").define())
               .put(new Info("arrow::Future<arrow::util::optional<int64_t> >",
                             "arrow::Future<arrow::Future<arrow::util::optional<int64_t> >::ValueType>").pointerTypes("LongOptionalFuture").define())
               .put(new Info("arrow::Future<arrow::util::optional<int64_t> >::internal::Empty",
                             "arrow::Future<arrow::util::optional<int64_t> >::WrapResultyOnComplete",
                             "arrow::Future<arrow::util::optional<int64_t> >::WrapStatusyOnComplete",
                             "arrow::Future<arrow::util::optional<int64_t> >::ThenOnComplete",
                             "arrow::Future<arrow::util::optional<int64_t> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<arrow::util::optional<int64_t> >",
                             "arrow::Result<arrow::Future<arrow::util::optional<int64_t> >::ValueType>").pointerTypes("LongOptionalResult").define())
               .put(new Info("arrow::Result<size_t>").pointerTypes("SizeTResult").define())
               .put(new Info("arrow::Result<std::string>").pointerTypes("StringResult").define())
               .put(new Info("arrow::Result<arrow::ValueDescr>").pointerTypes("ValueDescrResult").define())
               .put(new Info("arrow::Result<arrow::ArrayVector>").pointerTypes("ArrayVectorResult").define())
               .put(new Info("arrow::Result<arrow::Compression::type>").pointerTypes("CompressionTypeResult").define())
               .put(new Info("arrow::Result<arrow::FieldRef>").pointerTypes("FieldRefResult").define())
               .put(new Info("arrow::Result<arrow::Decimal128>").pointerTypes("Decimal128Result").define())
               .put(new Info("arrow::Result<arrow::Decimal256>").pointerTypes("Decimal256Result").define())
               .put(new Info("arrow::Result<std::pair<arrow::Decimal128,arrow::Decimal128> >").pointerTypes("Decimal128PairResult").define())
               .put(new Info("arrow::Result<std::pair<arrow::Decimal256,arrow::Decimal256> >").pointerTypes("Decimal256PairResult").define())
               .put(new Info("arrow::Result<std::pair<std::string,std::string> >").pointerTypes("StringPairResult").define())
               .put(new Info("arrow::Result<arrow::util::string_view>").pointerTypes("StringViewResult").define())
               .put(new Info("arrow::Result<arrow::fs::LocalFileSystemOptions>").pointerTypes("LocalFileSystemOptionsResult").define())
               .put(new Info("arrow::Result<arrow::fs::LocalFileSystemOptions>::Equals").skip())
               .put(new Info("arrow::Result<arrow::fs::HdfsOptions>").pointerTypes("HDFSOptionsResult").define())
               .put(new Info("arrow::Result<arrow::fs::HdfsOptions>::Equals").skip())
//               .put(new Info("arrow::Result<arrow::fs::S3Options>").pointerTypes("S3OptionsResult").define())
//               .put(new Info("arrow::Result<arrow::fs::S3Options>::Equals").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::internal::ThreadPool> >").pointerTypes("ThreadPoolResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Array> >").pointerTypes("ArrayResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ArrayData> >").pointerTypes("ArrayDataResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Buffer> >").pointerTypes("BufferResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::Buffer> >").pointerTypes("BufferUniqueResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::Buffer> >(const arrow::Result<std::unique_ptr<arrow::Buffer> >&)",
                             "arrow::Result<std::unique_ptr<arrow::Buffer> >::operator =").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ResizableBuffer> >").pointerTypes("ResizableResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ResizableBuffer> >").pointerTypes("ResizableUniqueResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ResizableBuffer> >(const arrow::Result<std::unique_ptr<arrow::ResizableBuffer> >&)",
                             "arrow::Result<std::unique_ptr<arrow::ResizableBuffer> >::operator =").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::DataType> >").pointerTypes("DataTypeResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Field> >").pointerTypes("FieldResult").define())
               .put(new Info("arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata> >").pointerTypes("KeyValueMetadataFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata> >").pointerTypes("KeyValueMetadataResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ListArray> >").pointerTypes("ListArrayResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::LargeListArray> >").pointerTypes("LargeListArrayResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::BinaryArray> >").pointerTypes("BinaryArrayResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::StructArray> >").pointerTypes("StructArrayResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::StructScalar> >").pointerTypes("StructScalarResult").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::RecordBatch> >").pointerTypes("RecordBatchFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::RecordBatch> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::RecordBatch> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::RecordBatch> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::RecordBatch> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::RecordBatch> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::RecordBatch> >").pointerTypes("RecordBatchResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Scalar> >").pointerTypes("ScalarResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ChunkedArray> >").pointerTypes("ChunkedArrayResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Schema> >").pointerTypes("SchemaResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::SparseTensor> >").pointerTypes("SparseTensorResult").define().purify())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::Table> >").pointerTypes("TableFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::Table> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::Table> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::Table> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::Table> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::Table> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Table> >").pointerTypes("TableResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::Tensor> >").pointerTypes("TensorResult").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::io::InputStream> >").pointerTypes("InputStreamFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::io::InputStream> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::io::InputStream> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::io::InputStream> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::io::InputStream> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::io::InputStream> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >").pointerTypes("RandomAccessFileFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::InputStream> >").pointerTypes("InputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::OutputStream> >", "arrow::Result<std::shared_ptr<io::OutputStream> >").pointerTypes("OutputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::MemoryMappedFile> >").pointerTypes("MemoryMappedFileResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::ReadableFile> >").pointerTypes("ReadableFileResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile> >").pointerTypes("RandomAccessFileResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::FileOutputStream> >").pointerTypes("FileOutputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::BufferOutputStream> >").pointerTypes("BufferOutputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::BufferedInputStream> >").pointerTypes("BufferedInputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::BufferedOutputStream> >").pointerTypes("BufferedOutputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::CompressedInputStream> >").pointerTypes("CompressedInputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::CompressedOutputStream> >").pointerTypes("CompressedOutputStreamResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile> >", "arrow::Result<std::shared_ptr<io::RandomAccessFile> >").pointerTypes("RandomAccessFileResult").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >",
                             "arrow::Future<arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::ValueType>").pointerTypes("StreamingReaderFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::csv::StreamingReader> >",
                             "arrow::Result<arrow::Future<std::shared_ptr<arrow::csv::StreamingReader> >::ValueType>").pointerTypes("StreamingReaderResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::csv::TableReader> >").pointerTypes("TableReaderResult").define())
//               .put(new Info("arrow::Result<std::shared_ptr<arrow::compute::CastFunction> >").pointerTypes("CastFunctionResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::compute::Function> >").pointerTypes("FunctionResult").define())
               .put(new Info("arrow::Result<arrow::Datum>").pointerTypes("DatumResult").define())
               .put(new Info("arrow::Result<arrow::Datum>::Equals").skip())
               .put(new Info("arrow::Result<arrow::StopSource*>").pointerTypes("StopSourceResult").define())
               .put(new Info("arrow::Result<arrow::compute::Expression>").pointerTypes("ExpressionResult").define())
               .put(new Info("arrow::Result<arrow::compute::ExecBatch>").pointerTypes("ExecBatchResult").define())
               .put(new Info("arrow::Result<arrow::compute::ExecBatch>::Equals").skip())
               .put(new Info("arrow::Result<arrow::compute::ExecNode*>").pointerTypes("ExecNodeResult").define())
               .put(new Info("arrow::Result<arrow::compute::FunctionOptionsType*>", "arrow::Result<const arrow::compute::FunctionOptionsType*>").cast().pointerTypes("FunctionOptionsTypeResult").define())
               .put(new Info("arrow::Result<arrow::compute::Kernel*>", "arrow::Result<const arrow::compute::Kernel*>").cast().pointerTypes("KernelResult").define())
               .put(new Info("arrow::Result<arrow::compute::ScalarKernel*>", "arrow::Result<const arrow::compute::ScalarKernel*>").cast().pointerTypes("ScalarKernelResult").define())
               .put(new Info("arrow::Result<arrow::compute::ScalarAggregateKernel*>", "arrow::Result<const arrow::compute::ScalarAggregateKernel*>").cast().pointerTypes("ScalarAggregateKernelResult").define())
               .put(new Info("arrow::Result<arrow::compute::VectorKernel*>", "arrow::Result<const arrow::compute::VectorKernel*>").cast().pointerTypes("VectorKernelResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::compute::SelectionVector> >").pointerTypes("SelectionVectorResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::compute::FunctionOptions> >").pointerTypes("FunctionOptionsResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::compute::FunctionOptions> >(const arrow::Result<std::unique_ptr<arrow::compute::FunctionOptions> >&)",
                             "arrow::Result<std::unique_ptr<arrow::compute::FunctionOptions> >::operator =").skip())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::compute::KernelState> >").pointerTypes("KernelStateResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::compute::KernelState> >(const arrow::Result<std::unique_ptr<arrow::compute::KernelState> >&)",
                             "arrow::Result<std::unique_ptr<arrow::compute::KernelState> >::operator =").skip())
               .put(new Info("arrow::Result<arrow::fs::FileInfo>").pointerTypes("FileInfoResult").define())
//               .put(new Info("arrow::Result<arrow::fs::FileStats>").pointerTypes("FileStatsResult").define())
//               .put(new Info("arrow::Result<arrow::fs::PathForest>").pointerTypes("PathForestResult").define())
               .put(new Info("arrow::Result<arrow::ipc::DictionaryVector>").pointerTypes("DictionaryVectorResult").define())
               .put(new Info("arrow::Result<std::vector<arrow::Datum> >").pointerTypes("DatumVectorResult").define())
               .put(new Info("arrow::Future<std::vector<arrow::fs::FileInfo> >", "arrow::Future<arrow::fs::FileInfoVector>",
                             "arrow::Future<arrow::Future<std::vector<arrow::fs::FileInfo> >::ValueType>").pointerTypes("FileInfoVectorFuture").define())
               .put(new Info("arrow::Future<std::vector<arrow::fs::FileInfo> >::internal::Empty",
                             "arrow::Future<std::vector<arrow::fs::FileInfo> >::WrapResultyOnComplete",
                             "arrow::Future<std::vector<arrow::fs::FileInfo> >::WrapStatusyOnComplete",
                             "arrow::Future<std::vector<arrow::fs::FileInfo> >::ThenOnComplete",
                             "arrow::Future<std::vector<arrow::fs::FileInfo> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::vector<arrow::fs::FileInfo> >", "arrow::Result<arrow::fs::FileInfoVector>",
                             "arrow::Result<arrow::Future<std::vector<arrow::fs::FileInfo> >::ValueType>").pointerTypes("FileInfoVectorResult").define())
//               .put(new Info("arrow::Result<std::vector<arrow::fs::FileStats> >").pointerTypes("FileStatsVectorResult").define())
               .put(new Info("arrow::Result<std::vector<std::shared_ptr<arrow::Buffer> > >").pointerTypes("BufferVectorResult").define())
               .put(new Info("arrow::Result<std::vector<std::shared_ptr<arrow::ChunkedArray> > >").pointerTypes("ChunkedArrayVectorResult").define())
               .put(new Info("arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >", "arrow::Future<arrow::RecordBatchVector>",
                             "arrow::Future<arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::ValueType>").pointerTypes("RecordBatchVectorFuture").define())
               .put(new Info("arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::internal::Empty",
                             "arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::WrapResultyOnComplete",
                             "arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::WrapStatusyOnComplete",
                             "arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::ThenOnComplete",
                             "arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch> > >", "arrow::Result<arrow::RecordBatchVector>",
                             "arrow::Result<arrow::Future<std::vector<std::shared_ptr<arrow::RecordBatch> > >::ValueType>").pointerTypes("RecordBatchVectorResult").define())
               .put(new Info("arrow::Result<std::vector<std::shared_ptr<arrow::Schema> > >").pointerTypes("SchemaVectorResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::fs::FileSystem> >").pointerTypes("FileSystemResult").define())
//               .put(new Info("arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem> >").pointerTypes("S3FileSystemResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::util::Compressor> >").pointerTypes("CompressorResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::util::Decompressor> >").pointerTypes("DecompressorResult").define())
               .put(new Info("arrow::Result<arrow::util::Compressor::CompressResult>").pointerTypes("CompressResultResult").define())
               .put(new Info("arrow::Result<arrow::util::Compressor::CompressResult>::Equals").skip())
               .put(new Info("arrow::Result<arrow::util::Compressor::EndResult>").pointerTypes("EndResultResult").define())
               .put(new Info("arrow::Result<arrow::util::Compressor::EndResult>::Equals").skip())
               .put(new Info("arrow::Result<arrow::util::Compressor::FlushResult>").pointerTypes("FlushResultResult").define())
               .put(new Info("arrow::Result<arrow::util::Compressor::FlushResult>::Equals").skip())
               .put(new Info("arrow::Result<arrow::util::Decompressor::DecompressResult>").pointerTypes("DecompressResultResult").define())
               .put(new Info("arrow::Result<arrow::util::Decompressor::DecompressResult>::Equals").skip())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ipc::internal::IpcPayloadWriter> >").pointerTypes("IpcPayloadWriterResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ipc::internal::IpcPayloadWriter> >(const arrow::Result<std::unique_ptr<arrow::ipc::internal::IpcPayloadWriter> >&)",
                             "arrow::Result<std::unique_ptr<arrow::ipc::internal::IpcPayloadWriter> >::operator =").skip())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::ipc::Message> >",
                             "arrow::Future<arrow::Future<std::shared_ptr<arrow::ipc::Message> >::ValueType>").pointerTypes("MessageFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::ipc::Message> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::ipc::Message> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::ipc::Message> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::ipc::Message> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::ipc::Message> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ipc::Message> >",
                             "arrow::Result<arrow::Future<std::shared_ptr<arrow::ipc::Message> >::ValueType>").pointerTypes("MessageSharedResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ipc::Message> >").pointerTypes("MessageUniqueResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ipc::Message> >(const arrow::Result<std::unique_ptr<arrow::ipc::Message> >&)",
                             "arrow::Result<std::unique_ptr<arrow::ipc::Message> >::operator =").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ipc::feather::Reader> >").pointerTypes("FeatherReaderResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchStreamReader> >").pointerTypes("RecordBatchStreamReaderResult").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >",
                             "arrow::Future<arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::ValueType>").pointerTypes("RecordBatchFileReaderFuture").define())
               .put(new Info("arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::internal::Empty",
                             "arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::WrapResultyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::WrapStatusyOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::ThenOnComplete",
                             "arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::PassthruOnFailure").skip())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >",
                             "arrow::Result<arrow::Future<std::shared_ptr<arrow::ipc::RecordBatchFileReader> >::ValueType>").pointerTypes("RecordBatchFileReaderResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchReader> >", "arrow::Result<std::shared_ptr<arrow::RecordBatchReader> >").pointerTypes("RecordBatchReaderSharedResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter> >").pointerTypes("RecordBatchWriterSharedResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ipc::RecordBatchWriter> >").pointerTypes("RecordBatchWriterUniqueResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::ipc::RecordBatchWriter> >(const arrow::Result<std::unique_ptr<arrow::ipc::RecordBatchWriter> >&)",
                             "arrow::Result<std::unique_ptr<arrow::ipc::RecordBatchWriter> >::operator =").skip())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::util::Codec> >").pointerTypes("CodecResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::util::Codec> >(const arrow::Result<std::unique_ptr<arrow::util::Codec> >&)",
                             "arrow::Result<std::unique_ptr<arrow::util::Codec> >::operator =").skip())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::DictionaryUnifier> >").pointerTypes("DictionaryUnifierResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::DictionaryUnifier> >(const arrow::Result<std::unique_ptr<arrow::DictionaryUnifier> >&)",
                             "arrow::Result<std::unique_ptr<arrow::DictionaryUnifier> >::operator =").skip())
               .put(new Info("arrow::Result<arrow::Iterator<std::shared_ptr<arrow::Buffer> > >").pointerTypes("BufferIteratorResult").define())
               .put(new Info("arrow::Result<arrow::Iterator<std::shared_ptr<arrow::Buffer> > >(arrow::Iterator<std::shared_ptr<arrow::Buffer> >)",
                             "arrow::Result<arrow::Iterator<std::shared_ptr<arrow::Buffer> > >(const arrow::Result<arrow::Iterator<std::shared_ptr<arrow::Buffer> > >&)",
                             "arrow::Result<arrow::Iterator<std::shared_ptr<arrow::Buffer> > >::operator =").skip())
               .put(new Info("arrow::Result<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > >", "arrow::Result<arrow::RecordBatchIterator>").pointerTypes("RecordBatchIteratorResult").define())
               .put(new Info("arrow::Result<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > >(arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >)",
                             "arrow::Result<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > >(const arrow::Result<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > >&)",
                             "arrow::Result<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > >::operator =").skip())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::Buffer> >").pointerTypes("BufferIterator").define())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::Buffer> >(const arrow::Iterator<std::shared_ptr<arrow::Buffer> >&)",
                             "arrow::Iterator<std::shared_ptr<arrow::Buffer> >::RangeIterator(arrow::Iterator<std::shared_ptr<arrow::Buffer> >)",
                             "arrow::Iterator<std::shared_ptr<arrow::Buffer> >::operator =").skip())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::Buffer> >::RangeIterator").pointerTypes("BufferIterator.RangeIterator").define())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >").pointerTypes("RecordBatchIterator").define())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >(const arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >&)",
                             "arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >::RangeIterator(arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >)",
                             "arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >::operator =").skip())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::RecordBatch> >::RangeIterator").pointerTypes("RecordBatchIterator.RangeIterator").define())
               .put(new Info("std::unordered_map<std::string,std::shared_ptr<arrow::DataType> >").pointerTypes("StringDataTypeMap").define())
               .put(new Info("std::unordered_map<arrow::FieldRef,arrow::Datum,arrow::FieldRef::Hash>").pointerTypes("FieldRefDatumMap").define())
               .put(new Info("arrow::Result<std::unordered_map<arrow::FieldRef,arrow::Datum,arrow::FieldRef::Hash> >").pointerTypes("FieldRefDatumMapResult").define())

               .put(new Info("arrow::BaseListBuilder<arrow::ListType>").pointerTypes("BaseListBuilder").define().purify())
               .put(new Info("arrow::BaseListBuilder<arrow::LargeListType>").pointerTypes("BaseLargeListBuilder").define().purify())
               .put(new Info("arrow::BaseBinaryBuilder<arrow::BinaryType>").pointerTypes("BaseBinaryBuilder").define().purify())
               .put(new Info("arrow::BaseBinaryBuilder<arrow::LargeBinaryType>").pointerTypes("BaseLargeBinaryBuilder").define().purify())
               .put(new Info("arrow::internal::BinaryDictionaryBuilderImpl<arrow::BinaryType>",
                             "arrow::internal::BinaryDictionaryBuilderImpl<arrow::StringType>",
                             "arrow::internal::BinaryDictionary32BuilderImpl<arrow::BinaryType>",
                             "arrow::internal::BinaryDictionary32BuilderImpl<arrow::StringType>",
                             "arrow::CTypeTraits<const char*>", "arrow::CTypeTraits<std::string>", "arrow::DictionaryScalar::ValueType", "arrow::fs::TimePoint",
                             "std::enable_shared_from_this<arrow::fs::FileSystem>", "std::enable_shared_from_this<FileSystem>",
                             "std::enable_shared_from_this<arrow::io::RandomAccessFile>", "std::enable_shared_from_this<RandomAccessFile>",
                             "std::enable_shared_from_this<arrow::io::InputStream>",
                             "std::enable_shared_from_this<arrow::ipc::RecordBatchFileReader>",
                             "std::enable_shared_from_this<arrow::internal::TaskGroup>",
                             "std::function<std::unique_ptr<arrow::detail::ReadaheadPromise>()>",
                             "std::function<arrow::Result<arrow::Future<> >()>",
                             "std::function<arrow::csv::InvalidRowResult(const arrow::csv::InvalidRow&)>", "arrow::csv::InvalidRowHandler",
                             "arrow::BooleanArray::IteratorType",
                             "arrow::NumericArray<arrow::Int8Type>::IteratorType",
                             "arrow::NumericArray<arrow::UInt8Type>::IteratorType",
                             "arrow::NumericArray<arrow::Int16Type>::IteratorType",
                             "arrow::NumericArray<arrow::UInt16Type>::IteratorType",
                             "arrow::NumericArray<arrow::Int32Type>::IteratorType",
                             "arrow::NumericArray<arrow::UInt32Type>::IteratorType",
                             "arrow::NumericArray<arrow::Date32Type>::IteratorType",
                             "arrow::NumericArray<arrow::Time32Type>::IteratorType",
                             "arrow::NumericArray<arrow::MonthIntervalType>::IteratorType",
                             "arrow::NumericArray<arrow::Int64Type>::IteratorType",
                             "arrow::NumericArray<arrow::UInt64Type>::IteratorType",
                             "arrow::NumericArray<arrow::Date64Type>::IteratorType",
                             "arrow::NumericArray<arrow::Time64Type>::IteratorType",
                             "arrow::NumericArray<arrow::DurationType>::IteratorType",
                             "arrow::NumericArray<arrow::TimestampType>::IteratorType",
                             "arrow::NumericArray<arrow::HalfFloatType>::IteratorType",
                             "arrow::NumericArray<arrow::FloatType>::IteratorType",
                             "arrow::NumericArray<arrow::DoubleType>::IteratorType",
                             "arrow::util::ToStringOstreamable<arrow::Schema>",
                             "arrow::util::ToStringOstreamable<arrow::Status>", "arrow::util::ToStringOstreamable<Status>",
                             "arrow::util::EqualityComparable<arrow::Scalar>",
                             "arrow::util::EqualityComparable<arrow::Schema>",
                             "arrow::util::EqualityComparable<arrow::Status>", "arrow::util::EqualityComparable<Status>",
                             "arrow::util::EqualityComparable<arrow::compute::FunctionOptions>",
                             "arrow::util::EqualityComparable<arrow::compute::SortKey>", "arrow::util::EqualityComparable<SortKey>",
                             "arrow::util::EqualityComparable<arrow::fs::FileInfo>", "arrow::util::EqualityComparable<FileInfo>",
//                             "arrow::util::EqualityComparable<arrow::fs::FileStats>", "arrow::util::EqualityComparable<FileStats>",
//                             "arrow::util::EqualityComparable<arrow::fs::PathForest>", "arrow::util::EqualityComparable<PathForest>",
                             "arrow::util::EqualityComparable<arrow::Result<bool> >",
                             "arrow::util::EqualityComparable<arrow::Result<int> >",
                             "arrow::util::EqualityComparable<arrow::Result<int64_t> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::util::optional<int64_t> > >",
                             "arrow::util::EqualityComparable<arrow::Result<size_t> >",
                             "arrow::util::EqualityComparable<arrow::Result<std::string> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::ValueDescr> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::ArrayVector> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Compression::type> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::FieldRef> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Decimal128> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Decimal256> >",
                             "arrow::util::EqualityComparable<arrow::Result<std::pair<arrow::Decimal128,arrow::Decimal128> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::pair<arrow::Decimal256,arrow::Decimal256> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::pair<std::string,std::string> > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::util::string_view> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::fs::LocalFileSystemOptions> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::fs::HdfsOptions> >",
//                             "arrow::util::EqualityComparable<arrow::Result<arrow::fs::S3Options> >",
                             "arrow::util::EqualityComparable<arrow::Iterator<std::shared_ptr<arrow::Buffer> > >",
                             "arrow::util::EqualityComparable<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Iterator<std::shared_ptr<arrow::Buffer> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Iterator<std::shared_ptr<arrow::RecordBatch> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::RecordBatchIterator> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Datum> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::StopSource*> >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unordered_map<arrow::FieldRef,arrow::Datum,arrow::FieldRef::Hash> > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::Expression> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::ExecBatch> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::ExecNode*> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::FunctionOptionsType*> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::Kernel*> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::ScalarKernel*> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::ScalarAggregateKernel*> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::compute::VectorKernel*> >",
//                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::compute::CastFunction> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::compute::Function> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::compute::SelectionVector> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::compute::FunctionOptions> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::compute::KernelState> > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::fs::FileInfo> >",
//                             "arrow::util::EqualityComparable<arrow::Result<arrow::fs::FileStats> >",
//                             "arrow::util::EqualityComparable<arrow::Result<arrow::fs::PathForest> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::ipc::DictionaryVector> >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::fs::FileSystem> > >",
//                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::fs::S3FileSystem> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<arrow::Datum> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<arrow::fs::FileInfo> > >",
//                             "arrow::util::EqualityComparable<arrow::Result<std::vector<arrow::fs::FileStats> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<std::shared_ptr<arrow::Buffer> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<std::shared_ptr<arrow::ChunkedArray> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<std::shared_ptr<arrow::Schema> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::util::Compressor::CompressResult> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::util::Compressor::EndResult> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::util::Compressor::FlushResult> >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::util::Decompressor::DecompressResult> >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::util::Compressor> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::util::Decompressor> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::util::Codec> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::DictionaryUnifier> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::ipc::RecordBatchWriter> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchReader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchFileReader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchStreamReader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::ipc::internal::IpcPayloadWriter> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ipc::feather::Reader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ipc::Message> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::ipc::Message> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::InputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::OutputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::MemoryMappedFile> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::ReadableFile> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::FileOutputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::BufferOutputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::BufferedInputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::BufferedOutputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::CompressedInputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::CompressedOutputStream> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::csv::StreamingReader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::csv::TableReader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::internal::ThreadPool> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Array> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ArrayData> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Buffer> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::Buffer> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ResizableBuffer> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::ResizableBuffer> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::DataType> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Field> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ListArray> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::LargeListArray> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::BinaryArray> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::RecordBatch> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Scalar> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::ChunkedArray> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Schema> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::StructArray> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::StructScalar> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::SparseTensor> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Table> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::Tensor> > >").cast().pointerTypes("Pointer"))
               .put(new Info("arrow::internal::PrimitiveScalar",
                             "arrow::internal::PrimitiveScalar<arrow::DurationType>",
                             "arrow::internal::PrimitiveScalar<arrow::DayTimeIntervalType>",
                             "arrow::internal::PrimitiveScalar<arrow::MonthDayNanoIntervalType>",
                             "arrow::internal::PrimitiveScalar<arrow::Time32Type>",
                             "arrow::internal::PrimitiveScalar<arrow::Time64Type>",
                             "arrow::internal::PrimitiveScalar<arrow::TimestampType>",
                             "arrow::internal::PrimitiveScalar<arrow::MonthIntervalType>",
                             "arrow::internal::PrimitiveScalar<arrow::BooleanType,bool>",
                             "arrow::internal::PrimitiveScalar<arrow::Date32Type>",
                             "arrow::internal::PrimitiveScalar<arrow::Date64Type>",
                             "arrow::internal::PrimitiveScalar<arrow::Int8Type>",
                             "arrow::internal::PrimitiveScalar<arrow::Int16Type>",
                             "arrow::internal::PrimitiveScalar<arrow::Int32Type>",
                             "arrow::internal::PrimitiveScalar<arrow::Int64Type>",
                             "arrow::internal::PrimitiveScalar<arrow::UInt8Type>",
                             "arrow::internal::PrimitiveScalar<arrow::UInt16Type>",
                             "arrow::internal::PrimitiveScalar<arrow::UInt32Type>",
                             "arrow::internal::PrimitiveScalar<arrow::UInt64Type>",
                             "arrow::internal::PrimitiveScalar<arrow::HalfFloatType>",
                             "arrow::internal::PrimitiveScalar<arrow::FloatType>",
                             "arrow::internal::PrimitiveScalar<arrow::DoubleType>").cast().pointerTypes("PrimitiveScalarBase"))
               .put(new Info("arrow::compute::detail::FunctionImpl<arrow::compute::Kernel>",
                             "arrow::compute::detail::FunctionImpl<arrow::compute::ScalarKernel>",
                             "arrow::compute::detail::FunctionImpl<arrow::compute::ScalarAggregateKernel>",
                             "arrow::compute::detail::FunctionImpl<arrow::compute::HashAggregateKernel>",
                             "arrow::compute::detail::FunctionImpl<arrow::compute::VectorKernel>").pointerTypes("Function"))
               .put(new Info("arrow::internal::DictionaryScalar", "arrow::NullBuilder::Append(std::nullptr_t)", "arrow::Datum::value",
                             "arrow::compute::MakeCount", "arrow::internal::parallel_memcopy", "arrow::io::HdfsReadableFile::set_memory_pool",
                             "arrow::compute::MakeCompareKernel", "arrow::internal::InvertBitmap", "arrow::fs::FileSystemFromUri",
                             "arrow::io::BufferReader::ReadAsync", "arrow::io::MemoryMappedFile::ReadAsync", "arrow::io::RandomAccessFile::ReadAsync",
                             "arrow::ipc::DictionaryMemo(arrow::ipc::DictionaryMemo)", "arrow::ipc::DictionaryMemo::operator =",
                             "arrow::ipc::WriteMessage", "arrow::json::Convert", "arrow::csv::ConvertOptions::timestamp_parsers",
                             "arrow::ChunkedArray(arrow::ChunkedArray)", "arrow::ChunkedArray::operator =", "arrow::FutureImpl", "arrow::StopSourceImpl",
                             "arrow::AllocateBitmap", "arrow::ConcatenateBuffers", "arrow::FieldPath::Get", "arrow::GetCastFunction",
                             "arrow::compute::CastFunction", "arrow::compute::MinMax", "arrow::ipc::internal::CheckCompressionSupported",
                             "arrow::compute::KnownFieldValues", "arrow::Result<arrow::compute::KnownFieldValues>",
                             "arrow::compute::ScalarAggregateKernel::MergeAll").skip())
               .put(new Info("arrow::Datum::type").enumerate(false).valueTypes("int"))
               .put(new Info("arrow::Endianness").enumerate().valueTypes("Endianness", "@Cast(\"arrow::Endianness\") int"))

               .put(new Info("arrow::NumericBuilder<arrow::Int8Type>").pointerTypes("Int8Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::Int16Type>").pointerTypes("Int16Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::Int32Type>").pointerTypes("Int32Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::Int64Type>").pointerTypes("Int64Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::UInt8Type>").pointerTypes("UInt8Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::UInt16Type>").pointerTypes("UInt16Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::UInt32Type>").pointerTypes("UInt32Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::UInt64Type>").pointerTypes("UInt64Builder").define())
               .put(new Info("arrow::NumericBuilder<arrow::HalfFloatType>").pointerTypes("HalfFloatBuilder").define())
               .put(new Info("arrow::NumericBuilder<arrow::FloatType>").pointerTypes("FloatBuilder").define())
               .put(new Info("arrow::NumericBuilder<arrow::DoubleType>").pointerTypes("DoubleBuilder").define())
               .put(new Info("arrow::NumericBuilder<arrow::DayTimeIntervalType>").pointerTypes("DayTimeIntervalBuilder").define())
               .put(new Info("arrow::NumericBuilder<arrow::MonthDayNanoIntervalType>").pointerTypes("MonthDayNanoIntervalBuilder").define())

               .put(new Info("arrow::io::internal::SharedLockGuard<arrow::io::internal::SharedExclusiveChecker>").pointerTypes("SharedExclusiveCheckerSharedLockGuard").define())
               .put(new Info("arrow::io::internal::ExclusiveLockGuard<arrow::io::internal::SharedExclusiveChecker>").pointerTypes("SharedExclusiveCheckerExclusiveLockGuard").define())
               .put(new Info("arrow::io::internal::RandomAccessFileConcurrencyWrapper<arrow::io::ReadableFile>",
                             "arrow::io::internal::RandomAccessFileConcurrencyWrapper<ReadableFile>").pointerTypes("ReadableFileRandomAccessFileConcurrencyWrapper").define().purify())
               .put(new Info("arrow::io::internal::RandomAccessFileConcurrencyWrapper<arrow::io::BufferReader>",
                             "arrow::io::internal::RandomAccessFileConcurrencyWrapper<BufferReader>").pointerTypes("BufferReaderRandomAccessFileConcurrencyWrapper").define().purify())
               .put(new Info("arrow::io::internal::InputStreamConcurrencyWrapper<arrow::io::BufferedInputStream>",
                             "arrow::io::internal::InputStreamConcurrencyWrapper<BufferedInputStream>").pointerTypes("BufferedInputStreamConcurrencyWrapper").define().purify())
               .put(new Info("arrow::io::internal::InputStreamConcurrencyWrapper<arrow::io::CompressedInputStream>",
                             "arrow::io::internal::InputStreamConcurrencyWrapper<CompressedInputStream>").pointerTypes("CompressedInputStreamConcurrencyWrapper").define().purify())
               .put(new Info("arrow::io::SlowInputStreamBase<arrow::io::InputStream>").pointerTypes("InputStreamSlowInputStreamBase").define().purify())
               .put(new Info("arrow::io::SlowInputStreamBase<arrow::io::RandomAccessFile>").pointerTypes("RandomAccessFileSlowInputStreamBase").define().purify())
               .put(new Info("arrow::io::FileSystem").pointerTypes("IOFileSystem"))
               .put(new Info("arrow::csv::ParseOptions").pointerTypes("CsvParseOptions"))
               .put(new Info("arrow::json::ParseOptions").pointerTypes("JsonParseOptions"))

               .put(new Info("arrow::Buffer::FromString(std::string)").javaText(
                       "public static native @SharedPtr @ByVal ArrowBuffer FromString(@Cast({\"\", \"std::string&&\"}) @StdString BytePointer data);\n"
                     + "public static native @SharedPtr @ByVal ArrowBuffer FromString(@Cast({\"\", \"std::string&&\"}) @StdString String data);\n"))
               .put(new Info("arrow::Decimal128(const std::string&)").javaText(
                       "public Decimal128(@StdString String value) { super((Pointer)null); allocate(value); }\n"
                     + "private native void allocate(@StdString String value);\n"))
               .put(new Info("arrow::Decimal256(const std::string&)").javaText(
                       "public Decimal256(@StdString String value) { super((Pointer)null); allocate(value); }\n"
                     + "private native void allocate(@StdString String value);\n"))
        ;
    }
}
