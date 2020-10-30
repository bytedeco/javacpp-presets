/*
 * Copyright (C) 2020 Samuel Audet
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

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
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
    inherit = parquet.class,
    value = {
        @Platform(
            include = {
                "arrow/util/iterator.h",
                "arrow/result.h",
                "arrow/dataset/api.h",
                "arrow/dataset/visibility.h",
                "arrow/dataset/type_fwd.h",
                "arrow/dataset/dataset.h",
                "arrow/dataset/discovery.h",
                "arrow/dataset/scanner.h",
//                "arrow/dataset/writer.h",
                "arrow/dataset/file_base.h",
                "arrow/dataset/file_ipc.h",
                "arrow/dataset/file_parquet.h",
                "arrow/dataset/filter.h",
            },
            link = "arrow_dataset@.200"
        ),
    },
    target = "org.bytedeco.arrow_dataset",
    global = "org.bytedeco.arrow.global.arrow_dataset"
)
public class arrow_dataset implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "arrow_dataset"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("ARROW_DS_EXPORT").cppTypes().annotations())
               .put(new Info("std::enable_shared_from_this<arrow::dataset::Dataset>",
                             "std::enable_shared_from_this<arrow::dataset::FileFormat>",
                             "arrow::dataset::FileSource::CustomOpenWithCompression",
                             "arrow::util::EqualityComparable<arrow::dataset::RowGroupInfo>",
                             "arrow::util::EqualityComparable<arrow::Iterator<std::shared_ptr<arrow::dataset::Fragment> > >",
                             "arrow::util::EqualityComparable<arrow::Iterator<std::unique_ptr<arrow::dataset::ScanTask> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::Dataset> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::DatasetFactory> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::Expression> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::FileSystemDataset> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::FileFragment> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::FileWriter> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::Fragment> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::Scanner> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::ScannerBuilder> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<arrow::dataset::ScanTask> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::unique_ptr<parquet::arrow::FileReader> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::Source> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::SourceFactory> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::UnionDataset> > >",
//                             "arrow::util::EqualityComparable<arrow::Result<std::shared_ptr<arrow::dataset::WriteTask> > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<std::shared_ptr<arrow::dataset::Fragment> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<std::vector<std::unique_ptr<arrow::dataset::ScanTask> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::Iterator<std::shared_ptr<arrow::dataset::Fragment> > > >",
                             "arrow::util::EqualityComparable<arrow::Result<arrow::dataset::ScanTaskIterator> >").cast().pointerTypes("Pointer"))
               .put(new Info("std::unordered_set<std::string>").pointerTypes("StringUnorderedSet").define())
               .put(new Info("std::shared_ptr<arrow::dataset::Dataset>").annotations("@SharedPtr").pointerTypes("Dataset"))
               .put(new Info("std::shared_ptr<arrow::dataset::DatasetFactory>").annotations("@SharedPtr").pointerTypes("DatasetFactory"))
//               .put(new Info("std::shared_ptr<arrow::dataset::DataSource>").annotations("@SharedPtr").pointerTypes("DataSource"))
               .put(new Info("std::shared_ptr<arrow::dataset::Expression>").annotations("@SharedPtr").pointerTypes("Expression"))
               .put(new Info("std::shared_ptr<arrow::dataset::FileFragment>").annotations("@SharedPtr").pointerTypes("FileFragment"))
               .put(new Info("std::shared_ptr<arrow::dataset::Fragment>").annotations("@SharedPtr").pointerTypes("Fragment"))
//               .put(new Info("std::shared_ptr<arrow::dataset::FileScanOptions>").annotations("@SharedPtr").pointerTypes("FileScanOptions"))
               .put(new Info("std::shared_ptr<arrow::dataset::Scanner>").annotations("@SharedPtr").pointerTypes("Scanner"))
               .put(new Info("std::shared_ptr<arrow::dataset::ScannerBuilder>").annotations("@SharedPtr").pointerTypes("ScannerBuilder"))
               .put(new Info("std::shared_ptr<arrow::dataset::ScanTask>").annotations("@SharedPtr").pointerTypes("ScanTask"))
               .put(new Info("std::unique_ptr<arrow::dataset::ScanTask>").annotations("@UniquePtr").pointerTypes("ScanTask"))
               .put(new Info("std::unique_ptr<parquet::arrow::FileReader>").annotations("@UniquePtr").pointerTypes("FileReader"))
               .put(new Info("std::shared_ptr<arrow::dataset::Source>").annotations("@SharedPtr").pointerTypes("Source"))
               .put(new Info("std::shared_ptr<arrow::dataset::SourceFactory>").annotations("@SharedPtr").pointerTypes("SourceFactory"))
               .put(new Info("std::shared_ptr<arrow::dataset::UnionDataset>").annotations("@SharedPtr").pointerTypes("UnionDataset"))
//               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::DataSource> >").pointerTypes("DataSourceVector").define())
//               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::FileScanOptions> >").pointerTypes("FileScanOptionsVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::Expression> >").pointerTypes("ExpressionVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::FileFragment> >").pointerTypes("FileFragmentVector").define())
               .put(new Info("std::vector<std::unique_ptr<arrow::dataset::ScanTask> >").pointerTypes("ScanTaskVector").define())
//               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::SourceFactory> >").pointerTypes("SourceFactoryVector").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::Dataset> >").pointerTypes("DatasetResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::DatasetFactory> >").pointerTypes("DatasetFactoryResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::Expression> >").pointerTypes("ExpressionResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::FileSystemDataset> >").pointerTypes("FileSystemDatasetResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::FileFragment> >").pointerTypes("FileFragmentResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::FileWriter> >").pointerTypes("FileWriterResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::Fragment> >").pointerTypes("FragmentResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::Scanner> >").pointerTypes("ScannerResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::ScannerBuilder> >").pointerTypes("ScannerBuilderResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::dataset::ScanTask> >").pointerTypes("ScanTaskResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<arrow::dataset::ScanTask> >(const arrow::Result<std::unique_ptr<arrow::dataset::ScanTask> >&)",
                             "arrow::Result<std::unique_ptr<arrow::dataset::ScanTask> >::operator =").skip())
               .put(new Info("arrow::Result<std::unique_ptr<parquet::arrow::FileReader> >").pointerTypes("FileReaderResult").define())
               .put(new Info("arrow::Result<std::unique_ptr<parquet::arrow::FileReader> >(const arrow::Result<std::unique_ptr<parquet::arrow::FileReader> >&)",
                             "arrow::Result<std::unique_ptr<parquet::arrow::FileReader> >::operator =").skip())

//               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::Source> >").pointerTypes("SourceResult").define())
//               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::SourceFactory> >").pointerTypes("SourceFactoryResult").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::UnionDataset> >").pointerTypes("UnionDatasetResult").define())
//               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::WriteTask> >").pointerTypes("WriteTaskResult").define())
               .put(new Info("arrow::Result<arrow::Iterator<std::shared_ptr<arrow::dataset::Fragment> > >", "arrow::Result<arrow::dataset::FragmentIterator>").pointerTypes("FragmentIteratorResult").define())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::dataset::Fragment> >").pointerTypes("FragmentIterator").define())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::dataset::Fragment> >::RangeIterator").pointerTypes("FragmentIterator.RangeIterator").define())
               .put(new Info("arrow::Result<arrow::dataset::ScanTaskIterator>").pointerTypes("ScanTaskIteratorResult").define())
//               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::dataset::DataFragment> >").pointerTypes("DataFragmentIterator").define())
               .put(new Info("arrow::Iterator<std::unique_ptr<arrow::dataset::ScanTask> >").pointerTypes("ScanTaskIterator").define())
               .put(new Info("arrow::Iterator<std::unique_ptr<arrow::dataset::ScanTask> >::RangeIterator").pointerTypes("ScanTaskIterator.RangeIterator").define())
               .put(new Info("arrow::Result<std::vector<std::unique_ptr<arrow::dataset::ScanTask> > >").pointerTypes("ScanTaskVectorResult").define())
               .put(new Info("arrow::Result<std::vector<std::shared_ptr<arrow::dataset::Fragment> > >").pointerTypes("FragmentVectorResult").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::UnaryExpression,arrow::dataset::CastExpression,arrow::dataset::ExpressionType::CAST>").pointerTypes("CastExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::UnaryExpression,arrow::dataset::InExpression,arrow::dataset::ExpressionType::IN>").pointerTypes("InExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::UnaryExpression,arrow::dataset::IsValidExpression,arrow::dataset::ExpressionType::IS_VALID>").pointerTypes("IsValidExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::UnaryExpression,arrow::dataset::NotExpression,arrow::dataset::ExpressionType::NOT>",
                             "arrow::dataset::ExpressionImpl::ExpressionImpl<arrow::dataset::UnaryExpression,NotExpression,arrow::dataset::ExpressionType::NOT>").pointerTypes("NotExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,arrow::dataset::OrExpression,arrow::dataset::ExpressionType::OR>",
                             "arrow::dataset::ExpressionImpl::ExpressionImpl<arrow::dataset::BinaryExpression,OrExpression,arrow::dataset::ExpressionType::OR>").pointerTypes("OrExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,arrow::dataset::AndExpression,arrow::dataset::ExpressionType::AND>",
                             "arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,AndExpression,arrow::dataset::ExpressionType::AND>").pointerTypes("AndExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,arrow::dataset::ComparisonExpression,arrow::dataset::ExpressionType::COMPARISON>",
                             "arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,ComparisonExpression,arrow::dataset::ExpressionType::COMPARISON>").pointerTypes("ComparisonExpressionImpl").define())
               .put(new Info("arrow::dataset::DiscoverSource", "arrow::dataset::RowGroupStatisticsAsExpression",
                             "arrow::dataset::WritePlan::fragment_or_partition_expressions", "arrow::dataset::string_literals::operator \"\"_")/*.javaNames("quote")*/.skip())
               .put(new Info("std::function<arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile> >(arrow::Compression::type)>",
                             "std::function<arrow::Result<std::shared_ptr<io::RandomAccessFile> >(Compression::type)>").pointerTypes("CustomOpenWithCompression").define())
               .put(new Info("std::function<arrow::Status(const arrow::dataset::Expression&)>",
                             "std::function<arrow::Status(const Expression&)>").pointerTypes("ExpressionVisitor").define())
        ;
    }

    public static class CustomOpenWithCompression extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    CustomOpenWithCompression(Pointer p) { super(p); }
        protected CustomOpenWithCompression() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile> >*") Pointer call(@Cast("arrow::Compression::type") int type);
    }

    public static class ExpressionVisitor extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ExpressionVisitor(Pointer p) { super(p); }
        protected ExpressionVisitor() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("arrow::Status*") Pointer call(@ByRef @Cast("const arrow::dataset::Expression*") Pointer node);
    }
}
