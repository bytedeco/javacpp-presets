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

import org.bytedeco.javacpp.Loader;
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
                "arrow/dataset/writer.h",
                "arrow/dataset/file_base.h",
                "arrow/dataset/file_csv.h",
                "arrow/dataset/file_feather.h",
                "arrow/dataset/filter.h",
            },
            link = "arrow_dataset@.15"
        ),
    },
    target = "org.bytedeco.arrow_dataset",
    global = "org.bytedeco.arrow.global.arrow_dataset"
)
public class arrow_dataset implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "arrow_dataset"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("ARROW_DS_EXPORT").cppTypes().annotations())
               .put(new Info("std::enable_shared_from_this<arrow::dataset::Dataset>").pointerTypes("Pointer"))
               .put(new Info("std::shared_ptr<arrow::dataset::DataSource>").annotations("@SharedPtr").pointerTypes("DataSource"))
               .put(new Info("std::shared_ptr<arrow::dataset::FileScanOptions>").annotations("@SharedPtr").pointerTypes("FileScanOptions"))
               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::DataSource> >").pointerTypes("DataSourceVector").define())
               .put(new Info("std::vector<std::shared_ptr<arrow::dataset::FileScanOptions> >").pointerTypes("FileScanOptionsVector").define())
               .put(new Info("arrow::Result<std::shared_ptr<arrow::dataset::Expression> >").pointerTypes("ExpressionResult").define())
               .put(new Info("arrow::Iterator<std::shared_ptr<arrow::dataset::DataFragment> >").pointerTypes("DataFragmentIterator").define())
               .put(new Info("arrow::Iterator<std::unique_ptr<arrow::dataset::ScanTask> >").pointerTypes("ScanTaskIterator").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::UnaryExpression,arrow::dataset::NotExpression,arrow::dataset::ExpressionType::NOT>").pointerTypes("NotExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,arrow::dataset::OrExpression,arrow::dataset::ExpressionType::OR>").pointerTypes("OrExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,arrow::dataset::AndExpression,arrow::dataset::ExpressionType::AND>").pointerTypes("AndExpressionImpl").define())
               .put(new Info("arrow::dataset::ExpressionImpl<arrow::dataset::BinaryExpression,arrow::dataset::ComparisonExpression,arrow::dataset::ExpressionType::COMPARISON>").pointerTypes("ComparisonExpressionImpl").define())
               .put(new Info("arrow::dataset::DiscoverSource", "arrow::dataset::string_literals::operator \"\"_")/*.javaNames("quote")*/.skip())
        ;
    }
}
