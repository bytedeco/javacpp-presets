/*
 * Copyright (C) 2023 Sören Brunk
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

package org.bytedeco.sentencepiece.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

/**
 *
 * @author Sören Brunk
 */
@Properties(
    value = {
        @Platform(
            compiler = "cpp17",
            include = {"<sentencepiece_processor.h>", "<sentencepiece_trainer.h>"},
            link = {"sentencepiece", "sentencepiece_train"}
        ),
        @Platform(value = "windows", link = {"sentencepiece#", "sentencepiece_train#"})
    },
    target = "org.bytedeco.sentencepiece",
    global = "org.bytedeco.sentencepiece.global.sentencepiece"
)
public class sentencepiece implements InfoMapper {
    static {
        Loader.checkVersion("org.bytedeco", "sentencepiece");
    }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("SPP_SWIG_CHECK_AND_THROW").cppTypes().annotations())
            .put(new Info("std::unordered_map<std::string,std::string>").pointerTypes("StringStringMap").define())
            .put(new Info("string_view", "absl::string_view").annotations("@StdString").valueTypes("String").pointerTypes("String"))
            .put(new Info("std::string").annotations("@StdString").valueTypes("String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
            .put(new Info("std::vector<std::string>", "std::vector<absl::string_view>").pointerTypes("StringVector").define())
            .put(new Info("std::vector<int>").pointerTypes("IntVector").define())
            .put(new Info("std::vector<std::pair<std::vector<std::string>,float> >").pointerTypes("StringVectorFloatPairVector").define())
            .put(new Info("std::vector<std::pair<std::vector<int>,float> >").pointerTypes("IntVectorFloatPairVector").define())
            .put(new Info(
                "sentencepiece::ModelInterface",
                "sentencepiece::normalizer::Normalizer",
                "sentencepiece::SentencePieceTrainer::GetNormalizerSpec",
                "sentencepiece::SentencePieceProcessor::SetVocabulary"
            ).skip());
    }
}
