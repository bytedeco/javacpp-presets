package org.bytedeco.sentencepiece.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

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
            .put(new Info("string_view", "absl::string_view").pointerTypes("@StdString String"))
            .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
            .put(new Info("std::vector<int>").pointerTypes("IntVector").define())
            .put(new Info("std::vector<std::pair<std::vector<std::string>,float> >").pointerTypes("StringVectorFloatPairVector").define())
            .put(new Info("std::vector<std::pair<std::vector<int>,float> >").pointerTypes("IntVectorFloatPairVector").define())
            .put(new Info(
                "sentencepiece::ModelInterface",
                "sentencepiece::normalizer::Normalizer",
                "sentencepiece::SentencePieceTrainer::GetNormalizerSpec",
                "sentencepiece::SentencePieceTrainer::Train",
                "sentencepiece::SentencePieceProcessor::DecodePiecesAsImmutableProto",
                "sentencepiece::SentencePieceProcessor::DecodePiecesAsSerializedProto",
                "sentencepiece::SentencePieceProcessor::DecodePieces",
                "sentencepiece::SentencePieceProcessor::Decode",
                "sentencepiece::SentencePieceProcessor::SetVocabulary"
            ).skip());
    }
}