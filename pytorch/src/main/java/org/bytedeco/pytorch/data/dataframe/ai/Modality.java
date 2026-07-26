package org.bytedeco.pytorch.data.dataframe.ai;

/**
 * Input modality for an {@link EmbeddingModel}.
 */
public enum Modality {
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    MULTIMODAL,  // accepts mixed / already-tensor inputs
    TENSOR
}
