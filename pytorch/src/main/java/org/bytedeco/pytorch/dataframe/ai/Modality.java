package org.bytedeco.pytorch.dataframe.ai;

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
