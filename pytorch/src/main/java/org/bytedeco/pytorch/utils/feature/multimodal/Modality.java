/*
 * Feature modality kinds for multi-modal feature stores.
 */
package org.bytedeco.pytorch.utils.feature.multimodal;

/** Modality of a feature column / view. */
public enum Modality {
    TABULAR,
    TEXT,
    IMAGE,
    AUDIO,
    EMBEDDING,
    GRAPH,
    SEQUENCE,
    VIDEO
}
