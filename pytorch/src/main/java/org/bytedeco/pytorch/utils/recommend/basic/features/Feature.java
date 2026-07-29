/*
 * Ported from torch-rechub-scala: torchrec/basic/features/Feature.scala
 */
package org.bytedeco.pytorch.utils.recommend.basic.features;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Base interface for all feature types in TorchRec / recommend.
 */
public interface Feature {
    String name();
    int embedDim();
    long vocabSize();

    /** Whether this is a sequence feature */
    default boolean isSequence() {
        return false;
    }
}
