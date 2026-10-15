package org.bytedeco.pytorchvision;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.presets.torch;

/**
 * This is only a placeholder to facilitate loading the {@code torch_vision}
 * module with JavaCPP.
 * <p>
 * Call {@code Loader.load(thorch_vision.class)} before loading torch models
 * that use thorch_vision.
 *
 * @author Ivo Lucas
 */
@Properties(
        inherit = {
            torch.class

        },
        value = {
            @Platform(preload = {"torchvision"})
        }
)
public class torch_vision {

    static {
        Loader.load();
    }
}
