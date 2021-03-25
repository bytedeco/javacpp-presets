package org.bytedeco.nvcodec.samples.util;

import java.util.Vector;

public class VectorEx<T> extends Vector<T> {
    public void resize(int size, T defaultValue) {
        int expendCapacity = size - this.size();

        for (int index = 0; index < expendCapacity; index++) {
            this.add(defaultValue);
        }
    }
}
