/*
 * Copyright (C) 2018-2019 Samuel Audet
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

package org.bytedeco.leptonica;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.leptonica.global.lept.*;

@Properties(inherit = org.bytedeco.leptonica.presets.lept.class)
public abstract class AbstractFPIX extends Pointer implements Indexable {
    protected static class DestroyDeallocator extends FPIX implements Pointer.Deallocator {
        DestroyDeallocator(FPIX p) { super(p); }
        @Override public void deallocate() { if (isNull()) return; fpixDestroy(this); setNull(); }
    }

    public AbstractFPIX(Pointer p) { super(p); }

    /**
     * Calls fpixCreate(), and registers a deallocator.
     * @return FPIX created. Do not call fpixDestroy() on it.
     */
    public static FPIX create(int width, int height) {
        FPIX p = fpixCreate(width, height);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls fpixCreateTemplate(), and registers a deallocator.
     * @return FPIX created. Do not call fpixDestroy() on it.
     */
    public static FPIX createTemplate(FPIX fpixs) {
        FPIX p = fpixCreateTemplate(fpixs);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls fpixClone(), and registers a deallocator.
     * @return FPIX cloned. Do not call fpixDestroy() on it.
     */
    @Override public FPIX clone() {
        // make sure we return a new object
        FPIX p = new FPIX(fpixClone((FPIX)this));
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /** @return {@code createBuffer(0)} */
    public FloatBuffer createBuffer() {
        return createBuffer(0);
    }
    /** @return {@link FPIX#data()} wrapped in a {@link FloatBuffer} starting at given byte index. */
    public FloatBuffer createBuffer(int index) {
        int[] w = {0}, h = {0};
        fpixGetDimensions((FPIX)this, w, h);
        int wpl = fpixGetWpl((FPIX)this);
        FloatPointer data = new FloatPointer(fpixGetData((FPIX)this)).position(index).capacity(h[0] * wpl);
        return data.asBuffer();
    }

    /** @return {@code createIndexer(true)} */
    public FloatIndexer createIndexer() {
        return createIndexer(true);
    }
    @Override public FloatIndexer createIndexer(boolean direct) {
        int[] w = {0}, h = {0};
        fpixGetDimensions((FPIX)this, w, h);
        int wpl = fpixGetWpl((FPIX)this);
        long[] sizes = {h[0], w[0], wpl / w[0]};
        long[] strides = {wpl, wpl / w[0], 1};
        FloatPointer data = new FloatPointer(fpixGetData((FPIX)this)).capacity(h[0] * wpl);
        return FloatIndexer.create(data, sizes, strides, direct);
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void destroy() {
        deallocate();
    }
}
