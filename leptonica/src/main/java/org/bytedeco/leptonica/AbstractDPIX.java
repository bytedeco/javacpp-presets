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
public abstract class AbstractDPIX extends Pointer implements Indexable {
    protected static class DestroyDeallocator extends DPIX implements Pointer.Deallocator {
        DestroyDeallocator(DPIX p) { super(p); }
        @Override public void deallocate() { if (isNull()) return; dpixDestroy(this); setNull(); }
    }

    public AbstractDPIX(Pointer p) { super(p); }

    /**
     * Calls dpixCreate(), and registers a deallocator.
     * @return DPIX created. Do not call dpixDestroy() on it.
     */
    public static DPIX create(int width, int height) {
        DPIX p = dpixCreate(width, height);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls dpixCreateTemplate(), and registers a deallocator.
     * @return DPIX created. Do not call dpixDestroy() on it.
     */
    public static DPIX createTemplate(DPIX dpixs) {
        DPIX p = dpixCreateTemplate(dpixs);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls dpixClone(), and registers a deallocator.
     * @return DPIX cloned. Do not call dpixDestroy() on it.
     */
    @Override public DPIX clone() {
        // make sure we return a new object
        DPIX p = new DPIX(dpixClone((DPIX)this));
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /** @return {@code createBuffer(0)} */
    public DoubleBuffer createBuffer() {
        return createBuffer(0);
    }
    /** @return {@link DPIX#data()} wrapped in a {@link DoubleBuffer} starting at given byte index. */
    public DoubleBuffer createBuffer(int index) {
        int[] w = {0}, h = {0};
        dpixGetDimensions((DPIX)this, w, h);
        int wpl = dpixGetWpl((DPIX)this);
        DoublePointer data = new DoublePointer(dpixGetData((DPIX)this)).position(index).capacity(h[0] * wpl);
        return data.asBuffer();
    }

    /** @return {@code createIndexer(true)} */
    public DoubleIndexer createIndexer() {
        return createIndexer(true);
    }
    @Override public DoubleIndexer createIndexer(boolean direct) {
        int[] w = {0}, h = {0};
        dpixGetDimensions((DPIX)this, w, h);
        int wpl = dpixGetWpl((DPIX)this);
        long[] sizes = {h[0], w[0], wpl / w[0]};
        long[] strides = {wpl, wpl / w[0], 1};
        DoublePointer data = new DoublePointer(dpixGetData((DPIX)this)).capacity(h[0] * wpl);
        return DoubleIndexer.create(data, sizes, strides, direct);
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void destroy() {
        deallocate();
    }
}
