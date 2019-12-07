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
public abstract class AbstractPIX extends Pointer implements Indexable {
    protected IntPointer pointer; // a reference to prevent deallocation

    protected static class DestroyDeallocator extends PIX implements Pointer.Deallocator {
        boolean header = false;
        DestroyDeallocator(PIX p) { this(p, false); }
        DestroyDeallocator(PIX p, boolean header) { super(p); this.header = header; }
        @Override public void deallocate() { if (isNull()) return; if (header) this.data(null); pixDestroy(this); setNull(); }
    }

    public AbstractPIX(Pointer p) { super(p); }

    /**
     * Calls pixCreate(), and registers a deallocator.
     * @return PIX created. Do not call pixDestroy() on it.
     */
    public static PIX create(int width, int height, int depth) {
        PIX p = pixCreate(width, height, depth);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls pixCreateNoInit(), and registers a deallocator.
     * @return PIX created. Do not call pixDestroy() on it.
     */
    public static PIX createNoInit(int width, int height, int depth) {
        PIX p = pixCreateNoInit(width, height, depth);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls pixCreateTemplate(), and registers a deallocator.
     * @return PIX created. Do not call pixDestroy() on it.
     */
    public static PIX createTemplate(PIX pixs) {
        PIX p = pixCreateTemplate(pixs);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls pixCreateTemplateNoInit(), and registers a deallocator.
     * @return PIX created. Do not call pixDestroy() on it.
     */
    public static PIX createTemplateNoInit(PIX pixs) {
        PIX p = pixCreateTemplateNoInit(pixs);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /**
     * Calls pixCreateHeader(), and registers a deallocator.
     * @return PIX created. Do not call pixDestroy() on it.
     */
    public static PIX createHeader(int width, int height, int depth) {
        PIX p = pixCreateHeader(width, height, depth);
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p, true));
        }
        return p;
    }

    /**
     * Calls createHeader(), and initializes data, keeping a reference to prevent deallocation.
     * @return PIX created and initialized. Do not call pixDestroy() on it.
     */
    public static PIX create(int width, int height, int depth, Pointer data) {
        PIX p = createHeader(width, height, depth);
        p.data(p.pointer = new IntPointer(data));
        return p;
    }

    /**
     * Calls pixClone(), and registers a deallocator.
     * @return PIX cloned. Do not call pixDestroy() on it.
     */
    @Override public PIX clone() {
        // make sure we return a new object
        PIX p = new PIX(pixClone((PIX)this));
        if (p != null) {
            p.deallocator(new DestroyDeallocator(p));
        }
        return p;
    }

    /** @return {@code createBuffer(0)} */
    public ByteBuffer createBuffer() {
        return createBuffer(0);
    }
    /** @return {@link PIX#data()} wrapped in a {@link ByteBuffer} starting at given byte index. */
    public ByteBuffer createBuffer(int index) {
        int h = pixGetHeight((PIX)this);
        int wpl = pixGetWpl((PIX)this);
        BytePointer data = new BytePointer(pixGetData((PIX)this)).position(index).capacity(h * wpl * 4);
        return data.asByteBuffer();
    }

    /** @return {@code createIndexer(true)} */
    public UByteIndexer createIndexer() {
        return createIndexer(true);
    }
    @Override public UByteIndexer createIndexer(boolean direct) {
        int w = pixGetWidth((PIX)this);
        int h = pixGetHeight((PIX)this);
        int d = pixGetDepth((PIX)this);
        int wpl = pixGetWpl((PIX)this);
        long[] sizes = {h, w, d / 8};
        long[] strides = {wpl * 4, d / 8, 1};
        BytePointer data = new BytePointer(pixGetData((PIX)this)).capacity(h * wpl * 4);
        return UByteIndexer.create(data, sizes, strides, direct);
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void destroy() {
        deallocate();
    }
}
