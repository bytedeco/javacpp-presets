/*
 * Copyright (C) 2015-2019 Samuel Audet
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

package org.bytedeco.tensorflow;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.tensorflow.global.tensorflow.*;

@Properties(inherit = org.bytedeco.tensorflow.presets.tensorflow.class)
public abstract class AbstractSession extends Pointer {
    static { Loader.load(); }

    SessionOptions options; // a reference to prevent deallocation

    public AbstractSession(Pointer p) { super(p); }
    /** Calls {@link org.bytedeco.javacpp.tensorflow#NewSession(SessionOptions)} and registers a deallocator. */
    public AbstractSession(SessionOptions options) {
        this.options = options;
        if (NewSession(options, (Session)this).ok() && !isNull()) {
            deallocator(new DeleteDeallocator((Session)this));
        }
    }

    @Namespace public static native void delete(Session session);

    protected static class DeleteDeallocator extends Session implements Pointer.Deallocator {
        DeleteDeallocator(Session p) { super(p); }
        @Override public void deallocate() { if (!isNull()) Session.delete(this); setNull(); }
    }
}
