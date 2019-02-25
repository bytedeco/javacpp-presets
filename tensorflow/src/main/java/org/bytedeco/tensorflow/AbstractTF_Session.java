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
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.tensorflow.global.tensorflow.*;

@Properties(inherit = org.bytedeco.tensorflow.presets.tensorflow.class)
public abstract class AbstractTF_Session extends Pointer {
    protected static class DeleteDeallocator extends TF_Session implements Pointer.Deallocator {
        DeleteDeallocator(TF_Session s) { super(s); }
        @Override public void deallocate() { if (!isNull()) TF_DeleteSession(this, TF_Status.newStatus()); setNull(); }
    }

    /** References to prevent deallocation. */
    protected TF_Graph graph;
    protected TF_SessionOptions opts;
    protected TF_Status status;

    public AbstractTF_Session(Pointer p) { super(p); }

    /**
     * Calls TF_NewSession(), and registers a deallocator.
     * @return TF_Session created. Do not call TF_DeleteSession() on it.
     */
    public static TF_Session newSession(TF_Graph graph, TF_SessionOptions opts, TF_Status status) {
        TF_Session s = TF_NewSession(graph, opts, status);
        if (s != null) {
            s.graph = graph;
            s.opts = opts;
            s.status = status;
            s.deallocator(new DeleteDeallocator(s));
        }
        return s;
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void delete() {
        deallocate();
    }
}
