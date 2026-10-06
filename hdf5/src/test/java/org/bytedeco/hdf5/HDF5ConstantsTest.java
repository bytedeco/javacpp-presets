/*
 * Copyright (C) 2026 Mark Kittisopikul
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

package org.bytedeco.hdf5;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import hdf.hdf5lib.HDF5Constants;
import org.bytedeco.hdf5.global.hdf5;
import org.bytedeco.javacpp.Loader;
import org.junit.Test;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * Compares every integer constant in {@link hdf5} with the constant of the same name in the
 * official {@link HDF5Constants}, to catch constants that are mapped with the wrong type or value
 * (see issue #1812). Optional, as it is a broad check rather than a targeted one: run it with
 * {@code -Dhdf5.test.allConstants=true}.
 *
 * @author Mark Kittisopikul
 */
public class HDF5ConstantsTest {

    /** Constants that legitimately differ between the two bindings, with the reason. */
    private static final Map<String, String> KNOWN_DIFFERENCES = new HashMap<String, String>();
    static {
        KNOWN_DIFFERENCES.put("H5O_INFO_ALL", "the presets map the API without deprecated symbols"
                + " (H5_NO_DEPRECATED_SYMBOLS), where it excludes H5O_INFO_HDR | H5O_INFO_META_SIZE");
    }

    @Test
    public void allConstantsMatchOfficialBindings() throws java.lang.Exception {
        assumeTrue("Set -Dhdf5.test.allConstants=true to compare all constants",
                Boolean.getBoolean("hdf5.test.allConstants"));

        long start = System.nanoTime();
        Loader.load(hdf5_java.class);
        long loaded = System.nanoTime();

        List<String> mismatches = new ArrayList<String>();
        int compared = 0;
        for (Field field : hdf5.class.getFields()) {
            if (!isIntegerConstant(field)) {
                continue;
            }
            Field official;
            try {
                official = HDF5Constants.class.getField(field.getName());
            } catch (NoSuchFieldException e) {
                continue;
            }
            if (!isIntegerConstant(official)) {
                continue;
            }
            if (KNOWN_DIFFERENCES.containsKey(field.getName())) {
                continue;
            }
            compared++;
            long value = field.getLong(null);
            long officialValue = official.getLong(null);
            if (value != officialValue) {
                mismatches.add(field.getName() + ": " + field.getType() + " " + value
                        + " in org.bytedeco.hdf5, " + official.getType() + " " + officialValue
                        + " in hdf.hdf5lib");
            }
        }
        long done = System.nanoTime();

        System.out.println("HDF5ConstantsTest: compared " + compared + " constants in "
                + (done - loaded) / 1000000 + " ms (plus " + (loaded - start) / 1000000
                + " ms loading hdf5_java), skipped " + KNOWN_DIFFERENCES.size() + " known differences");
        assertTrue("Expected to compare many constants, compared " + compared, compared > 100);
        assertTrue(mismatches.size() + " constants differ:\n" + String.join("\n", mismatches),
                mismatches.isEmpty());
    }

    private static boolean isIntegerConstant(Field field) {
        int modifiers = field.getModifiers();
        return Modifier.isPublic(modifiers) && Modifier.isStatic(modifiers) && Modifier.isFinal(modifiers)
                && (field.getType() == long.class || field.getType() == int.class);
    }
}
