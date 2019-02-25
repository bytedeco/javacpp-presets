/*
 * Copyright (C) 2019 Greg Hart
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

#include <QtCore/QDebug>

#ifdef __APPLE__
#include <pthread.h>
#endif

void QtCore_verifyMainThread() {
#ifdef __APPLE__
    if (!pthread_main_np()) {
        qWarning("\n\n\nWARNING!!\n\n\n"
                 "Qt does not appear to be running on the main thread and will most likely be "
                 "unstable and crash. Please make sure to launch your 'java' command with the "
                 "'-XstartOnFirstThread' command line option. For instance:\n\n"
                 "> java -XstartOnFirstThread org.bytedeco.javacpp.samples.qt.CalculatorExample\n\n");
    }
#endif
}
