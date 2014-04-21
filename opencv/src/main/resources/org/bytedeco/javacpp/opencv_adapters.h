/*
 * Copyright (C) 2011,2012,2014 Samuel Audet
 *
 * This file is part of JavaCV.
 *
 * JavaCV is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCV is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCV.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif
static inline void SetLibraryPath(const char *path) {
#if _WIN32_WINNT >= 0x0502
    SetDllDirectory(path);
#endif
}

template<class T> class PtrAdapter {
public:
    PtrAdapter(const T* ptr, int size)  : ptr((T*)ptr), size(size), cvPtr(cvPtr2) {
            cvPtr2.obj = (T*)ptr; cvPtr2.refcount = 0; }
    PtrAdapter(const cv::Ptr<T>& cvPtr) : ptr(0), size(0), cvPtr2(cvPtr), cvPtr(cvPtr2) { }
    PtrAdapter(      cv::Ptr<T>& cvPtr) : ptr(0), size(0), cvPtr(cvPtr) { }
    void assign(T* ptr, int size) {
        this->ptr = ptr;
        this->size = size;
        this->cvPtr = ptr;
    }
    static void deallocate(void* ptr) { cv::Ptr<T> deallocator((T*)ptr); }
    operator T*() {
        // take ownership
        ptr = cvPtr.obj;
        cvPtr.obj = 0;
        return ptr;
    }
    operator const T*()    { return (const T*)cvPtr; }
    operator cv::Ptr<T>&() { return cvPtr; }
    operator cv::Ptr<T>*() { return ptr ? &cvPtr : 0; }
    T* ptr;
    int size;
    cv::Ptr<T> cvPtr2;
    cv::Ptr<T>& cvPtr;
};

