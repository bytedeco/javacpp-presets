/*
 * Copyright (C) 2011,2012,2014,2015 Samuel Audet
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
#define explicit // Make all constructors of Affine3<T> implicit
#include <opencv2/core/affine.hpp>
#undef explicit

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
    PtrAdapter(const T* ptr, int size)  : ptr((T*)ptr), size(size), cvPtr2((T*)ptr), cvPtr(cvPtr2) { }
    PtrAdapter(const cv::Ptr<T>& cvPtr) : ptr(0), size(0), cvPtr2(cvPtr), cvPtr(cvPtr2) { }
    PtrAdapter(      cv::Ptr<T>& cvPtr) : ptr(0), size(0), cvPtr(cvPtr) { }
    void assign(T* ptr, int size) {
        this->ptr = ptr;
        this->size = size;
        this->cvPtr = ptr;
    }
    static void deallocate(void* ptr) { cv::Ptr<T> deallocator((T*)ptr); }
    operator T*() {
        // take ownership, if unique
        ptr = cvPtr.get();
        if (&cvPtr == &cvPtr2) {
            // XXX: this probably causes a small memory leak
            memset(&cvPtr, 0, sizeof(cv::Ptr<T>));
        }
        return ptr;
    }
    operator const T*()    { return cvPtr.get(); }
    operator cv::Ptr<T>&() { return cvPtr; }
    operator cv::Ptr<T>*() { return ptr ? &cvPtr : 0; }
    T* ptr;
    int size;
    cv::Ptr<T> cvPtr2;
    cv::Ptr<T>& cvPtr;
};

class StrAdapter {
public:
    StrAdapter(const          char* ptr, size_t size) : ptr((char*)ptr), size(size),
        str2(ptr ? (char*)ptr : ""), str(str2) { }
    StrAdapter(const signed   char* ptr, size_t size) : ptr((char*)ptr), size(size),
        str2(ptr ? (char*)ptr : ""), str(str2) { }
    StrAdapter(const unsigned char* ptr, size_t size) : ptr((char*)ptr), size(size),
        str2(ptr ? (char*)ptr : ""), str(str2) { }
    StrAdapter(const cv::String& str) : ptr(0), size(0), str2(str), str(str2) { }
    StrAdapter(      cv::String& str) : ptr(0), size(0), str(str) { }
    void assign(char* ptr, size_t size) {
        this->ptr = ptr;
        this->size = size;
        str = ptr ? ptr : "";
    }
    static void deallocate(void* ptr) { free(ptr); }
    operator char*() {
        const char* c_str = str.c_str();
        if (ptr == NULL || strcmp(c_str, ptr) != 0) {
            ptr = strdup(c_str);
        }
        size = strlen(c_str) + 1;
        return ptr;
    }
    operator       signed   char*() { return (signed   char*)(operator char*)(); }
    operator       unsigned char*() { return (unsigned char*)(operator char*)(); }
    operator const          char*() { return                 str.c_str(); }
    operator const signed   char*() { return (signed   char*)str.c_str(); }
    operator const unsigned char*() { return (unsigned char*)str.c_str(); }
    operator         cv::String&() { return str; }
    operator         cv::String*() { return ptr ? &str : 0; }
    char* ptr;
    size_t size;
    cv::String str2;
    cv::String& str;
};
