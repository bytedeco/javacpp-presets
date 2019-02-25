/*
 * Copyright (C) 2015 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License
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

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/core/stringpiece.h"

using namespace tensorflow;
using namespace tensorflow::gtl;

template<typename T> class ArraySliceAdapter {
public:
    ArraySliceAdapter(T const * ptr, typename ArraySlice<T>::size_type size, void* owner) : ptr((T*)ptr), size(size), owner(owner),
        arr2(ptr ? ArraySlice<T>((T*)ptr, size) : ArraySlice<T>()), arr(arr2) { }
    ArraySliceAdapter(const ArraySlice<T>& arr) : ptr(0), size(0), owner(0), arr2(arr), arr(arr2) { }
    ArraySliceAdapter(      ArraySlice<T>& arr) : ptr(0), size(0), owner(0), arr(arr) { }
    ArraySliceAdapter(const ArraySlice<T>* arr) : ptr(0), size(0), owner(0), arr(*(ArraySlice<T>*)arr) { }
    void assign(T* ptr, typename ArraySlice<T>::size_type size, void* owner) {
        this->ptr = ptr;
        this->size = size;
        this->owner = owner;
        arr.set(ptr, size);
    }
    static void deallocate(void* owner) { free(owner); }
    operator T*()             { size = arr.size(); return (T*)arr.data(); }
    operator ArraySlice<T>&() { return arr; }
    operator ArraySlice<T>*() { return ptr ? &arr : 0; }
    T* ptr;
    typename ArraySlice<T>::size_type size;
    void* owner;
    ArraySlice<T> arr2;
    ArraySlice<T>& arr;
};

class StringPieceAdapter {
public:
    StringPieceAdapter(const          char* ptr, size_t size, void* owner) : ptr((char*)ptr), size(size), owner(owner),
        str2(ptr ? (char*)ptr : "", ptr ? (size > 0 ? size : strlen((char*)ptr)) : 0), str(str2) { }
    StringPieceAdapter(const signed   char* ptr, size_t size, void* owner) : ptr((char*)ptr), size(size), owner(owner),
        str2(ptr ? (char*)ptr : "", ptr ? (size > 0 ? size : strlen((char*)ptr)) : 0), str(str2) { }
    StringPieceAdapter(const unsigned char* ptr, size_t size, void* owner) : ptr((char*)ptr), size(size), owner(owner),
        str2(ptr ? (char*)ptr : "", ptr ? (size > 0 ? size : strlen((char*)ptr)) : 0), str(str2) { }
    StringPieceAdapter(const StringPiece& str) : ptr(0), size(0), owner(0), str2(str), str(str2) { }
    StringPieceAdapter(      StringPiece& str) : ptr(0), size(0), owner(0), str(str) { }
    StringPieceAdapter(const StringPiece* str) : ptr(0), size(0), owner(0), str(*(StringPiece*)str) { }
    void assign(char* ptr, size_t size, void* owner) {
        this->ptr = ptr;
        this->size = size;
        this->owner = owner;
        str = StringPiece(ptr ? ptr : "", ptr ? (size > 0 ? size : strlen((char*)ptr)) : 0);
    }
    static void deallocate(void* owner) { free(owner); }
    operator          char*() { size = str.size(); return (         char*)str.data(); }
    operator signed   char*() { size = str.size(); return (signed   char*)str.data(); }
    operator unsigned char*() { size = str.size(); return (unsigned char*)str.data(); }
    operator   StringPiece&() { return str; }
    operator   StringPiece*() { return ptr ? &str : 0; }
    char* ptr;
    size_t size;
    void* owner;
    StringPiece str2;
    StringPiece& str;
};
