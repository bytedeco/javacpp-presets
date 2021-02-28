/*
 * Copyright (C) 2021 Mats Larsen
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

#ifndef NAMED_METADATA_OPERATIONS_H
#define NAMED_METADATA_OPERATIONS_H

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm-c/Core.h"
#include "llvm-c/Types.h"

namespace llvm {

/**
 * Exact re-implementation of LLVMGetNamedMetadataNumOperands without providing
 * a LLVMModuleRef parameter
 *
 * Pass a LLVMNamedMDNodeRef instead, which is the true owner for the data
 * anyways
 *
 * See /llvm/lib/IR/Core.cpp for original implementation
 */
extern "C" unsigned getNamedMDNodeNumOperands(LLVMNamedMDNodeRef NodeRef) {
    NamedMDNode *N = unwrap(NodeRef);
    return N->getNumOperands();
}

/**
 * Exact re-implementation of LLVMGetNamedMetadataOperands without providing
 * a LLVMModuleRef parameter.
 *
 * This requires a LLVMContextRef instance because conversion from Metadata to
 * Value requires a context in which the new Value will reside in.
 *
 * See /llvm/lib/IR/Core.cpp for original implementation
 */
extern "C" void getNamedMDNodeOperands(
    LLVMNamedMDNodeRef NodeRef,
    LLVMValueRef *Dest,
    LLVMContextRef InContext
) {
    NamedMDNode *N = unwrap(NodeRef);
    LLVMContext *C = unwrap(InContext);
    for (unsigned i = 0; i < N->getNumOperands(); i++) {
        Dest[i] = wrap(MetadataAsValue::get(*C, N->getOperand(i)));
    }
}

/**
 * Inlined re-implementation of LLVMAddNamedMetadataOperand without providing
 * a LLVMModuleRef parameter.
 *
 * This implementation inlines the statically defined "extractMDNode" in
 * Core.cpp which the original implementation uses.
 *
 * See /llvm/lib/IR/Core.cpp for original implementation
 */
extern "C" void addNamedMDNodeOperand(
    LLVMNamedMDNodeRef NodeRef,
    LLVMValueRef Val
) {
    assert(Val && "Expected not-null value Val");

    NamedMDNode *N = unwrap(NodeRef);
    MetadataAsValue *MAV = unwrap<MetadataAsValue>(Val);
    Metadata *MD = MAV->getMetadata();
    MDNode* Metadata;
    assert((isa<MDNode>(MD) || isa<ConstantAsMetadata>(MD)) && "Expected a metadata node or a canonicalized constant");

    if (MDNode* NN = dyn_cast<MDNode>(MD)) {
        Metadata = NN;
    }
    Metadata = MDNode::get(MAV->getContext(), MD);

    N->addOperand(Metadata);
}

/**
 * Inlined re-implementation of LLVMGetMDString without providing a wrapped
 * MetadataAsValue instance
 *
 * See /llvm/lib/IR/Core.cpp for original implementation
 */
extern "C" const char* getMDString(LLVMMetadataRef M, unsigned *Length) {
    Metadata *MD = unwrap(M);

    if (const MDString *S = dyn_cast<MDString>(MD)) {
        *Length = S->getString().size();
        return S->getString().data();
    }
    *Length = 0;
    return nullptr;
}

/**
 * Inlined re-implementation of LLVMGetMDNodeNumOperands
 *
 * See /llvm/lib/IR/Core.cpp for original implementation
 */
extern "C" unsigned getMDNodeNumOperands(LLVMMetadataRef M) {
    Metadata *MD = unwrap(M);
    if (isa<ValueAsMetadata>(MD)) {
        return 1;
    }
    return cast<MDNode>(MD)->getNumOperands();
}

/**
 * Inlined re-implementation of LLVMGetMDNodeOperands
 *
 * Accepts an additional LLVMContextRef argument in which all ConstantAsMetadata
 * values will be unwrapped and stored in. (see C API implementation)
 *
 * This implementation inlines the statically defined "getMDNodeOperandsImpl" in
 * Core.cpp which the original implementation uses.
 *
 * See /llvm/lib/IR/Core.cpp for original implementation
 */
extern "C" void getMDNodeOperands(
    LLVMMetadataRef M,
    LLVMContextRef C,
    LLVMValueRef *Dest) {
    Metadata *MD = unwrap(M);
    LLVMContext *Context = unwrap(C);

    if (auto *MDV = dyn_cast<ValueAsMetadata>(MD)) {
        *Dest = wrap(MDV->getValue());
        return;
    }

    const auto *N = cast<MDNode>(MD);
    const unsigned numOperands = N->getNumOperands();

    // Inlined code of "static LLVMValueRef getMDNodeOperandsImpl(LLVMContext &, const MDNode *, unsigned)"
    for (unsigned i = 0; i < numOperands; i++) {
        Metadata *Op = N->getOperand(i);

        if (!Op) {
            Dest[i] = nullptr;
            continue;
        }

        if (auto *C = dyn_cast<ConstantAsMetadata>(Op)) {
            Dest[i] = wrap(C->getValue());
            continue;
        }

        Dest[i] = wrap(MetadataAsValue::get(*Context, Op));
    }
}

} // namespace llvm

#endif // NAMED_METADATA_OPERATIONS_H