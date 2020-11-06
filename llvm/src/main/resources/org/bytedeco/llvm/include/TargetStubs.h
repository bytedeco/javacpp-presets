/*
 * Copyright (C) 2015-2020 Mats Larsen
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

/**
 * Stubs for all LLVM Target initialization routines
 *
 * See #935 https://github.com/bytedeco/javacpp-presets/issues/935
 */
#define LLVM_TARGET(TargetName) \
void LLVMInitialize##TargetName##TargetInfo(void); \
void LLVMInitialize##TargetName##Target(void); \
void LLVMInitialize##TargetName##TargetMC(void); \
void LLVMInitialize##TargetName##AsmPrinter(void); \
void LLVMInitialize##TargetName##AsmParser(void); \
void LLVMInitialize##TargetName##Disassembler(void);

// Stable Targets
LLVM_TARGET(AArch64)
LLVM_TARGET(AMDGPU)
LLVM_TARGET(ARM)
LLVM_TARGET(AVR)
LLVM_TARGET(BPF)
LLVM_TARGET(Hexagon)
LLVM_TARGET(Lanai)
LLVM_TARGET(MSP430)
LLVM_TARGET(Mips)
LLVM_TARGET(PowerPC)
LLVM_TARGET(RISCV)
LLVM_TARGET(Sparc)
LLVM_TARGET(SystemZ)
LLVM_TARGET(WebAssembly)
LLVM_TARGET(X86)

// XCore Target - Does not ship AsmParser
void LLVMInitializeXCoreTargetInfo(void);
void LLVMInitializeXCoreTarget(void);
void LLVMInitializeXCoreTargetMC(void);
void LLVMInitializeXCoreAsmPrinter(void);
void LLVMInitializeXCoreDisassembler(void);

// NVPTX Target - Does not ship with AsmParser or Disassembler
void LLVMInitializeNVPTXTargetInfo(void);
void LLVMInitializeNVPTXTarget(void);
void LLVMInitializeNVPTXTargetMC(void);
void LLVMInitializeNVPTXAsmPrinter(void);

/*
JavaCPP Presets does currently not build experimental targets.

LLVM_TARGET(ARC)
LLVM_TARGET(VE)
*/
#undef LLVM_TARGET
