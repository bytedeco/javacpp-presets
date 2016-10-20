//========================================================================
// GLFW 3.1 - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2010 Camilla Berglund <elmindreda@elmindreda.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// As glfw_config.h.in, this file is used by CMake to produce the
// glfw_config.h configuration header file.  If you are adding a feature
// requiring conditional compilation, this is where to add the macro.
//========================================================================
// As glfw_config.h, this file defines compile-time option macros for a
// specific platform and development environment.  If you are using the
// GLFW CMake files, modify glfw_config.h.in instead of this file.  If you
// are using your own build system, make this file define the appropriate
// macros in whatever way is suitable.
//========================================================================

// Define this to 1 if building GLFW for X11
/* #undef _GLFW_X11 */
// Define this to 1 if building GLFW for Win32
#define _GLFW_WIN32
// Define this to 1 if building GLFW for Cocoa
/* #undef _GLFW_COCOA */
// Define this to 1 if building GLFW for Wayland
/* #undef _GLFW_WAYLAND */
// Define this to 1 if building GLFW for Mir
/* #undef _GLFW_MIR */

// Define this to 1 if building GLFW for EGL
/* #undef _GLFW_EGL */
// Define this to 1 if building GLFW for GLX
/* #undef _GLFW_GLX */
// Define this to 1 if building GLFW for WGL
#define _GLFW_WGL
// Define this to 1 if building GLFW for NSGL
/* #undef _GLFW_NSGL */

// Define this to 1 if building as a shared library / dynamic library / DLL
/* #undef _GLFW_BUILD_DLL */

// Define this to 1 to force use of high-performance GPU on hybrid systems
/* #undef _GLFW_USE_HYBRID_HPG */

// Define this to 1 if the XInput X11 extension is available
/* #undef _GLFW_HAS_XINPUT */
// Define this to 1 if the Xxf86vm X11 extension is available
/* #undef _GLFW_HAS_XF86VM */

// Define this to 1 if glfwInit should change the current directory
/* #undef _GLFW_USE_CHDIR */
// Define this to 1 if glfwCreateWindow should populate the menu bar
/* #undef _GLFW_USE_MENUBAR */
// Define this to 1 if windows should use full resolution on Retina displays
/* #undef _GLFW_USE_RETINA */

// Define this to 1 if using OpenGL as the client library
#define _GLFW_USE_OPENGL
// Define this to 1 if using OpenGL ES 1.1 as the client library
/* #undef _GLFW_USE_GLESV1 */
// Define this to 1 if using OpenGL ES 2.0 as the client library
/* #undef _GLFW_USE_GLESV2 */

