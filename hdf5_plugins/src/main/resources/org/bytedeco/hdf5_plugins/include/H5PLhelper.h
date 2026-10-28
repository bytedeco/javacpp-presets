/*
 * Helper to get the filter class of a given HDF5 plugin library, as the Julia
 * package H5ZzstdC does, for use with H5Zregister(). All the plugins export the same
 * H5PLget_plugin_info() symbol, so it has to be looked up in one specific library.
 * The library is intentionally never unloaded, since HDF5 keeps pointing into it.
 */
#ifndef H5PLHELPER_H
#define H5PLHELPER_H

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef const void *(*h5pl_get_plugin_info_t)(void);

/* Returns the result of H5PLget_plugin_info() of the plugin at the given path, or NULL on failure */
static inline const void *h5pl_get_plugin_info(const char *path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path);
    if (handle == NULL) {
        return NULL;
    }
    h5pl_get_plugin_info_t fn = (h5pl_get_plugin_info_t)(void *)GetProcAddress(handle, "H5PLget_plugin_info");
#else
    void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle == NULL) {
        return NULL;
    }
    h5pl_get_plugin_info_t fn = (h5pl_get_plugin_info_t)dlsym(handle, "H5PLget_plugin_info");
#endif
    return fn != NULL ? fn() : NULL;
}

#endif
