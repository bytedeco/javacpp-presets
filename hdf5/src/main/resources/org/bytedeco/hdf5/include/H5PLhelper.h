/*
 * Helpers to call H5PLget_plugin_info() and H5PLget_plugin_type() of a given HDF5 plugin library,
 * for use with H5Zregister(), as the Julia package H5ZzstdC does. All the plugins export the same
 * symbols, so they have to be looked up in one specific library, given by its path.
 * The libraries are intentionally never unloaded, since HDF5 keeps pointing into them.
 */
#ifndef H5PLHELPER_H
#define H5PLHELPER_H

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef const void *(*h5pl_get_plugin_info_t)(void);
typedef int (*h5pl_get_plugin_type_t)(void);

/* Returns the address of the given symbol in the library at the given path, or NULL on failure */
static inline void *h5pl_lookup(const char *path, const char *symbol) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path);
    if (handle == NULL) {
        return NULL;
    }
    return (void *)GetProcAddress(handle, symbol);
#else
    void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle == NULL) {
        return NULL;
    }
    return dlsym(handle, symbol);
#endif
}

/* Returns the result of H5PLget_plugin_info() of the plugin at the given path, or NULL on failure */
static inline const void *h5pl_get_plugin_info(const char *path) {
    h5pl_get_plugin_info_t fn = (h5pl_get_plugin_info_t)h5pl_lookup(path, "H5PLget_plugin_info");
    return fn != NULL ? fn() : NULL;
}

/* Returns the result of H5PLget_plugin_type() (an H5PL_type_t) of the plugin at the given path, or -1 on failure */
static inline int h5pl_get_plugin_type(const char *path) {
    h5pl_get_plugin_type_t fn = (h5pl_get_plugin_type_t)h5pl_lookup(path, "H5PLget_plugin_type");
    return fn != NULL ? fn() : -1;
}

#endif
