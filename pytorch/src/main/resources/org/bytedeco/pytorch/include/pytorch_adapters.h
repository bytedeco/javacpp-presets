class JavaCPP_hidden StringViewAdapter final {
    public:
        using SizeT = typename std::basic_string<char>::size_type;
        StringViewAdapter(const c10::string_view &sv) : ptr(sv.data()), size(sv.size()), svRef((c10::string_view &)sv) {}
        StringViewAdapter(      c10::string_view &sv) : ptr(sv.data()), size(sv.size()), svRef(sv) {}

        StringViewAdapter(const unsigned char *p, SizeT s, void *o) : ptr((const char *) p), size(s > 0 ? s : strlen(ptr)), sv(ptr, size), svRef(sv) { }
        StringViewAdapter(      unsigned char *p, SizeT s, void *o) : ptr((const char *) p), size(s > 0 ? s : strlen(ptr)), sv(ptr, size), svRef(sv) { }
        StringViewAdapter(const   signed char *p, SizeT s, void *o) : ptr((const char *) p), size(s > 0 ? s : strlen(ptr)), sv(ptr, size), svRef(sv) { }
        StringViewAdapter(        signed char *p, SizeT s, void *o) : ptr((const char *) p), size(s > 0 ? s : strlen(ptr)), sv(ptr, size), svRef(sv) { }
        StringViewAdapter(const          char *p, SizeT s, void *o) : ptr(               p), size(s > 0 ? s : strlen(ptr)), sv(ptr, size), svRef(sv) { }
        StringViewAdapter(               char *p, SizeT s, void *o) : ptr((const char *) p), size(s > 0 ? s : strlen(ptr)), sv(ptr, size), svRef(sv) { }

        static void deallocate(void *owner) { }

        operator signed char *() { return (signed char *) ptr; } // Used when a string_view argument is passed as BytePointer
        operator const char *() { return ptr; } // Used when a string_view is returned by a function (as String)

        operator c10::string_view&() { return svRef; }
        operator c10::string_view*() { return &svRef; }

        const char *ptr;
        SizeT size;
        c10::string_view sv;
        c10::string_view &svRef;
        void *owner = NULL;
};