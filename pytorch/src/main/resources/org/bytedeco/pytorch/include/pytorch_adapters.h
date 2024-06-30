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

template<class T, class NullType = c10::detail::intrusive_target_default_null_type<T>> class IntrusivePtrAdapter {
public:
    typedef c10::intrusive_ptr<T,NullType> I;
    IntrusivePtrAdapter(const T* ptr, size_t size, void* owner) : ptr((T*)ptr), size(size), owner(owner),
            intrusivePtr2(owner != NULL && owner != ptr ? *(I*)owner : I::reclaim((T*)ptr)), intrusivePtr(intrusivePtr2) { }
    IntrusivePtrAdapter(const I& intrusivePtr) : ptr(0), size(0), owner(0), intrusivePtr2(intrusivePtr), intrusivePtr(intrusivePtr2) { }
    IntrusivePtrAdapter(      I& intrusivePtr) : ptr(0), size(0), owner(0), intrusivePtr(intrusivePtr) { }
    IntrusivePtrAdapter(const I* intrusivePtr) : ptr(0), size(0), owner(0), intrusivePtr(*(I*)intrusivePtr) { }
    IntrusivePtrAdapter(c10::weak_intrusive_ptr<T> wp) : ptr(0), size(0), owner(0), intrusivePtr2(wp.lock()), intrusivePtr(intrusivePtr2) { }

    void assign(T* ptr, size_t size, void* owner) {
        this->ptr = ptr;
        this->size = size;
        this->owner = owner;
        this->intrusivePtr = owner != NULL && owner != ptr ? *(I*)owner : I((T*)ptr);
    }
    static void deallocate(void* owner) { delete (I*)owner; }

    operator T*() {
        if (ptr == NULL) ptr = intrusivePtr.get();
        return ptr;
    }
    operator T&() {
        if (ptr == NULL) ptr = intrusivePtr.get();
        return *ptr;
    }
    /* Necessary because, without it, assigning an adapter to an optional<I> will
     * pick up the T*() conversion operator which will make the type checking
     * in optional fail for some reason. */
    operator c10::optional<I>() {
        return c10::optional(intrusivePtr);
    }

    operator I&() { return intrusivePtr; }
    operator I*() { return &intrusivePtr; }
    T* ptr;
    size_t size;
    void* owner;
    I intrusivePtr2;
    I& intrusivePtr;
};

template<class T> class WeakPtrAdapter {
public:
    typedef std::shared_ptr<T> S;
    typedef std::weak_ptr<T> W;
    WeakPtrAdapter(const T* ptr, size_t size, void* owner) : ptr((T*)ptr), size(size), owner(owner),
            sharedPtr2(owner != NULL && owner != ptr ? *(S*)owner : S((T*)ptr)), sharedPtr(sharedPtr2) { }
    WeakPtrAdapter(const W& weakPtr) : ptr(0), size(0), owner(0), sharedPtr2(weakPtr.lock()), sharedPtr(sharedPtr2) { }
    WeakPtrAdapter(      W& weakPtr) : ptr(0), size(0), owner(0), sharedPtr2(weakPtr.lock()), sharedPtr(sharedPtr2) { }
    WeakPtrAdapter(const W* weakPtr) : ptr(0), size(0), owner(0), sharedPtr2((*weakPtr).lock()), sharedPtr(sharedPtr2) { }

    void assign(T* ptr, size_t size, void* owner) {
        this->ptr = ptr;
        this->size = size;
        this->owner = owner;
        this->sharedPtr = owner != NULL && owner != ptr ? *(S*)owner : S((T*)ptr);
    }
    static void deallocate(void* owner) { delete (S*)owner; }

    operator typename std::remove_const<T>::type*() {
        ptr = sharedPtr.get();
        if (owner == NULL || owner == ptr) {
          owner = new S(sharedPtr);;
        }
        return (typename std::remove_const<T>::type*)ptr;;
    }

    operator W() { return W(sharedPtr); }
    T* ptr;
    size_t size;
    void* owner;
    S sharedPtr2;
    S& sharedPtr;
};