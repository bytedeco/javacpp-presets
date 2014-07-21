if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

case $PLATFORM in
    linux-*)
        if [[ ! -d "/usr/include/flycapture/" ]]; then
            echo "Please install FlyCapture under the default installation directory"
        fi
        ;;
    windows-*)
        if [[ ! -d "/C/Program Files/Point Grey Research/" ]]; then
            echo "Please install FlyCapture under the default installation directory"
        fi
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac
