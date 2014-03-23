if [[ -z "$PLATFORM" ]]; then
    echo "This file is meant to be included by the parent cppbuild.sh script"
    exit 1
fi

VIDEOINPUT_VERSION=update2013
download https://github.com/ofTheo/videoInput/archive/$VIDEOINPUT_VERSION.zip videoInput-$VIDEOINPUT_VERSION.zip

unzip -o videoInput-$VIDEOINPUT_VERSION.zip -d C:/
