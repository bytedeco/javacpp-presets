#!/bin/bash

cd ..
./cppbuild.sh -platform $1 install helloworld
mvn install -Djavacpp.cppbuild.skip=true -Djavacpp.platform=$1 --projects .,helloworld
cd helloworld
