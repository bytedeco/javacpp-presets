#!/bin/bash

cd ..
./cppbuild.sh clean helloworld
mvn clean -Djavacpp.cppbuild.skip=true -Djavacpp.platform=$1 --projects .,helloworld
cd helloworld
