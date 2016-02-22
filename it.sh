#!/bin/bash

#
# Travis CI run script
#

set -Eex
set -o pipefail

wget -c https://github.com/bytedeco/javacpp/archive/master.zip
unzip master.zip
mvn clean install -f javacpp-master

for project in $PROJECTS; do
  ./cppbuild.sh install $project
  mvn install -pl $project -am -amd
done

