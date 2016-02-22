#!/bin/bash

#
# Travis CI run script
#

set -Eex
set -o pipefail

wget -c https://github.com/bytedeco/javacpp/archive/master.zip
rm -rf javacpp-master
unzip master.zip
mvn clean install -f javacpp-master

for project in $PROJECTS; do
  ./cppbuild.sh install $project
  if test -d $project-it; then
    mvn install -pl $project-it -am
  else
    mvn install -pl $project -am
  fi
done

