#!/bin/bash

#
# Travis CI run script
#

set -Eex
set -o pipefail

projects=opencv

for project in $projects; do
  ./cppbuild.sh install $project
  mvn install -pl $project -am
done

