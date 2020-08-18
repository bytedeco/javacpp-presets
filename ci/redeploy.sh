#!/bin/bash
# Redeploys in one shot main and secondary artifacts to get consistent timestamps among them as required by Gradle, sbt, etc
set -euv

if [[ "$TRAVIS_PULL_REQUEST" != "false" ]] || [[ "$TRAVIS_BRANCH" == "release" ]]; then
    # We are not deploying snapshots
    exit 0
fi

GROUP="org.bytedeco"
REPOSITORY_ID="sonatype-nexus-snapshots"
REPOSITORY_URL="https://oss.sonatype.org/content/repositories/snapshots/"
MAVEN_ARGS="-N -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings $TRAVIS_BUILD_DIR/ci/settings.xml"
REDEPLOY_STATUS=0

cd $TRAVIS_BUILD_DIR
rm -f dependencies.txt
for DIR in *; do
    for POM in $(find $DIR/platform/ -iname pom.xml); do
        JAVACPP_ARGS=
        if [[ $POM =~ tensorflow/platform/python ]]; then
            # Skip over currently broken builds on Windows
            JAVACPP_ARGS="-Djavacpp.platform.windows-x86_64="
        fi
        mvn dependency:list $MAVEN_ARGS $JAVACPP_ARGS -f $POM | tee /dev/tty | grep "$GROUP:.*:compile" >> dependencies.txt || REDEPLOY_STATUS=$?
    done
done

for LINE in $(sort -u dependencies.txt); do
    if [[ $LINE =~ $GROUP:([^:]*):jar:([^:]*):compile ]]; then
        ARTIFACT=${BASH_REMATCH[1]}
        VERSION=${BASH_REMATCH[2]}
        FILES=
        TYPES=
        CLASSIFIERS=
        for LINE2 in $(sort -u dependencies.txt); do
            if [[ $LINE2 =~ $GROUP:$ARTIFACT:jar:([^:]*):$VERSION:compile ]]; then
                CLASSIFIER=${BASH_REMATCH[1]}
                FILE=$ARTIFACT-$VERSION-$CLASSIFIER.jar
                cp -v $HOME/.m2/repository/${GROUP//.//}/$ARTIFACT/$VERSION/$FILE .
                [[ -n $FILES ]] && FILES=$FILES,$FILE || FILES=$FILE
                [[ -n $TYPES ]] && TYPES=$TYPES,jar || TYPES=jar
                [[ -n $CLASSIFIERS ]] && CLASSIFIERS=$CLASSIFIERS,$CLASSIFIER || CLASSIFIERS=$CLASSIFIER
            fi
        done
        if [[ -n $FILES ]]; then
            FILE=$ARTIFACT-$VERSION.jar
            cp -v $HOME/.m2/repository/${GROUP//.//}/$ARTIFACT/$VERSION/$FILE .
            unzip -o $FILE META-INF/maven/$GROUP/$ARTIFACT/pom.xml
            mvn deploy:deploy-file $MAVEN_ARGS -DrepositoryId=$REPOSITORY_ID -Durl=$REPOSITORY_URL -DpomFile=META-INF/maven/$GROUP/$ARTIFACT/pom.xml \
                    -Dfile=$FILE -DgroupId=$GROUP -DartifactId=$ARTIFACT -Dversion=$VERSION -Dfiles=$FILES -Dtypes=$TYPES -Dclassifiers=$CLASSIFIERS || REDEPLOY_STATUS=$?
        fi
    fi
done

# Prevent all this from getting cached by the CI server
rm -Rf $(find $HOME/.m2/repository -name '*SNAPSHOT*')

exit $REDEPLOY_STATUS
