#!/usr/bin/env python3
"""
Checks, before the redeploy action republishes a snapshot, that the jar it is about to republish
for each platform was built by the current workflow run.

The platform jobs each update the shared maven-metadata.xml when they deploy. If one of those
updates gets lost, Maven resolves that platform's classifier to a jar from an earlier run, and
redeploy would republish it without any error (issue #1815). Snapshot versions encode when they
were built ("<version>-yyyyMMdd.HHmmss-<build number>", in UTC), so a jar built before this
workflow run started is stale.

Only classifiers that start with the name of a job in this run are checked: platforms without a
job in this run, such as disabled ones, keep whatever is published.

Usage: snapshots.py check <directory of the version in the local Maven repository>
"""

import argparse
import glob
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

TIMESTAMPED = re.compile(r"-(\d{8}\.\d{6})-(\d+)$")


def github(path):
    request = urllib.request.Request(
        os.environ.get("GITHUB_API_URL", "https://api.github.com") + path,
        headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def run_start_and_platforms():
    """Returns when this workflow run was created, as a snapshot timestamp, and the names of its
    jobs (all attempts), which are the platform names."""
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    created = github("/repos/%s/actions/runs/%s" % (repository, run_id))["created_at"]
    start = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d.%H%M%S")
    names = set()
    page = 1
    while True:
        jobs = github("/repos/%s/actions/runs/%s/jobs?filter=all&per_page=100&page=%d"
                      % (repository, run_id, page))
        names.update(job["name"].split(" ")[0] for job in jobs["jobs"])
        if not jobs["jobs"] or page * 100 >= jobs["total_count"]:
            break
        page += 1
    names.discard("redeploy")
    return start, names


def resolved_versions(directory):
    """Maps each jar classifier ("" for the main jar) to the snapshot version Maven resolved,
    from the remote metadata it left in the local repository."""
    versions = {}
    for path in glob.glob(os.path.join(directory, "maven-metadata-*.xml")):
        if path.endswith("-local.xml"):
            continue
        for entry in ET.parse(path).getroot().iter("snapshotVersion"):
            if entry.findtext("extension") != "jar":
                continue
            classifier = entry.findtext("classifier") or ""
            value = entry.findtext("value")
            if TIMESTAMPED.search(value) and (classifier not in versions
                                              or key(value) > key(versions[classifier])):
                versions[classifier] = value
    return versions


def key(version):
    match = TIMESTAMPED.search(version)
    return (match.group(1), int(match.group(2)))


def built_by(classifier, platforms):
    return classifier == "" or any(classifier == platform or classifier.startswith(platform + "-")
                                   for platform in platforms)


def check(args):
    if args.since is not None and args.platforms is not None:
        start, platforms = args.since, set(args.platforms.split(","))
    else:
        try:
            start, platforms = run_start_and_platforms()
        except Exception as e:
            print("::warning::Not checking for stale snapshot jars: could not query the workflow run (%s)" % e)
            return 0
    versions = resolved_versions(args.directory)
    if not versions:
        print("::warning::Not checking for stale snapshot jars: no resolved snapshot versions in " + args.directory)
        return 0

    stale = []
    print("Snapshot jars to republish (this run started at %s):" % start)
    for classifier in sorted(versions):
        version = versions[classifier]
        if not built_by(classifier, platforms):
            status = "not built by this run, not checked"
        elif key(version)[0] < start:
            status = "STALE: built before this run"
            stale.append((classifier, version))
        else:
            status = "ok"
        print("  %-24s %-36s %s" % (classifier or "(main jar)", version, status))
    for classifier, version in stale:
        print("::error::%s resolved to %s, which was built before this workflow run started at %s;"
              " the job that built it may have lost its update to maven-metadata.xml"
              % (classifier or "The main jar", version, start))
    return 1 if stale else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="fail if a jar to republish was built before this run")
    check_parser.add_argument("directory")
    # For testing outside GitHub Actions: give the run's start and platforms instead of querying them.
    check_parser.add_argument("--since", help=argparse.SUPPRESS)
    check_parser.add_argument("--platforms", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
