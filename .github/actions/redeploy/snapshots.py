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

With --use-uploads, a lost metadata update doesn't matter in the first place: the jobs of this run
log every jar they upload ("Uploaded to <repository>: <url>"), so before checking, any jar that a
job uploaded more recently than the version Maven resolved replaces the resolved one.

Usage: snapshots.py check [--use-uploads <directory of the jars to republish>]
                          <directory of the version in the local Maven repository>
"""

import argparse
import glob
import json
import os
import re
import sys
import urllib.error
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


def run_start_and_jobs():
    """Returns when this workflow run was created, as a snapshot timestamp, and its jobs (all
    attempts)."""
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    created = github("/repos/%s/actions/runs/%s" % (repository, run_id))["created_at"]
    start = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d.%H%M%S")
    jobs = []
    page = 1
    while True:
        response = github("/repos/%s/actions/runs/%s/jobs?filter=all&per_page=100&page=%d"
                          % (repository, run_id, page))
        jobs += response["jobs"]
        if not response["jobs"] or page * 100 >= response["total_count"]:
            break
        page += 1
    return start, [job for job in jobs if job["name"].split(" ")[0] != "redeploy"]


def platforms_of(jobs):
    return set(job["name"].split(" ")[0] for job in jobs)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def job_log(job_id):
    """Returns the log of a job. The API redirects to a signed URL, which must be fetched without
    the API token."""
    request = urllib.request.Request(
        os.environ.get("GITHUB_API_URL", "https://api.github.com")
        + "/repos/%s/actions/jobs/%d/logs" % (os.environ["GITHUB_REPOSITORY"], job_id),
        headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"]})
    try:
        urllib.request.build_opener(NoRedirect).open(request, timeout=60)
        raise IOError("no redirect for the log of job %d" % job_id)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        location = e.headers["Location"]
    with urllib.request.urlopen(location, timeout=300) as response:
        return response.read().decode("utf-8", "replace")


def uploads(jobs, artifact, version):
    """Maps each jar classifier ("" for the main jar) to the URL of the most recent jar of
    <artifact>:<version> that a job of this run logged as uploaded."""
    base = version[:-len("-SNAPSHOT")]
    pattern = re.compile(r"Uploaded to [^:\s]+: (https?://\S+/" + re.escape(artifact) + "-"
                         + re.escape(base) + r"-(\d{8}\.\d{6}-\d+)(?:-([^/\s]+?))?\.jar)\s")
    found = {}
    for job in jobs:
        try:
            log = job_log(job["id"])
        except Exception as e:
            print("::warning::Could not read the log of job %s (%s)" % (job["name"], e))
            continue
        for url, stamp, classifier in pattern.findall(log):
            uploaded = base + "-" + stamp
            classifier = classifier or ""
            if classifier not in found or key(uploaded) > key(found[classifier][1]):
                found[classifier] = (url, uploaded)
    return found


def use_uploads(versions, jobs, directory, jars):
    """Replaces each jar to republish by a more recent one that a job of this run uploaded."""
    artifact = os.path.basename(os.path.dirname(os.path.normpath(directory)))
    version = os.path.basename(os.path.normpath(directory))
    for classifier, (url, uploaded) in sorted(uploads(jobs, artifact, version).items()):
        if classifier not in versions or key(uploaded) <= key(versions[classifier]):
            continue
        name = "%s-%s%s.jar" % (artifact, version, "-" + classifier if classifier else "")
        print("%s resolved to %s, but this run uploaded %s: using that instead"
              % (classifier or "The main jar", versions[classifier], uploaded))
        target = os.path.join(jars, name)
        with urllib.request.urlopen(url, timeout=600) as response, open(target + ".part", "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(target + ".part", target)
        versions[classifier] = uploaded


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
    jobs = []
    if args.since is not None and args.platforms is not None:
        start, platforms = args.since, set(args.platforms.split(","))
    else:
        try:
            start, jobs = run_start_and_jobs()
        except Exception as e:
            print("::warning::Not checking for stale snapshot jars: could not query the workflow run (%s)" % e)
            return 0
        platforms = platforms_of(jobs)
    versions = resolved_versions(args.directory)
    if not versions:
        print("::warning::Not checking for stale snapshot jars: no resolved snapshot versions in " + args.directory)
        return 0
    if args.use_uploads is not None and jobs:
        use_uploads(versions, jobs, args.directory, args.use_uploads)

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
    check_parser.add_argument("--use-uploads", metavar="JARS",
                              help="first replace the jars in JARS by more recent ones this run uploaded")
    check_parser.add_argument("directory")
    # For testing outside GitHub Actions: give the run's start and platforms instead of querying them.
    check_parser.add_argument("--since", help=argparse.SUPPRESS)
    check_parser.add_argument("--platforms", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
