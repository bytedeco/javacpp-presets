# Publish `io.github.mullerhai` JavaCPP stack to Maven Central

Automates a **formal pre-release** (`*-beta-01`) of:

| Artifact | Version |
|---|---|
| `javacpp` / `javacpp-platform` | `1.5.14-beta-01` |
| `openblas` / `openblas-platform` | `0.3.33-1.5.14-beta-01` |
| `cuda` / `cuda-platform` | `13.3-9.24-1.5.14-beta-01` |
| `pytorch` / `pytorch-platform` | `2.13.0-1.5.14-beta-01` |

**groupId:** `io.github.mullerhai`  
**Java packages stay** `org.bytedeco.*` (binary-compatible with upstream class names).  
Artifacts are **republished from local `~/.m2` SNAPSHOTs** (no full native rebuild).

---

## One-time setup

### 1. GPG key (already done on this machine)

```
Key ID:        7AD293084072FD9F
Fingerprint:   ED8664EC6A980A6C6CF721787AD293084072FD9F
UID:           mullerhai (Maven Central signing key) <hai710459649@foxmail.com>
```

Public key: `mullerhai-public-key.asc` (sent to keys.openpgp.org / keyserver.ubuntu.com).

**Important:** open https://keys.openpgp.org/ and **verify the email** so Central can resolve the key.

Backup the secret key somewhere safe (not in git):

```bash
gpg --export-secret-keys --armor 7AD293084072FD9F > ~/mullerhai-maven-secret.asc
chmod 600 ~/mullerhai-maven-secret.asc
```

### 2. Sonatype Central Portal

1. Sign up: https://central.sonatype.com/
2. Create a **User Token**: Account → Generate User Token  
   - `CENTRAL_USERNAME` = token username  
   - `CENTRAL_PASSWORD` = token password
3. **Claim namespace** `io.github.mullerhai`:
   - Namespaces → Add namespace → `io.github.mullerhai`
   - Prove ownership of GitHub user/org **mullerhai** (verification file or DNS as instructed)
4. Wait until the namespace shows as **Verified**.

### 3. Credentials on this machine

```bash
export CENTRAL_USERNAME='...'   # portal token name
export CENTRAL_PASSWORD='...'   # portal token value
```

Or merge `settings.xml.template` into `~/.m2/settings.xml` (server id must be `central`).

---

## Publish commands

```bash
cd pytorch/scripts/publish
chmod +x publish.sh

# Stage from local SNAPSHOTs, GPG-sign, checksum, zip, install to local m2
./publish.sh all

# Upload newest bundle (USER_MANAGED = review in UI before publish)
./publish.sh upload

# Or full auto after namespace is verified:
./publish.sh all --upload --publishing-type AUTOMATIC
# or USER_MANAGED then publish after validation:
./publish.sh upload --publish
```

Individual steps:

```bash
./publish.sh stage
./publish.sh sign
./publish.sh bundle
./publish.sh install-local
./publish.sh summary
./publish.sh status --deployment-id <id>
```

Outputs:

- `staging/io/github/mullerhai/...` — Maven repo layout  
- `bundles/mullerhai-javacpp-stack-beta-01-*.zip` — Central Portal deployment bundle  
- Local m2 also gets `io.github.mullerhai:*:*-beta-01` after `install-local` / `all`

---

## Consumer coordinates

```xml
<dependency>
  <groupId>io.github.mullerhai</groupId>
  <artifactId>pytorch</artifactId>
  <version>2.13.0-1.5.14-beta-01</version>
</dependency>
<dependency>
  <groupId>io.github.mullerhai</groupId>
  <artifactId>pytorch</artifactId>
  <version>2.13.0-1.5.14-beta-01</version>
  <classifier>macosx-arm64</classifier>
</dependency>
```

sbt (matches your style):

```scala
ThisBuild / organization := "io.github.mullerhai"
libraryDependencies ++= Seq(
  "io.github.mullerhai" % "pytorch" % "2.13.0-1.5.14-beta-01",
  "io.github.mullerhai" % "pytorch" % "2.13.0-1.5.14-beta-01" classifier "macosx-arm64",
  "io.github.mullerhai" % "openblas" % "0.3.33-1.5.14-beta-01",
  "io.github.mullerhai" % "openblas" % "0.3.33-1.5.14-beta-01" classifier "macosx-arm64",
  "io.github.mullerhai" % "javacpp" % "1.5.14-beta-01",
  "io.github.mullerhai" % "javacpp" % "1.5.14-beta-01" classifier "macosx-arm64",
)
```

---

## What the rewrite does

From each local `org.bytedeco` SNAPSHOT:

1. Copies main / sources / javadoc / platform classifier jars  
2. Rewrites POM: `groupId` → `io.github.mullerhai`, version `*-SNAPSHOT` → `*-beta-01`  
3. Sets developers / SCM / license to mullerhai metadata  
4. Rewrites inter-deps (`javacpp`, `openblas`, `cuda`) to the new GAV  
5. Drops bytedeco parent POM (Central needs a self-contained POM)  
6. GPG-signs every file + writes md5/sha1/sha256/sha512  
7. Zips Maven layout for Central Portal Publisher API  

Missing javadoc/sources (e.g. plain javacpp) get a **minimal valid jar** so Central validation passes.

---

## Notes / limits

- **Only platforms present in local m2 are published.**  
  Current pytorch native jar is mainly `macosx-arm64`. Linux/Windows pytorch classifiers are skipped if absent.  
  openblas/javacpp/cuda multi-platform jars from Central snapshots are included when present.
- Re-run `./publish.sh all` after you install more platform jars into `~/.m2`.
- Do **not** commit `staging/`, `bundles/`, or secret keys.
- Namespace `io.github.mullerhai` requires control of https://github.com/mullerhai.

## Config

Edit `config.env` for versions, groupId, developer info, GPG key id.
