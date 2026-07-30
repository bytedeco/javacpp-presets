# Multi-platform native release (io.github.mullerhai)

Goal: publish **linux / windows / macos** classifier jars (+ **cuda** + **pytorch-platform-gpu**)
under the same versions as the pure-Java / mac jars already on Central.

Central versions are **immutable**. Each multi-platform drop needs a **new** suffix
(`beta-05`, `beta-06`, …) even if only classifiers are added.

## Consumer coordinates (target)

```scala
"io.github.mullerhai" % "pytorch"  % "2.13.0-1.5.14-beta-05"
"io.github.mullerhai" % "pytorch"  % "2.13.0-1.5.14-beta-05" classifier "macosx-arm64"
"io.github.mullerhai" % "pytorch"  % "2.13.0-1.5.14-beta-05" classifier "linux-x86_64"
"io.github.mullerhai" % "pytorch"  % "2.13.0-1.5.14-beta-05" classifier "linux-x86_64-gpu"
"io.github.mullerhai" % "pytorch"  % "2.13.0-1.5.14-beta-05" classifier "windows-x86_64"
"io.github.mullerhai" % "openblas" % "0.3.33-1.5.14-beta-05"
"io.github.mullerhai" % "openblas" % "0.3.33-1.5.14-beta-05" classifier "linux-x86_64"
"io.github.mullerhai" % "javacpp"  % "1.5.14-beta-05"
"io.github.mullerhai" % "javacpp"  % "1.5.14-beta-05" classifier "linux-x86_64"
"io.github.mullerhai" % "cuda"     % "13.3-9.24-1.5.14-beta-05"
"io.github.mullerhai" % "cuda"     % "13.3-9.24-1.5.14-beta-05" classifier "linux-x86_64"
"io.github.mullerhai" % "pytorch-platform-gpu" % "2.13.0-1.5.14-beta-05"
```

(or `*-platform` aggregators that pull all classifiers).

## How bytedeco does it (and what we mirror)

Upstream workflows (`.github/workflows/{openblas,ffmpeg,opencv,pytorch,cuda}.yml`):

1. **Per-OS job** builds one `javacpp.platform` (linux-x86_64, windows-x86_64, …)
   via composite actions under `.github/actions/deploy-*`.
2. Artifacts land in the deployer’s local `~/.m2` as
   `org.bytedeco:<module>:<ver>:<classifier>`.
3. A final **`platform` / `redeploy` job** re-uploads main + all classifiers with
   aligned timestamps so Gradle/sbt resolve consistently.

Our fork:

| Piece | Path |
|-------|------|
| Multi-OS build + assemble | [`.github/workflows/mullerhai-multiplatform.yml`](../../.github/workflows/mullerhai-multiplatform.yml) |
| GAV rewrite + GPG + Central bundle | [`prepare_and_publish.py`](prepare_and_publish.py) / [`publish.sh`](publish.sh) |
| Config (suffix, versions) | [`config.env`](config.env) |

Rebrand: **`org.bytedeco` → `io.github.mullerhai`**, version `*-SNAPSHOT` → `*-${PUBLISH_SUFFIX}`.

## Path A — Local multi-platform publish (no CI yet)

If `~/.m2/repository/org/bytedeco/<module>/<ver>/` already has classifier jars
(`linux-x86_64`, `windows-x86_64`, …) from earlier downloads/builds:

```bash
cd pytorch/scripts/publish
# edit config.env → PUBLISH_SUFFIX=beta-05  (must be unused on Central)
source ../../../../.m2/central.env   # or export CENTRAL_USERNAME/PASSWORD
./publish.sh all
# then upload:
source ~/.m2/central.env
BUNDLE=$(ls bundles/mullerhai-javacpp-stack-beta-05-*.zip | sort | tail -1)
AUTH=$(python3 -c "import base64,os; print(base64.b64encode(f'{os.environ[\"CENTRAL_USERNAME\"]}:{os.environ[\"CENTRAL_PASSWORD\"]}'.encode()).decode())")
curl --max-time 3600 -o /tmp/up.txt -w "HTTP %{http_code}\n" \
  -X POST "https://central.sonatype.com/api/v1/publisher/upload?publishingType=AUTOMATIC&name=mullerhai-beta-05" \
  -H "Authorization: Bearer $AUTH" \
  -F "bundle=@$BUNDLE;type=application/zip"
cat /tmp/up.txt
```

Missing classifiers are **skipped** (logged); present ones are signed and bundled.

**Today on this machine (typical):**

| Module | Classifiers usually present |
|--------|-----------------------------|
| javacpp / openblas / ffmpeg / opencv / cpython / numpy | linux-*, windows-x86_64, macosx-* |
| cuda | linux-x86_64, linux-arm64, windows-x86_64 |
| pytorch | **macosx-arm64 only** until CI builds linux/windows (/gpu) |

## Path B — GitHub Actions (full rebuild + publish)

### 1. Secrets (required)

In the GitHub repo (e.g. `CSharpHai/javacpp-presets` or `mullerhai/javacpp-presets`):

| Secret | Value |
|--------|--------|
| `CENTRAL_USERNAME` | Sonatype user-token name |
| `CENTRAL_PASSWORD` | Sonatype user-token password |
| `GPG_PRIVATE_KEY` | `gpg --export-secret-keys --armor 7AD293084072FD9F` |
| `GPG_KEY_ID` | `7AD293084072FD9F` |
| `GPG_PASSPHRASE` | empty if key has no passphrase |

### 2. Push workflow + scripts

```bash
git add .github/workflows/mullerhai-multiplatform.yml \
        pytorch/scripts/publish/
git commit -m "ci: multi-platform native publish for io.github.mullerhai"
git push origin master
```

### 3. Run workflow

GitHub → **Actions** → **mullerhai-multiplatform** → **Run workflow**:

- `modules`: e.g. `javacpp,openblas,ffmpeg,opencv,cpython,numpy,cuda`
- `publish_suffix`: `beta-05` (new!)
- `build_pytorch_gpu`: true when ready for linux/windows GPU
- `publish`: true

Jobs:

1. `linux-x86_64` / `linux-arm64` / `windows-x86_64` / `macosx-arm64` / `macosx-x86_64`  
   → `mvn clean install -Djavacpp.platform=…` per module  
2. `assemble-and-publish`  
   → merge all `org/bytedeco` m2 trees → `publish.sh all` → Central AUTOMATIC upload → poll `PUBLISHED`

### 4. pytorch GPU

Matches bytedeco `pytorch.yml` matrix `ext: ["", -gpu]`:

```bash
mvn -f pytorch/pom.xml clean install \
  -Djavacpp.platform=linux-x86_64 \
  -Djavacpp.platform.extension=-gpu
```

Produces classifier `linux-x86_64-gpu`. Aggregator POM:
`pytorch/platform/gpu` → artifact `pytorch-platform-gpu`
(depends on `cuda-platform` + gpu classifiers).

Mac does **not** ship cuda / pytorch-gpu (same as bytedeco).

## What you need to provide

1. **Confirm GitHub remote** for Actions (`origin` is currently `CSharpHai/javacpp-presets`).  
   For `io.github.mullerhai` namespace, the verified GitHub user/org should be **mullerhai**  
   (or keep publishing under the already-verified namespace from beta-01..04).
2. **Repo secrets** listed above (Central token + GPG).
3. **Permission to push** the workflow file (or open a PR).
4. Optional: **self-hosted runners** with CUDA if GitHub-hosted disk/time is insufficient for full libtorch-gpu.

## Incremental strategy (recommended)

1. **beta-05 now (local):** republish all modules with **every classifier already in local m2**, including **cuda** linux/windows. pytorch still mac-only classifiers.
2. **CI pass 1:** openblas + ffmpeg + opencv + cpython + numpy + javacpp + cuda on linux/windows/mac → beta-06 with full non-pytorch natives.
3. **CI pass 2:** pytorch cpu linux/windows → beta-07.
4. **CI pass 3:** pytorch `-gpu` + `pytorch-platform-gpu` → beta-08.

## Troubleshooting

- **HTTP 408 on upload:** large zip; use `curl --max-time 3600` (script/docs already do).
- **Version already exists:** bump `PUBLISH_SUFFIX`.
- **POM still has org.bytedeco / SNAPSHOT:** fixed in `prepare_and_publish.py` rewrite (drop foreign bytedeco; map ours to `io.github.mullerhai`).
- **Missing classifier:** only platforms present under `~/.m2/.../org/bytedeco/<art>/<ver>/` are packaged; CI must produce the rest.
