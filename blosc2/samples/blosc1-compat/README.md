Blosc1 compatibility check
==========================
Zarr (numcodecs) and N5 use the Blosc1 chunk format. Stock C-Blosc2 chunks, including those from `blosc1_compress()`, cannot be decoded by C-Blosc 1.x ([c-blosc2 PR #815](https://github.com/Blosc/c-blosc2/pull/815)). Use `blosc2_ctx_set_blosc1_format()` or set `BLOSC_BLOSC1_COMPAT=1`.

`run.sh` writes 720 chunks per mode with this preset (5 codecs x no/byte/bit shuffle x 8 typesizes x 6 sizes) and decodes them with a real C-Blosc 1.x obtained with [pixi](https://pixi.sh). The stock mode is expected to fail, the other two to pass.

```bash
 $ mvn package   # in the parent directory
 $ ./run.sh
```
