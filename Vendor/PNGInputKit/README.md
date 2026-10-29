# PNG input pixels

This package preserves encoded, unassociated RGB/RGBA samples for model inputs. The C bridge accepts standard, eight-bit RGB/RGBA PNGs, including interlacing, and checks dimensions before allocating the decoded image. It does not apply display color management or premultiply alpha. ImageIO remains responsible for metadata and orientation at the Swift boundary.

LodePNG is pinned to `e584a8490db1b76b81bf298c11c09d3b55d9c270` (20261006) from [the upstream repository](https://github.com/lvandeve/lodepng). `lodepng.c` is the unchanged upstream `lodepng.cpp` under the upstream-supported C filename; `lodepng.h` is unchanged. File hashes are in `upstream.json`. The original zlib license notice is retained in both files and in `LICENSE`.

The build excludes encoding, disk access, C++ conveniences and ancillary metadata decoding. Core PNG transparency remains enabled. The app validates frame count and source metadata separately, bounds encoded input, and retains the existing unsupported-format checks. The bridge sets a decompressed-output limit derived from the validated dimensions; LodePNG additionally requires the exact scanline size. This limit rejects oversized output but is not a hard ceiling on temporary allocations. The wrapper checks every chunk CRC and rejects bytes after IEND.
