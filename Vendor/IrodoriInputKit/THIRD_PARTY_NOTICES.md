# Third-party notices

## MT19937 / PyTorch CPU algorithm

IrodoriNoise adapts the MT19937 and normal_fill_16 algorithms from PyTorch 2.10.0. The original MT19937 notice follows.

```text
 * Copyright notice:
 * A C-program for MT19937, with initialization improved 2002/2/10.
 * Coded by Takuji Nishimura and Makoto Matsumoto.
 * This is a faster version by taking Shawn Cokus's optimization,
 * Matthe Bellew's simplification, Isaku Wada's real version.
 *
 * Before using, initialize the state by using init_genrand(seed)
 * or init_by_array(init_key, key_length).
 *
 * Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote
 *   products derived from this software without specific prior written
 *   permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * Any feedback is very welcome.
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 * email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
 */


```

## Hugging Face tokenizers

The Rust bridge links tokenizers 0.22.2 under Apache-2.0. Dependency versions are pinned in rust/Cargo.lock. The full license is in TOKENIZERS_LICENSE. Model and tokenizer assets retain their upstream licenses.

The full bundled PyTorch license is retained in PYTORCH_LICENSE.

## Irodori-TTS normalization

The text replacement rules and bracket removal are adapted from Aratako/Irodori-TTS commit 89f9d8fbd4d51ea019867ee1197725ede1df13c5 (irodori_tts/text_normalization.py). Copyright (c) 2026 Aratako. The full MIT license is retained in IRODORI_LICENSE.

## Unicode normalization

NFKC uses unicode-normalization 0.1.22 (Unicode 15.0.0), pinned to the Python 3.12 reference. The selected MIT license is retained in UNICODE_NORMALIZATION_LICENSE; Cargo.lock fixes the dependency graph.
