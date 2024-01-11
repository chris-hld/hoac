# Higher-Order Ambisonics Codec (HOAC)

Parametric spatial audio codec based on Higher-Order Directional Audio Coding (HO-DirAC).

Companion page http://research.spa.aalto.fi/publications/papers/hoac/.

The encoder extracts a set of transport audio channels and metadata.
The decoder reconstructs low orders, and resynthesizes high orders from the input parameterization.

The audio transport channel coding uses [Opus](https://github.com/xiph/opus) in discrete channel mode (implemented [here](https://github.com/xiph/opus-tools/pull/80)). It is also possible to use other audio-codecs.

The codec is currently prototyped for 5th order HOA (and higher), at a total bit rate of ~512 kbit/s for 'low', ~768 kbit/s for 'med', and ~1280kbit for 'high' profiles.

Reference implementation for papers

[C. Hold, L. McCormack, A. Politis and V. Pulkki, "PERCEPTUALLY-MOTIVATED SPATIAL AUDIO CODEC FOR HIGHER-ORDER AMBISONICS COMPRESSION", Accepted at ICASSP, 2024]

[C. Hold, L. McCormack, A. Politis and V. Pulkki, "Optimizing Higher-Order Directional Audio Coding with Adaptive Mixing and Energy Matching for Ambisonic Compression and Upmixing," 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2023.]

[C. Hold, V. Pulkki, A. Politis and L. McCormack, "Compression of Higher-Order Ambisonic Signals Using Directional Audio Coding," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, 2024.]


## Quickstart
For example, using conda
```
conda create --name hoac python=3.11 numpy scipy matplotlib portaudio cffi
```
Then download and install [spaudiopy](https://github.com/chris-hld/spaudiopy) and [safpy](https://github.com/chris-hld/SAFpy).
For the latter you need to follow the build instructions in its README.

HOAC calls `opusenc` and `opusdec`, which need to be build from https://github.com/xiph/opus-tools.

If all dependencies are met, an example [encoder](https://github.com/chris-hld/hoac/blob/main/hoac-encoder.py) and [decoder](https://github.com/chris-hld/hoac/blob/main/hoac-decoder.py) shows the basic functionality.


