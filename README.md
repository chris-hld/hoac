# Higher-Order Ambisonics Codec (HOAC)

Parametric spatial audio codec based on Higher-Order Directional Audio Coding (HO-DirAC).

Companion page http://research.spa.aalto.fi/publications/papers/hoac/

The encoder extracts a set of transport audio channels and metadata.
The decoder reconstructs low orders, and resynthesizes high orders.

Reference implementation for papers

[C. Hold, L. McCormack, A. Politis and V. Pulkki, "PERCEPTUALLY-MOTIVATED SPATIAL AUDIO CODEC FOR HIGHER-ORDER AMBISONICS COMPRESSION", Accepted at ICASSP, 2024]

[C. Hold, L. McCormack, A. Politis and V. Pulkki, "Optimizing Higher-Order Directional Audio Coding with Adaptive Mixing and Energy Matching for Ambisonic Compression and Upmixing," 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2023.]

[C. Hold, V. Pulkki, A. Politis and L. McCormack, "Compression of Higher-Order Ambisonic Signals Using Directional Audio Coding," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, 2024.]


## Quickstart
HOAC uses [spaudiopy](https://github.com/chris-hld/spaudiopy) and [safpy](https://github.com/chris-hld/SAFpy).
Make sure to install these packages and its dependencies.

For example, using conda
```
conda create --name hoac python=3.11 anaconda portaudio cffi
```
Then download and install [spaudiopy](https://github.com/chris-hld/spaudiopy) and [safpy](https://github.com/chris-hld/SAFpy).
For the latter you need to follow the build instructions in its README.

WIP

