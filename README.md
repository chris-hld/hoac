# Higher-Order Ambisonics Codec (HOAC)

Parametric spatial audio codec based on Higher-Order Directional Audio Coding (HO-DirAC).

Companion page http://research.spa.aalto.fi/publications/papers/hoac/.

The encoder extracts a set of transport audio channels and metadata.
The decoder reconstructs low orders, and resynthesizes high orders from the input parameterization.

The audio transport channel coding uses [Opus](https://github.com/xiph/opus) in discrete channel mode (implemented [here](https://github.com/xiph/opus-tools/pull/80)). It is also possible to use other audio-codecs.

The codec is currently prototyped for 5th order HOA (and higher), at a total bit rate of ~512 kbit/s for 'low', ~768 kbit/s for 'med', and ~1280kbit for 'high' profiles.

Reference implementation for papers

[C. Hold, L. McCormack, A. Politis and V. Pulkki, "Perceptually-Motivated Spatial Audio Codec for Higher-Order Ambisonics Compression", Accepted at IEEE ICASSP, 2024]

[C. Hold, L. McCormack, A. Politis and V. Pulkki, "Optimizing Higher-Order Directional Audio Coding with Adaptive Mixing and Energy Matching for Ambisonic Compression and Upmixing," 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2023.]

[C. Hold, V. Pulkki, A. Politis and L. McCormack, "Compression of Higher-Order Ambisonic Signals Using Directional Audio Coding," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, 2024.]


## Quickstart
Clone the repository
```
git clone https://github.com/chris-hld/hoac.git
```
It's a good idea to use conda for the python environment and package management
```
conda create --name hoac python=3.11 numpy scipy matplotlib portaudio cffi
```
To use the new environment
```
conda activate hoac
```
Then download and install [spaudiopy](https://github.com/chris-hld/spaudiopy) and [safpy](https://github.com/chris-hld/SAFpy).
For spaudiopy, in the conda environment, you can use for example
```
pip install spaudiopy
```
For safpy you need to follow the build instructions in its [safpy-README](https://github.com/chris-hld/SAFpy).

HOAC calls `opusenc` and `opusdec`, which can be built from https://github.com/xiph/opus-tools.
You should check support for the `opusenc --channels discrete` option.
Tested against these forks/branches: https://github.com/chris-hld/opus/tree/update_ambi_map3, https://github.com/chris-hld/opusfile/tree/channel-mapping-2-and-3, https://github.com/chris-hld/opus-tools/tree/channels-individual

If all dependencies are met, an example [encoder](https://github.com/chris-hld/hoac/blob/main/hoac_encoder.py) and [decoder](https://github.com/chris-hld/hoac/blob/main/hoac_decoder.py) shows the basic functionality.

## Details
If you would like to explore beyond the provided interfaces please reach out!
All function docstrings can be shown e.g. with
```python
In [1]: import hoac

In [2]: hoac.grouped_sector_parameters??
Signature:      hoac.grouped_sector_parameters(x_nm, A_wxyz_c, M_grouper, TRANSPOSE=False)
Call signature: hoac.grouped_sector_parameters(*args, **kwargs)
Type:           cython_function_or_method
String form:    <cyfunction grouped_sector_parameters at 0x7f9d83048fb0>
Docstring:     
Sector S parameters from SH signals L, frequency K band G grouped.

Parameters
----------
x_nm : np.ndarray
    L x K.
A_wxyz_c : np.ndarray, complex
    4*S x L.
M_grouper : np.ndarray
    K x G.
TRANSPOSE : np.ndarray, optional
    The default is False.

Returns
-------
azi_s, zen_s, dif_s, ene_s, int_s : np.ndarray
    S x G, or 3*S x G, or transposed

```
