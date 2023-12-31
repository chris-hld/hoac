#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:31:31 2023

HOAC - Higher-Order Ambisonics Audio Compression

@author: Chris Hold
"""
from pathlib import Path
import numpy as np
import subprocess

import time

import spaudiopy as spa
import safpy

import hoac


import matplotlib.pyplot as plt
plt.close('all')


user_pars = {'bitrate': 48,
             'numTC': 12,
             'metaDecimate': 2,
             'metaDecimateFreqLim': 0,
             'metaDoaGridOrder': 38,
             'metaDifBins': 8
             }

PEAK_NORM = True
PLOT = False

# create signal
fs = 48000
num_smpls = fs * 10
N_sph_in = 5

# file_name = 'Audio/Ambisonics/HOA-Mix/moving_pink_sn3d.wav'
# file_name = 'Audio/Ambisonics/test_scenes/bruckner_multichannelSH5N3D.wav'
# file_name = 'Audio/Ambisonics/test_scenes/BAND_shakers_bass_strings_drums.wav_ambi_o5_rev.wav'
# file_name = 'Audio/Ambisonics/test_scenes/em64_testScene_o5_ACN_N3D.wav'
file_name = 'Audio/Ambisonics/test_scenes/movingScene_o5_ACN_N3D.wav'

in_path = Path('~/OneDrive - Aalto University') / Path(file_name)

# defaults
hopsize = 128
blocksize = 8*hopsize
if user_pars['numTC'] == 6:
    N_sph_tcs = 2
    N_sph_pars = 3
    sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design((N_sph_tcs+1)).T)
    w = np.ones(len(sec_dirs[0]))
if user_pars['numTC'] == 9:
    N_sph_tcs = 3
    N_sph_pars = 4
    v, w = spa.grids.load_maxDet(N_sph_tcs-1)
    sec_dirs = spa.utils.cart2sph(*v.T)
if user_pars['numTC'] == 12:
    N_sph_tcs = 3
    N_sph_pars = 4
    sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design(N_sph_tcs+1).T)
    w = np.ones(len(sec_dirs[0]))


in_file = spa.io.load_audio(in_path, fs)
in_sig = in_file.get_signals()[:(N_sph_in+1)**2, :num_smpls]
if 'sn3d' in str(in_path).lower():
    in_sig = spa.sph.sn3d_to_n3d(in_sig)
    print("Converted SN3D input")
if PEAK_NORM:
    gain = 0.3 / np.max(np.abs(in_sig))
    in_sig *= gain
    print(f"applied gain {gain}")

x_nm = in_sig[:(N_sph_pars+1)**2, :]

# Prepare
num_secs = len(sec_dirs[0])
c_n = spa.sph.maxre_modal_weights(N_sph_tcs)
[A_nm, B_nm] = spa.sph.design_sph_filterbank(N_sph_tcs, sec_dirs[0],
                                             sec_dirs[1], c_n, 'real',
                                             'perfect')
beta = spa.sph.sph_filterbank_reconstruction_factor(A_nm[0, :], num_secs,
                                                    mode='amplitude')
beta = beta * (w / w.sum() * len(w))
A_nm_pars = spa.parsa.sh_beamformer_from_pattern('max_re', N_sph_pars-1,
                                                 sec_dirs[0], sec_dirs[1])
A_wxyz_c = np.array(spa.parsa.sh_sector_beamformer(A_nm_pars),
                    dtype=np.complex64)


num_slots = blocksize // hopsize

num_chin = (N_sph_pars+1)**2
hSTFT = safpy.afstft.AfSTFT(num_chin, 0, hopsize, fs)
hSTFT.clear_buffers()

num_bands = hSTFT.num_bands
f_qt = hoac.get_f_quantizer(num_bands)
num_fgroups = len(f_qt)
M_grouper = hoac.get_f_grouper(f_qt, num_fgroups, num_bands)
r_C = hoac.get_C_weighting(hSTFT.center_freqs)
M_grouper = r_C[:, None] * M_grouper
M_grouper = M_grouper / np.sum(M_grouper, axis=0)

# 38: 6.69, 48 : 5deg, 60, 66: 3.93 deg
qgrid, num_coarse = hoac.get_quant_grid(user_pars['metaDoaGridOrder'], None)
qdifbins = np.linspace(0, 0.99, user_pars['metaDifBins'], endpoint=False)**1.25

Path('./transport-data').mkdir(parents=True, exist_ok=True)

# Initialize and start timer
start_time = time.time()

x_nm_buf = np.hstack((x_nm, np.zeros((num_chin, hSTFT.processing_delay))))

azi = np.zeros((num_slots, num_secs, num_bands))
zen = np.zeros_like(azi)
dif = np.zeros_like(azi)
ene = np.zeros_like(azi)

azi_g = np.zeros((num_slots, num_secs, num_fgroups))
zen_g = np.zeros_like(azi_g)
dif_g = np.zeros_like(azi_g)
ene_g = np.zeros_like(azi_g)


num_blocks = x_nm_buf.shape[1] // blocksize
doa_idx_stream = np.zeros((num_blocks, num_slots, num_secs, num_fgroups),
                          dtype=np.int16)
dif_q_stream = np.zeros_like(doa_idx_stream)

start_smpl = 0
idx_blk = 0

print(f" HOAC encoding - {num_secs} TCs")

x_transport = A_nm @ x_nm[:A_nm.shape[1], :]

while idx_blk < num_blocks:
    blk_in = x_nm_buf[:, range(start_smpl, start_smpl+blocksize)]

    # afstft
    fd_sig_in = hSTFT.forward(blk_in)

    for idx_slt in range(num_slots):
        azi_g[idx_slt, ::], zen_g[idx_slt, ::], \
            dif_g[idx_slt, ::], \
            ene_g[idx_slt, ::], _ = hoac.grouped_sector_parameters(
                fd_sig_in[idx_slt, :, :], A_wxyz_c, M_grouper)

    dif_q_stream[idx_blk, ::] = hoac.quantize_dif(dif_g, qdifbins)
    doa_idx_stream[idx_blk, ::] = hoac.quantize_doa(azi_g, zen_g, qgrid,
                                                    dif_g, None, None)

    start_smpl += blocksize
    idx_blk += 1

# downsample side-info
doa_idx_stream, dif_q_stream = hoac.downsample_meta(
    doa_idx_stream, dif_q_stream, user_pars)

print('Parameterization: ', time.time()-start_time, 'seconds.')

pars_status = {
    'blocksize': blocksize,
    'hopsize': hopsize,
    'numTCs': num_secs,
    'metaDecimate': user_pars['metaDecimate'],
    'metaDecimateFreqLim': user_pars['metaDecimateFreqLim'],
    'numFreqs': num_fgroups,
    'qgrid': qgrid,
    'qdifbins': qdifbins,
    'A_nm': A_nm,
    'tc_v': sec_dirs,
    'beta': beta,
}


Path('./audio').mkdir(parents=True, exist_ok=True)
hoac.write_hoac(pars_status, np.array([doa_idx_stream, dif_q_stream]),
                x_transport, user_pars, fs)

print('Writing output: ', time.time()-start_time, 'seconds.')
subprocess.run(["du", "-sh", "transport-data/"])

spa.io.save_audio(spa.sph.n3d_to_sn3d(in_sig).T, './audio/in_sig_ambix.wav',
                  fs)

if PLOT:
    spa.plot.sh_rms_map(in_sig, TODB=True, title="Input SHD Signal")
    spa.plot.sh_rms_map(x_nm, TODB=True, title="Coded SHD Signal")
    f_idx = np.arange(1, num_fgroups)
    spa.plot.doa(azi_g[:, :, f_idx].ravel(), zen_g[:, :, f_idx].ravel(),
                 p=(ene_g *
                    (1.75**np.arange(num_fgroups)))[:, :, f_idx].ravel(),
                 alpha=1 - (dif_g[:, :, f_idx]**0.75).ravel(),
                 c=(np.arange(len(f_idx)) *
                    np.ones((num_slots, num_secs, len(f_idx)))).ravel(),
                 title="Parameterization")

    plt.plot(sec_dirs[0], np.pi/2-sec_dirs[1], 'k+', markeredgewidth=2)
    plt.show()
