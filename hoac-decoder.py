#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:07:27 2023

@author: Chris Hold
"""

import numpy as np
import matplotlib.pyplot as plt
import time


import spaudiopy as spa
import safpy
import hoac


#%reset -f

SAVE = True
PLOT = False
PLAY = False


conf, sig_tc, doa_idx_stream, dif_idx_stream = hoac.read_hoac()

x_tc = np.sqrt(4*np.pi)*sig_tc.get_signals()
fs = sig_tc.fs
num_ch = x_tc.shape[0]

blocksize = conf['blocksize']
hopsize = conf['hopsize']
num_slots = conf['blocksize'] // conf['hopsize']

N_sph_out = 5
num_sh_out = (N_sph_out+1)**2

hSTFT = safpy.afstft.AfSTFT(num_ch, num_sh_out, hopsize, fs)
hSTFT.clear_buffers()
num_bands = hSTFT.num_bands
freqs = hSTFT.center_freqs
f_qt = hoac.get_f_quantizer(num_bands)
f_qt_c = np.asarray([np.mean(freqs[idx[0]: idx[1]]) for idx in f_qt])
qgrid = conf['qgrid']

qdifbins = conf['qdifbins']
qdifbins = np.append(qdifbins, 1)

M_mavg = np.zeros((N_sph_out+1, num_sh_out))
for n in range(N_sph_out+1):
    M_mavg[n, n**2:(n+1)**2] = 1/(2*n+1)
num_m = np.asarray([2*n+1 for n in range(N_sph_out+1)])

x_tc = np.hstack((x_tc, np.zeros((num_ch, hSTFT.processing_delay))))
out_sig = np.zeros((num_sh_out, x_tc.shape[1]))
fd_sig_in = np.zeros((8, num_ch, num_bands), dtype='complex64')

A_nm = conf['A_nm']
beta = conf['beta']

B_nm, B_nm_trunc, num_recov = hoac.sph_filterbank_reconstruction(A_nm)
assert num_recov

B_nm_trunc = B_nm_trunc[:, np.newaxis, :, np.newaxis]
B_nm_exp = np.zeros((num_sh_out, 8, num_ch, num_bands))
B_nm_exp[:B_nm.shape[0], :, :, :] = B_nm[:, np.newaxis, :, np.newaxis]
N_sph_recov = int(np.sqrt(num_recov) - 1)

y_sec = spa.sph.sh_matrix(N_sph_out, conf['tc_v'][0], conf['tc_v'][1])
C_f_dif = np.ones((num_ch, num_sh_out, num_sh_out))
C_f_dif = 4*np.pi * np.array([np.outer(y_sec[s, :], y_sec[s, :])
                              for s in range(num_ch)])
orne = num_sh_out / np.trace(A_nm.conj().T @ A_nm)


# Initialize
doa = np.zeros((8, num_ch, num_bands, 3))
doa_prev = np.zeros_like(doa)
dif = np.zeros((8, num_ch, num_bands))
dif_prev = np.zeros_like(dif)
Y = np.zeros((num_sh_out, 8, num_ch, num_bands))
X_nm = np.zeros((8, num_sh_out, num_bands), dtype=np.complex_)

M = np.zeros_like(Y)
M_prev = np.zeros_like(M)
g = np.ones((N_sph_out+1, num_bands))

start_time = time.time()

print(" HOAC decoding.")

start_smpl = 0
idx_blk = 0
while idx_blk < doa_idx_stream.shape[0]:
    blk_in = x_tc[:, range(start_smpl, start_smpl+blocksize)]

    fd_sig_in[:] = hSTFT.forward(blk_in)

    doa[:], dif[:] = hoac.dequantize_dirac_pars(doa_idx_stream, dif_idx_stream,
                                                freqs, f_qt_c,
                                                qgrid, qdifbins, idx_blk)

    M[:], Y[:] = hoac.formulate_M_Y(doa, dif, N_sph_out, B_nm_exp, beta,
                                    num_recov, B_nm_trunc)

    if idx_blk == 0:
        M_prev = M
    X_nm[:] = np.einsum('ldsk,dsk->dlk', 2/3*M + 1/3*M_prev, fd_sig_in)

    ene_s = np.real((fd_sig_in * fd_sig_in.conj()))
    ene_dir = (1-dif) * ene_s
    ene_dif = dif * ene_s

    gp = hoac.post_gain(X_nm, Y, ene_dir, ene_dif, C_f_dif, orne, M_mavg)
    gp[gp > 2.] = 2.
    gp[gp < .5] = .5
    g = 2/3 * gp + 1/3*g
    X_nm[:, num_recov:, :] = np.repeat(g, num_m, axis=0)[
        np.newaxis, num_recov:, :] * X_nm[:, num_recov:, :]

    # back
    blk_out = hSTFT.backward(X_nm)
    out_sig[:, range(start_smpl, start_smpl+blocksize)] = blk_out
    M_prev[:] = M[:]
    start_smpl += blocksize
    idx_blk += 1

out_sig = out_sig[:, hSTFT.processing_delay:]
print("READY")
spa.io.save_audio(spa.sph.n3d_to_sn3d(out_sig).T,
                  "./audio/out_hoac_ambix.wav", fs)


print('Decoding: ', time.time()-start_time, 'seconds.')


if SAVE:
    hrirs_nm = spa.decoder.magls_bin(spa.io.load_sofa_hrirs(
        '~/data/HRTFs/THK_KU100/HRIR_L2354.sofa'), N_sph_out)
    out_bin = spa.sig.MultiSignal([*spa.decoder.sh2bin(0.3*out_sig, hrirs_nm)],
                                  fs=fs)
    if PLAY:
        print("Playing decoded")
        out_bin.play()
    out_bin.save("./audio/Hoac_bin.wav", "PCM_16")

    uncompressed_sig = spa.sph.sn3d_to_n3d(
        spa.io.load_audio("./audio/in_sig_ambix.wav", fs).get_signals()[:num_sh_out, :])
    in_bin = spa.sig.MultiSignal([*spa.decoder.sh2bin(0.3*uncompressed_sig, hrirs_nm)],
                                 fs=fs)
    if PLAY:
        print("playing input")
        in_bin.play()
    in_bin.save("./audio/Input_bin.wav", "PCM_16")

    print("RMSE(n) ratio:",
          np.round(((M_mavg @ spa.utils.rms(out_sig, axis=-1)) /
                    (10e-10 + M_mavg @ spa.utils.rms(uncompressed_sig, axis=-1))), 3))
    print("RMSE(n) dB:",
          np.round(spa.utils.db(
              (M_mavg @ spa.utils.rms(out_sig, axis=-1)) /
              (10e-10 + M_mavg @ spa.utils.rms(uncompressed_sig, axis=-1))), 3))
    print("RMSE dB:",
          np.round(np.mean(np.abs(spa.utils.db(
              (spa.utils.rms(out_sig, axis=-1)) /
              (10e-10 + spa.utils.rms(uncompressed_sig, axis=-1))))), 3))

    if PLOT:
        hoac.sh_bar([spa.utils.rms(out_sig) /
                     (spa.utils.rms(uncompressed_sig+1e-10))], TODB=1,
                    centered=True, s=800)

        spa.plot.sh_rms_map(out_sig+1e-10, TODB=True,
                            title="Output SHD Signal")
        plt.show()
