#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOAC - Higher-Order Ambisonics Audio Compression - Decoder

@author: Chris Hold
"""
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

import spaudiopy as spa
import safpy

import hoac


SAVE = True
PLOT = False
PLAY = False


# Read
hoac_file = Path("./out_file.hoac")
conf, sig_tc, doa_idx_stream, dif_idx_stream = hoac.read_hoac(file=hoac_file)


# Prepare
N_sph_out = conf['N_sph_in']
num_sh_out = (N_sph_out+1)**2
r_smooth = 0.9

x_tc = np.sqrt(4*np.pi)*sig_tc.get_signals()
fs = sig_tc.fs
num_ch = x_tc.shape[0]

blocksize = conf['blocksize']
hopsize = conf['hopsize']
num_slots = conf['blocksize'] // conf['hopsize']

hSTFT = safpy.afstft.AfSTFT(num_ch, num_sh_out, hopsize, fs)
hSTFT.clear_buffers()
num_bands = hSTFT.num_bands
freqs = hSTFT.center_freqs
f_qt = hoac.get_f_quantizer(num_bands)
f_qt_c = np.asarray([np.mean(freqs[idx[0]: idx[1]]) for idx in f_qt])
qgrid = conf['qgrid']
qdifbins = np.append(conf['qdifbins'], 1)

x_tc = np.hstack((x_tc, np.zeros((num_ch, hSTFT.processing_delay))))
out_sig = np.zeros((num_sh_out, x_tc.shape[1]))
fd_sig_in = np.zeros((num_slots, num_ch, num_bands), dtype='complex64')

A_nm = conf['A_nm']
beta = conf['beta']

B_nm, B_nm_trunc, num_recov = hoac.sph_filterbank_reconstruction(A_nm)
assert num_recov

B_nm_trunc = B_nm_trunc[:, np.newaxis, :, np.newaxis]
B_nm_exp = np.zeros((num_sh_out, num_slots, num_ch, num_bands))
B_nm_exp[:B_nm.shape[0], :, :, :] = B_nm[:, np.newaxis, :, np.newaxis]
N_sph_recov = int(np.sqrt(num_recov) - 1)

C_dif = hoac.get_cov_dif(N_sph_out, num_ch, conf)
orne = num_sh_out / np.trace(A_nm.conj().T @ A_nm)

M_mavg = np.zeros((N_sph_out+1, num_sh_out))
for n in range(N_sph_out+1):
    M_mavg[n, n**2:(n+1)**2] = 1/(2*n+1)
num_m = np.asarray([2*n+1 for n in range(N_sph_out+1)])


# Initialize and start timer
start_time = time.time()

doa = np.zeros((num_slots, num_ch, num_bands, 3))
doa_prev = np.zeros_like(doa)
dif = np.zeros((num_slots, num_ch, num_bands))
dif_prev = np.zeros_like(dif)
Y = np.zeros((num_sh_out, num_slots, num_ch, num_bands))
X_nm = np.zeros((num_slots, num_sh_out, num_bands), dtype=np.complex_)

M = np.zeros_like(Y)
M_prev = np.zeros_like(M)
g = np.ones((N_sph_out+1, num_bands))

print(" HOAC decoding.")

start_smpl = 0
idx_blk = 0
while idx_blk < doa_idx_stream.shape[0]:
    blk_in = x_tc[:, range(start_smpl, start_smpl+blocksize)]

    fd_sig_in[:] = hSTFT.forward(blk_in)

    doa[:], dif[:] = hoac.dequantize_dirac_pars(doa_idx_stream[idx_blk, ...],
                                                dif_idx_stream[idx_blk, ...],
                                                freqs, f_qt_c,
                                                qgrid, qdifbins)

    M[:], Y[:] = hoac.formulate_M_Y(doa, dif, N_sph_out, B_nm_exp, beta,
                                    num_recov, B_nm_trunc)

    if idx_blk == 0:
        M_prev[:] = M[:]
    M = r_smooth * M + (1. - r_smooth) * M_prev
    X_nm[:] = np.einsum('ldsk,dsk->dlk', M, fd_sig_in)

    gn = hoac.opt_gain(X_nm, Y, dif,
                       np.real((fd_sig_in * fd_sig_in.conj())),
                       C_dif, orne, M_mavg)
    np.clip(gn, 0.5, 2., out=gn)
    g = r_smooth * gn + (1. - r_smooth) * g

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
Path('./audio').mkdir(parents=True, exist_ok=True)
spa.io.save_audio(spa.sph.n3d_to_sn3d(out_sig).T,
                  "./audio/out_hoac_ambix.wav", fs)


print('Decoding: ', time.time()-start_time, 'seconds.')


if SAVE:
    hrirs_nm = spa.decoder.magls_bin(spa.io.load_sofa_hrirs(
        '~/data/HRTFs/THK_KU100/HRIR_L2354.sofa'), N_sph_out)
    out_bin = spa.sig.MultiSignal([*spa.decoder.sh2bin(0.3*out_sig, hrirs_nm)],
                                  fs=fs)
    out_bin.save("./audio/Hoac_bin.wav", "PCM_16")

    uncompressed_sig = spa.sph.sn3d_to_n3d(
        spa.io.load_audio("./audio/in_sig_ambix.wav", fs).get_signals()[:num_sh_out, :])
    in_bin = spa.sig.MultiSignal([*spa.decoder.sh2bin(0.3*uncompressed_sig, hrirs_nm)],
                                 fs=fs)
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

    if PLAY:
        print("Playing input")
        in_bin.play()
        print("Playing decoded")
        out_bin.play()

    if PLOT:
        spa.plot.sh_bar([spa.utils.rms(out_sig) /
                         (spa.utils.rms(uncompressed_sig+1e-10))], TODB=1,
                        centered=True, s=800)

        spa.plot.sh_rms_map(out_sig+1e-10, TODB=True,
                            title="Output SHD Signal")
        plt.show()
