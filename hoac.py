#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chris Hold
"""
import numpy as np
from scipy.ndimage import median_filter

from pathlib import Path
from warnings import warn
import subprocess
import threading
import bz2
import pickle

import spaudiopy as spa
try:
    import safpy
except ImportError as e:
    warn("SAFPY not available.", ImportWarning)

try:
    import pylibopus
except ImportError as e:
    warn("pylibopus not available.", ImportWarning)


HOAC_VERSION = '0.2'


def get_version():
    """HOAC version."""
    return HOAC_VERSION


class Encoder():
    """HOAC encoder."""
    def __init__(self, fs, profile='high'):
        self.fs = fs
        self.hopsize = 128
        self.blocksize = 8 * self.hopsize
        # defaults
        self.user_pars = {'bitrateTC': 48 if profile != 'low' else 32,
                    'numTC': 6 if profile == 'low' else 9 if profile == 'med' else 12,
                    'metaDecimateFreqLim': 8 if profile == 'low' else
                                            4 if profile == 'med' else 0,
                    'metaDecimate': 2,
                    'metaDoaGridOrder': 38,
                    'metaDifBins': 8
                    }
        self.hSTFT = None


    def prepare(self):
        """Prepare encoder."""
        if self.user_pars['numTC'] == 6:
            N_sph_tcs = 2
            sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design((N_sph_tcs+1)).T)
            w = np.ones(len(sec_dirs[0]))
        if self.user_pars['numTC'] == 9:
            N_sph_tcs = 3
            v, w = spa.grids.load_maxDet(N_sph_tcs-1)
            sec_dirs = spa.utils.cart2sph(*v.T)
        if self.user_pars['numTC'] == 12:
            N_sph_tcs = 3
            sec_dirs = spa.utils.cart2sph(*spa.grids.load_t_design(N_sph_tcs+1).T)
            w = np.ones(len(sec_dirs[0]))
        
        num_secs = len(sec_dirs[0])
        [A_nm, B_nm] = spa.sph.design_sph_filterbank(
            N_sph_tcs, sec_dirs[0], sec_dirs[1],
            spa.sph.maxre_modal_weights(N_sph_tcs), 'perfect')
        beta = (w / w.sum() * len(w)) * spa.sph.sph_filterbank_reconstruction_factor(
            A_nm[0, :], num_secs, mode='amplitude')
        N_sph_pars = N_sph_tcs + 1
        A_nm_pars = spa.parsa.sh_beamformer_from_pattern('max_re', N_sph_pars-1,
                                                        sec_dirs[0], sec_dirs[1])
        A_wxyz_c = np.array(spa.parsa.sh_sector_beamformer(A_nm_pars),
                            dtype=np.complex64)

        self.hSTFT = safpy.afstft.AfSTFT((N_sph_pars+1)**2, 0, self.hopsize, self.fs)
        self.hSTFT.clear_buffers()
        num_slots = self.blocksize // self.hopsize

        f_qt = get_f_quantizer(self.hSTFT.num_bands)
        num_fgroups = len(f_qt)
        M_grouper = get_C_weighting(self.hSTFT.center_freqs)[:, None] * \
            get_f_grouper(f_qt)
        M_grouper = M_grouper / np.sum(M_grouper, axis=0)

        qgrid, num_coarse = get_quant_grid(self.user_pars['metaDoaGridOrder'], None)
        qdifbins = np.linspace(0.01, 0.99, self.user_pars['metaDifBins'], False)**1.5

        self.N_sph_tcs = N_sph_tcs
        self.sec_dirs = sec_dirs
        self.A_nm = A_nm
        self.A_wxyz_c = A_wxyz_c
        self.beta = beta
        self.M_grouper = M_grouper
        self.qgrid = qgrid
        self.qdifbins = qdifbins


    def encode(self, in_sig):
        """Encode."""
        N_sph_in = int(np.sqrt(in_sig.shape[0]) - 1)
        N_sph_pars = self.N_sph_tcs + 1
        x_nm = in_sig[:(N_sph_pars+1)**2, :]
        x_nm_buf = np.hstack((x_nm, np.zeros((x_nm.shape[0], self.hSTFT.processing_delay))))

        num_slots = self.blocksize // self.hopsize
        num_secs = len(self.sec_dirs[0])
        num_fgroups = self.M_grouper.shape[1]

        azi_g = np.zeros((num_slots, num_secs, num_fgroups))
        zen_g = np.zeros_like(azi_g)
        dif_g = np.zeros_like(azi_g)
        ene_g = np.zeros_like(azi_g)

        num_blocks = x_nm_buf.shape[1] // self.blocksize
        doa_idx_stream = np.zeros((num_blocks, num_slots, num_secs, num_fgroups),
                                dtype=np.uint16)
        dif_idx_stream = np.zeros_like(doa_idx_stream, dtype=np.uint8)

        start_smpl = 0
        idx_blk = 0

        tc_sigs = self.A_nm @ x_nm[:self.A_nm.shape[1], :]

        while idx_blk < num_blocks:
            blk_in = x_nm_buf[:, range(start_smpl, start_smpl+self.blocksize)]

            # afstft
            fd_sig_in = self.hSTFT.forward(blk_in)

            for idx_slt in range(num_slots):
                azi_g[idx_slt, ...], zen_g[idx_slt, ...], \
                    dif_g[idx_slt, ...], \
                    ene_g[idx_slt, ...], _ = grouped_sector_parameters(
                        fd_sig_in[idx_slt, ...], self.A_wxyz_c, self.M_grouper)
            azi_g, zen_g, dif_g, ene_g = post_pars(azi_g, zen_g, dif_g, ene_g)
            dif_idx_stream[idx_blk, ...] = quantize_dif(dif_g, self.qdifbins)
            doa_idx_stream[idx_blk, ...] = quantize_doa(azi_g, zen_g, self.qgrid,
                                                      dif_g)

            start_smpl += self.blocksize
            idx_blk += 1

        # downsample side-info
        doa_idx_stream, dif_idx_stream = downsample_meta(
            doa_idx_stream, dif_idx_stream, self.user_pars)

        pars_status = {
            'N_sph_in': N_sph_in,
            'fs': self.fs,
            'bitrateTC': self.user_pars['bitrateTC'],
            'numTC': num_secs,
            'metaDecimate': self.user_pars['metaDecimate'],
            'metaDecimateFreqLim': self.user_pars['metaDecimateFreqLim'],
            'blocksize': self.blocksize,
            'hopsize': self.hopsize,
            'numFreqs': num_fgroups,
            'qgrid': self.qgrid,
            'qdifbins': self.qdifbins,
            'A_nm': self.A_nm,
            'tc_v': self.sec_dirs,
            'beta': self.beta,
        }
        
        
        if pars_status['bitrateTC'] > 0:
            data_tcs, enc_lookahead = encode_tcs(tc_sigs, pars_status['bitrateTC'],
                                                 pars_status['fs'])
            pars_status['enc_lookahead'] = enc_lookahead
        else:
            data_tcs = 1/(np.sqrt(4*np.pi)) * tc_sigs.T
            pars_status['enc_lookahead'] = 0

        data_pars_stream = encode_pars(doa_idx_stream, dif_idx_stream)

        return pars_status, data_tcs, data_pars_stream


class Decoder():
    """HOAC decoder."""
    def __init__(self, fs, N_sph_out, pars_smooth=0.3):
        self.fs = fs
        self.N_sph_out = N_sph_out
        self.pars_smooth = pars_smooth
        self.hSTFT = None


    def prepare(self, conf):
        """Prepare decoder."""
        num_sh_out = (self.N_sph_out+1)**2
        fs = self.fs
        num_tc = conf['numTC']

        blocksize = conf['blocksize']
        hopsize = conf['hopsize']
        num_slots = conf['blocksize'] // conf['hopsize']

        self.hSTFT = safpy.afstft.AfSTFT(num_tc, num_sh_out, hopsize, fs)
        self.hSTFT.clear_buffers()
        num_bands = self.hSTFT.num_bands
        freqs = self.hSTFT.center_freqs
        f_qt = get_f_quantizer(num_bands)
        f_qt_c = np.asarray([np.mean(freqs[idx[0]: idx[1]]) for idx in f_qt])
        qgrid = conf['qgrid']
        qdifbins = np.append(conf['qdifbins'], 1)

        A_nm = conf['A_nm']
        beta = conf['beta']

        B_nm, B_nm_trunc, num_recov = sph_filterbank_reconstruction(A_nm)
        assert num_recov

        B_nm_trunc = B_nm_trunc[:, np.newaxis, :, np.newaxis]
        B_nm_exp = np.zeros((num_sh_out, num_slots, num_tc, num_bands))
        B_nm_exp[:B_nm.shape[0], :, :, :] = B_nm[:, np.newaxis, :, np.newaxis]
        N_sph_recov = int(np.sqrt(num_recov) - 1)

        C_dif = get_cov_dif(self.N_sph_out, num_tc, conf)
        orne = num_sh_out / np.trace(A_nm.conj().T @ A_nm)

        M_mavg = np.zeros((self.N_sph_out+1, num_sh_out))
        for n in range(self.N_sph_out+1):
            M_mavg[n, n**2:(n+1)**2] = 1/(2*n+1)
        num_m = np.asarray([2*n+1 for n in range(self.N_sph_out+1)])

        self.conf = conf
        self.blocksize = blocksize
        self.num_slots = num_slots
        self.num_tc = num_tc
        self.qgrid = qgrid
        self.qdifbins = qdifbins
        self.f_qt_c = f_qt_c
        self.B_nm_exp = B_nm_exp
        self.B_nm_trunc = B_nm_trunc
        self.beta = beta
        self.num_recov = num_recov
        self.C_dif = C_dif
        self.orne = orne
        self.M_mavg = M_mavg
        self.num_m = num_m

    
    def decode(self, c_data_tcs, c_data_pars):
        """Decode."""
        conf = self.conf
        num_tc = self.num_tc
        num_bands = self.hSTFT.num_bands
        num_sh_out = (self.N_sph_out+1)**2
        blocksize = self.blocksize
        num_slots = self.num_slots
        freqs = self.hSTFT.center_freqs

        doa_idx_stream, dif_idx_stream = decode_pars(conf, c_data_pars)
        num_blocks = doa_idx_stream.shape[0]
        if conf['bitrateTC'] > 0:
            tc_sigs = decode_tcs(conf, c_data_tcs)
        else:
            tc_sigs = np.sqrt(4*np.pi)*c_data_tcs.T

        tc_sigs = np.hstack((tc_sigs, np.zeros((num_tc, self.hSTFT.processing_delay))))
        out_sig = np.zeros((num_sh_out, tc_sigs.shape[1]))
        fd_sig_in = np.zeros((num_slots, num_tc, num_bands), dtype='complex64')


        doa = np.zeros((num_slots, num_tc, num_bands, 3))
        doa_prev = np.zeros_like(doa)
        dif = np.zeros((num_slots, num_tc, num_bands))
        dif_prev = np.zeros_like(dif)
        Y = np.zeros((num_sh_out, num_slots, num_tc, num_bands))
        X_nm = np.zeros((num_slots, num_sh_out, num_bands), dtype=complex)

        M = np.zeros_like(Y)
        M_prev = np.zeros_like(M)
        g = np.ones((self.N_sph_out+1, num_bands))

        start_smpl = 0
        idx_blk = 0
        while idx_blk < num_blocks:
            blk_in = tc_sigs[:, range(start_smpl, start_smpl+blocksize)]

            fd_sig_in[:] = self.hSTFT.forward(blk_in)

            doa[:], dif[:] = dequantize_dirac_pars(doa_idx_stream[idx_blk, ...],
                                                   dif_idx_stream[idx_blk, ...],
                                                   freqs, self.f_qt_c,
                                                   self.qgrid, self.qdifbins,
                                                   self.pars_smooth)

            M[:], Y[:] = compute_M_Y(doa, dif,
                                     self.N_sph_out, self.B_nm_exp, self.beta,
                                     self.num_recov, self.B_nm_trunc)

            if idx_blk == 0:
                M_prev[:] = M[:]
            M = (1. - self.pars_smooth) * M + self.pars_smooth * M_prev
            X_nm[:] = np.einsum('ldsk,dsk->dlk', M, fd_sig_in)

            gn = opt_gain(X_nm, Y, dif,
                          np.real((fd_sig_in * fd_sig_in.conj())),
                          self.C_dif, self.orne, self.M_mavg)
            np.clip(gn, 0.5, 2., out=gn)
            g = (1. - self.pars_smooth) * gn + self.pars_smooth * g

            X_nm[:, self.num_recov:, :] = np.repeat(g, self.num_m, axis=0)[
                np.newaxis, self.num_recov:, :] * X_nm[:, self.num_recov:, :]

            # back
            blk_out = self.hSTFT.backward(X_nm)
            out_sig[:, range(start_smpl, start_smpl+blocksize)] = blk_out
            M_prev[:] = M[:]
            start_smpl += blocksize
            idx_blk += 1

        out_sig = out_sig[:, self.hSTFT.processing_delay:]
        return out_sig


def cart2sph(x, y, z):
    """Vectorized conversion of cartesian to spherical coordinates."""
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    azi = np.arctan2(y, x)
    zen = np.arccos(z / r)
    return azi, zen, r


def sph2cart(azi, zen, r):
    """Vectorized conversion of spherical to cartesian coordinates."""
    x = r * np.cos(azi) * np.sin(zen)
    y = r * np.sin(azi) * np.sin(zen)
    z = r * np.cos(zen)
    return x, y, z


def cart2dir(x, y, z):
    """Vectorized conversion of cartesian to spherical coordinates."""
    return np.arctan2(y, x), \
        np.arccos(z/(np.sqrt(np.square(x) + np.square(y) + np.square(z))))


def dir2cart(azi, zen):
    """Vectorized conversion of spherical to cartesian coordinates."""
    return np.cos(azi) * np.sin(zen), np.sin(azi) * np.sin(zen), np.cos(zen)


def vec2dir(vec):
    """Conversion of cartesian to spherical coordinates (along last axis)."""
    azi, zen = cart2dir(vec[..., 0], vec[..., 1], vec[..., 2])
    return np.stack((azi, zen), axis=-1)


def estimate_sector_parameters(x_nm, A_wxyz_c, TRANSPOSE=False):
    """
    Sector S parameters from SH signals L, frequency K band.

    Parameters
    ----------
    x_nm : TYPE
        L x K.
    A_wxyz : complex
        4*S x L.

    Returns
    -------
    azi_s, zen_s, dif_s, ene_s, int_s : np.ndarray
        S x L, or 3*S x L, or transposed

    """
    num_secs = A_wxyz_c.shape[0] // 4
    x_s = A_wxyz_c @ x_nm

    sec_intensity = np.empty((3*num_secs, x_s.shape[1]))
    sec_energy = np.empty((num_secs, x_s.shape[1]))
    azi_s = np.empty((num_secs, x_s.shape[1]))
    zen_s = np.empty((num_secs, x_s.shape[1]))
    dif_s = np.empty((num_secs, x_s.shape[1]))
    r_s = np.empty(x_s.shape[1])

    for idx_sec in range(num_secs):
        s_sec = x_s[idx_sec*4: idx_sec*4+4, :]
        sec_intensity[idx_sec*3: idx_sec*3+3, :] = \
            np.real((s_sec[0, :]) * s_sec[1:4, :].conj())
        sec_intensity[idx_sec*3, :] += 10e-12
        sec_energy[idx_sec, :] = 0.5 * (np.abs(s_sec[0, :])**2 +
                                        np.sum(s_sec[1:4, :].conj() *
                                               s_sec[1:4, :], axis=0).real)
        azi_s[idx_sec, :], zen_s[idx_sec, :], r_s[:] = cart2sph(
            sec_intensity[idx_sec*3+0, :],
            sec_intensity[idx_sec*3+1, :],
            sec_intensity[idx_sec*3+2, :])
        dif_s[idx_sec, :] = np.clip(
            1 - (r_s / (sec_energy[idx_sec, :] + 10e-12)), 0., 1.)

    if TRANSPOSE:
        return azi_s.T, zen_s.T, dif_s.T, sec_energy.T, sec_intensity.T
    else:
        return azi_s, zen_s, dif_s, sec_energy, sec_intensity


def grouped_sector_parameters(x_nm, A_wxyz_c, M_grouper, TRANSPOSE=False):
    """
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

    """
    num_secs = A_wxyz_c.shape[0] // 4
    x_s = A_wxyz_c @ x_nm

    sec_intensity = np.empty((3*num_secs, x_s.shape[1]))
    sec_energy = np.empty((num_secs, x_s.shape[1]))
    num_fgroups = M_grouper.shape[1]
    int_s = np.empty((3*num_secs, num_fgroups))
    ene_s = np.empty((num_secs, num_fgroups))
    azi_s = np.empty((num_secs, num_fgroups))
    zen_s = np.empty((num_secs, num_fgroups))
    r_s = np.empty(num_fgroups)
    dif_s = np.empty((num_secs, num_fgroups))

    for idx_sec in range(num_secs):
        s_sec = x_s[idx_sec*4: idx_sec*4+4, :]
        sec_intensity[idx_sec*3: idx_sec*3+3, :] = \
            np.real((s_sec[0, :]) * s_sec[1:4, :].conj())
        sec_intensity[idx_sec*3, :] += 10e-12
        sec_energy[idx_sec, :] = 0.5 * (np.abs(s_sec[0, :])**2 +
                                        np.sum(s_sec[1:4, :].conj() *
                                               s_sec[1:4, :], axis=0).real)

        int_s[idx_sec*3: idx_sec*3+3, :] = \
            sec_intensity[idx_sec*3: idx_sec*3+3, :] @ M_grouper
        ene_s[idx_sec, :] = sec_energy[idx_sec, :] @ M_grouper
        azi_s[idx_sec, :], zen_s[idx_sec, :], r_s[:] = cart2sph(
            int_s[idx_sec*3+0, :],
            int_s[idx_sec*3+1, :],
            int_s[idx_sec*3+2, :])
        dif_s[idx_sec, :] = np.clip(
            1 - (r_s / (ene_s[idx_sec, :] + 10e-12)), 0., 1.)

    if TRANSPOSE:
        return azi_s.T, zen_s.T, dif_s.T, ene_s.T, int_s.T
    else:
        return azi_s, zen_s, dif_s, ene_s, int_s


def dir_mean(azi, zen, weights=None):
    """
    Directional mean.

    Parameters
    ----------
    azi : np.ndarray
    zen : np.ndarray
    weights : np.ndarray, optional
        Averaging weights. The default is None.

    Returns
    -------
    azi_m : np.ndarray
    zen_m : np.ndarray

    """
    x, y, z = np.cos(azi) * np.sin(zen), np.sin(azi) * np.sin(zen), np.cos(zen)
    x_m, y_m, z_m = np.average(x, weights=weights),\
        np.average(y, weights=weights), np.average(z, weights=weights)
    azi_m, zen_m = np.arctan2(y_m, x_m), np.arccos(z_m)
    return azi_m, zen_m


def group_dirac_pars(azi, zen, dif, M_grouper, weights):
    """
    Group DirAC parameters K into G groups.

    Parameters
    ----------
    azi : np.ndarray
    zen : np.ndarray
    dif : np.ndarray
    M_grouper : np.ndarray
        Matrix with K x G.
    weights : np.ndarray

    Returns
    -------
    azi_g : np.ndarray
    zen_g : np.ndarray
    dif_g : np.ndarray

    """
    x, y, z = dir2cart(azi, zen)
    xs, ys, zs = (x*weights)@M_grouper, \
                 (y*weights)@M_grouper, \
                 (z*weights)@M_grouper
    azi_g, zen_g = cart2dir(10e-12 + xs, ys, zs)
    dif_g = dif@M_grouper
    return azi_g, zen_g, dif_g


def post_pars(azi, zen, dif, ene, a=0.75):
    """
    Stabilizing DoA in high diffuseness (above factor a).

    Parameters
    ----------
    azi : np.ndarray
    zen : np.ndarray
    dif : np.ndarray
    a : float

    Returns
    -------
    azi : np.ndarray
    zen : np.ndarray

    """
    num_slt = azi.shape[0]
    for idx_slt in range(1, num_slt):
        mask = np.where(dif[idx_slt, ...] > a)
        azi[idx_slt, mask[0], mask[1]] = azi[idx_slt - 1, mask[0], mask[1]]
        zen[idx_slt, mask[0], mask[1]] = zen[idx_slt - 1, mask[0], mask[1]]
    return azi, zen, dif, ene


def get_quant_grid(n_fine, n_coarse=None):
    """
    Get quantization grid from spherical designs.

    Parameters
    ----------
    n_fine : int
        Order.
    n_coarse : bool, optional
        Prepend coarse grid. The default is None.

    Returns
    -------
    qgrid : np.ndarray
    num_coarse : int

    """
    # 38: 6.69, 48 : 5deg, 60, 66: 3.93 deg
    if n_coarse is None:
        qgrid = np.vstack(([1., 0., 0.], spa.grids.load_n_design(n_fine)))
        num_coarse = None
    else:
        grid_coarse = spa.grids.load_n_design(n_coarse)
        qgrid = np.vstack(([1., 0., 0.],
                           grid_coarse,
                           spa.grids.load_n_design(n_fine)))
        num_coarse = len(grid_coarse)
    return qgrid, num_coarse


def quantize_doa(azi, zen, qgrid, dif, coarse_th=None, num_coarse=None,
                 dtype=np.int16):
    """
    Quantize DoA parameters to quantization grid.

    Parameters
    ----------
    azi : np.ndarray
    zen : np.ndarray
    qgrid : np.ndarray
    dif : np.ndarray
    coarse_th : float, optional
        Threshold. The default is None.
    num_coarse : int, optional
        Number of course grid points. The default is None.
    dtype : dype, optional
        The default is np.int16.

    Returns
    -------
    out : np.ndarray

    """
    xq, yq, zq = dir2cart(azi, zen)
    v = np.stack((xq, yq, zq), axis=-1)
    p_all = v @ qgrid.T[np.newaxis, np.newaxis, :, :]
    out = np.empty_like(azi, dtype=dtype)
    out = np.argmax(p_all, axis=-1, out=out)
    if coarse_th is not None and num_coarse is not None:
        mask = dif > coarse_th
        p_coarse = v @ qgrid.T[np.newaxis, np.newaxis, :, :num_coarse+1]
        out[mask[:]] = np.argmax(p_coarse[mask[:]], axis=-1)
    out[dif > 0.95] = dtype(0)
    return out


def quantize_dif(dif, qbins, kernel_size=3, dtype=np.uint8):
    """
    Quantize diffuseness parameter.

    Parameters
    ----------
    dif : np.ndarray
    qbins : int
    kernel_size : int, optional
        Median filter kernel size. The default is 3.
    dtype : dtype, optional
        The default is np.uint8.

    Returns
    -------
    out : np.ndarray

    """
    dif_filtered = median_filter(dif, size=kernel_size, axes=0)
    dif_filtered[dif_filtered > 0.95] = 1.
    return np.searchsorted(qbins, dif_filtered).astype(dtype)


def downsample_meta(doa_idx_stream, dif_idx_stream, user_pars):
    """
    Downsample metadata (by zeroing for now).

    Parameters
    ----------
    doa_idx_stream : np.ndarray
    dif_q_stream : np.ndarray
    user_pars : struct

    Returns
    -------
    doa_idx_stream : np.ndarray
    dif_q_stream : np.ndarray

    """
    if user_pars['metaDecimate'] >= 1:
        # no information in DC
        doa_idx_stream[:, :, :, 0] = 0
        dif_idx_stream[:, :, :, 0] = user_pars['metaDifBins']

        mask = np.ones_like(doa_idx_stream).astype(np.bool_)
        mask[:, 1::user_pars['metaDecimate'], :,
             :user_pars['metaDecimateFreqLim']] = False
        doa_idx_stream[~mask] = 0
        dif_idx_stream[~mask] = user_pars['metaDifBins']
    return doa_idx_stream, dif_idx_stream


def dequantize_dirac_pars(doa_idx_stream, dif_idx_stream, freqs, f_qt_c, qgrid,
                          qdifbins, a=0.33):
    """
    Dequantize / interpolate DirAC parameters.

    Parameters
    ----------
    doa_idx_stream : np.ndarray, [slt, ch, :]
    dif_idx_stream : np.ndarray, [slt, ch, :]
    freqs : np.ndarray
    f_qt_c : np.ndarray
    qgrid : np.ndarray
    qdifbins : np.ndarray
    a1 : float, optional

    Returns
    -------
    doa_s : np.ndarray, [slt, ch, :, 3]
    dif_s : np.ndarray, [slt, ch, :]

    """
    num_slt = doa_idx_stream.shape[0]
    num_ch = doa_idx_stream.shape[1]
    doa = np.empty((num_slt, num_ch, len(freqs), 3), dtype=np.double)
    dif = np.empty((num_slt, num_ch, len(freqs)), dtype=np.double)
    a1 = 1. - a
    a2 = a

    for idx_slt in range(num_slt):
        for idx_ch in range(num_ch):
            doa[idx_slt, idx_ch, :, 0] = np.interp(freqs, f_qt_c,
                                                   qgrid[doa_idx_stream[idx_slt, idx_ch, :], 0])
            doa[idx_slt, idx_ch, :, 1] = np.interp(freqs, f_qt_c,
                                                   qgrid[doa_idx_stream[idx_slt, idx_ch, :], 1])
            doa[idx_slt, idx_ch, :, 2] = np.interp(freqs, f_qt_c,
                                                   qgrid[doa_idx_stream[idx_slt, idx_ch, :], 2])

            dif[idx_slt, idx_ch, :] = np.interp(freqs, f_qt_c,
                                                qdifbins[dif_idx_stream[idx_slt, idx_ch, :]])
    doa_s = a1 * doa
    doa_s[0, ...] = doa[0, ...]
    doa_s[1:, ...] += a2 * doa[:-1, ...]
    dif_s = a1 * dif
    dif_s[0, ...] = dif[0, ...]
    dif_s[1:, ...] += a2 * dif[:-1, ...]
    return doa_s, dif_s


def compute_M_Y(doa, dif, N_sph, B_nm_exp, beta, num_recov, B_nm_low):
    """
    Get mixing matrix M and SH expansion Y.

    Parameters
    ----------
    doa : np.ndarray
    dif : np.ndarray
    N_sph : int
    B_nm_exp : np.ndarray
    beta : np.ndarray
    num_recov : int
    B_nm_low : np.ndarray

    Returns
    -------
    M : np.ndarray
    Y : np.ndarray

    References
    ----------
    C. Hold, L. McCormack, A. Politis and V. Pulkki, "Optimizing Higher-Order
    Directional Audio Coding with Adaptive Mixing and Energy Matching for
    Ambisonic Compression and Upmixing," 2023 IEEE WASPAA.

    """
    num_slt = doa.shape[0]
    num_bands = doa.shape[2]
    num_ch = doa.shape[1]
    azi, zen = cart2dir(doa[..., 0], doa[..., 1], doa[..., 2])
    v_dir = np.stack((np.reshape(azi, -1), np.reshape(zen, -1)), axis=1)
    Y_ = safpy.sh.getSHreal_part(int(np.sqrt(num_recov) - 1), N_sph, v_dir)
    Y = Y_.reshape((N_sph+1)**2, num_slt, num_ch, num_bands)
    M = beta[np.newaxis, :, np.newaxis] * (1 - dif) * Y + dif * B_nm_exp
    M[:num_recov, ...] = B_nm_low
    return M, Y


def opt_gain(X_nm, Y, dif, ene_s, C_f_dif, orne, M_mavg):
    """
    Post processing optimal mix/match gain to spatial model covariance.

    Parameters
    ----------
    X_nm : np.ndarray
    Y : np.ndarray
    dif : np.ndarray
    ene_s : np.ndarray
    C_f_dif : np.ndarray
    orne : float
    M_mavg : np.ndarray

    Returns
    -------
    gp : np.ndarray

    References
    ----------
    C. Hold, L. McCormack, A. Politis and V. Pulkki, "Optimizing Higher-Order
    Directional Audio Coding with Adaptive Mixing and Energy Matching for
    Ambisonic Compression and Upmixing," 2023 IEEE WASPAA.

    """
    num_slt = X_nm.shape[0]
    num_sh = X_nm.shape[1]
    ene_dir = (1-dif) * ene_s
    ene_dif = dif * ene_s
    # Cyd = 4*np.pi/(8*num_sh_out) * np.sum(Y * ene_dir[np.newaxis, ...] * Y,
    #                                       axis=(2, 3))
    Cyd = 4*np.pi/(num_slt*num_sh) * np.einsum('ldsk,ldsk->lk',
                                               Y * ene_dir, Y)
    Cyd += 1/(num_slt*num_sh) * np.einsum('dsk,sll->lk', ene_dif, C_f_dif)
    Cyd *= orne

    Cyn = M_mavg @ Cyd

    Cxn = 1/num_slt * (M_mavg @ np.real(np.einsum('dlk,dlk->lk',
                                                  X_nm, X_nm.conj())))
    gp = np.sqrt(Cyn / (10e-10 + Cxn))
    return gp


# PARAMETERIZATION
def get_f_quantizer(num_bands, DEFAULT=True):
    """
    Get default frequency band quantizer.

    Parameters
    ----------
    num_bands : int
        DESCRIPTION.
    DEFAULT : bool, optional
        False switches to log spaced. The default is True.

    Returns
    -------
    f_qt : list of tuples

    """
    if DEFAULT:
        f_qt = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 10),
                (10, 15), (15, 20), (20, 25), (25, 30),
                (30, 38), (38, 50), (50, 75), (75, 100), (100, num_bands)]
    else:
        s = np.append([1], np.ceil(np.geomspace(2, num_bands, 15)).astype(int))
        s0 = np.append([0], s)
        f_qt = [(s0[idx], s[idx]) for idx in range(len(s))]
    return f_qt


def get_f_grouper(f_qt):
    """
    Get frequency band grouping matrix.

    Parameters
    ----------
    f_qt : list[num_fgroups] of tuples(start_idx, end_idx)

    Returns
    -------
    M_grouper : np.ndarray


    """
    num_fgroups = len(f_qt)
    num_bands = max(max(f_qt)) - min(min(f_qt))
    M_grouper = np.zeros((num_fgroups, num_bands))
    for group, qt in enumerate(f_qt):
        M_grouper[group, qt[0]:qt[1]] = 1
        M_grouper[group, :] /= np.sum(M_grouper[group, :])
    return M_grouper.T


def get_C_weighting(freqs):
    """
    Get C weighting for frequency weighting.

    Parameters
    ----------
    freqs : np.ndarray

    Returns
    -------
    r_C : np.ndarray

    """
    r_C = (12194**2 * (freqs+1)**2) / \
        (((freqs+1)**2 + 20.6**2) *
         ((freqs+1)**2 + 12194**2))
    return r_C


def get_num_sh_recov(A_nm, B_nm):
    """
    Estimate number of recovered SH channels.

    Parameters
    ----------
    A_nm : np.ndarray
    B_nm : np.ndarray

    Returns
    -------
    num_recov : int

    """
    d_recov = np.diag(B_nm @ A_nm)
    if np.min(d_recov > 0.9):
        num_recov = len(d_recov)
    else:
        num_recov = np.argmax(d_recov <= 0.9)
    return num_recov


def get_cov_dif(N_sph, num_ch, conf):
    """
    Specify model covariance for diffuse components.

    Parameters
    ----------
    N_sph : int
    num_ch : int
    conf : struct
        Configuration struct.

    Returns
    -------
    C_dif : np.ndarray
        Covariance matrix stacked as [num_ch, num_sh, num_sh].

    """
    num_sh = (N_sph + 1)**2
    y_sec = spa.sph.sh_matrix(N_sph, conf['tc_v'][0], conf['tc_v'][1])
    C_dif = np.ones((num_ch, num_sh, num_sh))
    C_dif = 4*np.pi * np.array([np.outer(y_sec[s, :], y_sec[s, :])
                                for s in range(num_ch)])
    return C_dif


def sph_filterbank_reconstruction(A_nm):
    """
    Complementary spherical filterbank reconstruction of A.

    Parameters
    ----------
    A_nm : np.ndarray

    Returns
    -------
    B_nm : np.ndarray
    B_nm_trunc : np.ndarray
    num_recov : int

    References
    ----------
    C. Hold, V. Pulkki, A. Politis and L. McCormack, "Compression of
    Higher-Order Ambisonic Signals Using Directional Audio Coding," in
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
    """
    B_nm = np.linalg.pinv(A_nm)
    num_recov = get_num_sh_recov(A_nm, B_nm)
    B_nm_trunc = np.linalg.pinv(A_nm[:, :num_recov])
    return B_nm, B_nm_trunc, num_recov


# WRITE
def encode_pars(doa_q_stream, dif_q_stream):
    """
    Write parameter stream.

    Parameters
    ----------
    doa_q_stream : array_like
        DESCRIPTION.
    dif_q_stream : array_like
        DESCRIPTION.

    Returns
    -------
    data_pars_status : TYPE
        DESCRIPTION.
    data_pars_stream : TYPE
        DESCRIPTION.

    """
    data_pars_stream = bz2.compress(np.asarray([doa_q_stream, dif_q_stream]))
    return data_pars_stream


def encode_tcs(tc_sigs, tc_bitrate, fs):
    """
    Write transport audio channels stream.

    Parameters
    ----------
    tc_sigs : np.ndarray
        DESCRIPTION.
    user_pars : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.

    Returns
    -------
    data_tcs : TYPE
        DESCRIPTION.
    enc_lookahead : TYPE
        DESCRIPTION.

    """
    num_ch = tc_sigs.shape[0]
    num_samples = tc_sigs.shape[1]
    assert fs == 48000, "Opus expected 48kHz, please resample."
    mapping = list(range(num_ch))
    enc = pylibopus.MultiStreamEncoder(fs, num_ch, num_ch, 0, mapping,
                                       pylibopus.APPLICATION_AUDIO)
    enc.bitrate = int(tc_bitrate * 1000 * num_ch)
    enc.complexity = 10
    enc_lookahead = enc.lookahead  # check last
    frame_size = 960
    assert frame_size >= enc_lookahead

    audio_in = 1/(np.sqrt(4*np.pi)) * tc_sigs.T
    audio_in = np.append(audio_in, np.zeros((frame_size, num_ch)), axis=0)
    sample_idx = 0
    opus_data = []
    package_idx = 0

    if np.max(np.abs(audio_in)) > 1.0:
        warn("Audio TCs clipping!")

    while sample_idx + frame_size <= num_samples + frame_size:
        opus_package = enc.encode_float(
            audio_in[sample_idx:sample_idx+frame_size, :].astype(np.float32).tobytes(),
            frame_size)
        opus_data.append(opus_package)
        sample_idx += frame_size
        package_idx += 1

    return opus_data, enc_lookahead


def write_hoac(pars_status, data_tcs, data_pars_stream, file):
    """
    Write HOAC file.

    Parameters
    ----------
    pars_status : TYPE
        DESCRIPTION.
    data_tcs : TYPE
        DESCRIPTION.
    data_pars_stream : TYPE
        DESCRIPTION.
    file : Path
        HOAC file.

    Returns
    -------
    None.

    """
    pars_status['hoac_version'] = get_version()

    with open(file, "wb") as f:
        pickle.dump(pars_status, f)
        pickle.dump(data_pars_stream, f)
        pickle.dump(data_tcs, f)


def read_hoac(file):
    """
    Read HOAC file.

    Parameters
    ----------
    file : Path
        HOAC file.

    Returns
    -------
    conf : TYPE
        DESCRIPTION.
    c_data_tcs : TYPE
        DESCRIPTION.
    c_data_pars : TYPE
        DESCRIPTION.

    """
    with open(file, 'rb') as f_hoac:
        conf = pickle.load(f_hoac)
        c_data_pars = pickle.load(f_hoac)
        c_data_tcs = pickle.load(f_hoac)

        assert (conf['hoac_version'] == get_version())

    return conf, c_data_tcs, c_data_pars


def decode_pars(conf, c_pars):
    num_slots = conf['blocksize'] // conf['hopsize']
    pars = np.reshape(np.frombuffer(bz2.decompress(c_pars),
                                    dtype=np.int16),
                        (2, -1, num_slots, conf['numTC'],
                        conf['numFreqs'])).copy()

    if conf['metaDecimate'] > 1:
        pars[:, :, :, :, 0] = pars[:, :, :, :, 1]  # no information in DC
        # pars = np.repeat(pars, conf['metaDecimate'], axis=2)  # upsample
        pars_f = pars.copy()
        pars_lo = pars_f[:, :, ::conf['metaDecimate'], :, :conf['metaDecimateFreqLim']]
        pars_lo = np.repeat(pars_lo, conf['metaDecimate'], axis=2)  # upsample
        pars = np.concatenate((pars_lo, pars_f[:, :, :, :, conf['metaDecimateFreqLim']:]), axis=-1)

    doa_idx = pars[0, ...]
    dif_idx = pars[1, ...]
    return doa_idx, dif_idx


def decode_tcs(conf, c_data_tcs):
    print("Decoding Audio")
    num_frames = len(c_data_tcs)

    num_ch = conf['numTC']
    assert (conf['fs'] == 48000)
    fs = conf['fs']

    frame_size = 960
    mapping = list(range(num_ch))
    dec = pylibopus.MultiStreamDecoder(fs, num_ch, num_ch, 0, mapping)
    enc_lookahead = conf['enc_lookahead']

    audio_out = np.zeros((frame_size * num_frames + enc_lookahead, num_ch))
    sample_idx = 0

    for package_idx in range(num_frames):
        opus_package = c_data_tcs[package_idx]
        b_res = dec.decode_float(opus_package, frame_size)
        res = np.frombuffer(b_res, dtype=np.float32)

        audio_out[sample_idx:sample_idx+frame_size, :] = res.reshape((frame_size, num_ch))
        sample_idx += frame_size
    audio_out = audio_out[enc_lookahead: -frame_size, :]
    
    audio_out = np.sqrt(4*np.pi)*audio_out.T
    return audio_out
