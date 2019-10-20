import numpy as np
import pyworld
import pysptk

from hparams import hparams

class World(object):
    """WORLD-based speech analyzer
    Parameters
    ----------
    fs : int, optional
        Sampling frequency
        Default set to 16000
    fftl : int, optional
        FFT length
        Default set to 1024
    shiftms : int, optional
        Shift lengs [ms]
        Default set to 5.0
    minf0 : int, optional
        Floor in f0 estimation
        Default set to 50
    maxf0 : int, optional
        Ceil in f0 estimation
        Default set to 500
    """

    def __init__(self):
        self.fs = hparams.fs
        self.fftl = hparams.fftl
        self.shiftms = hparams.shiftms
        self.minf0 = hparams.minf0
        self.maxf0 = hparams.maxf0

    def analyze(self, x):
        """Analyze acoustic features based on WORLD
        analyze F0, spectral envelope, aperiodicity
        Paramters
        ---------
        x : array, shape (`T`)
            monoral speech signal in time domain
        Returns
        ---------
        f0 : array, shape (`T`,)
            F0 sequence
        spc : array, shape (`T`, `fftl / 2 + 1`)
            Spectral envelope sequence
        ap: array, shape (`T`, `fftl / 2 + 1`)
            aperiodicity sequence
        """
        f0, time_axis = pyworld.harvest(x, self.fs, f0_floor=self.minf0,
                                        f0_ceil=self.maxf0, frame_period=self.shiftms)
        spc = pyworld.cheaptrick(x, f0, time_axis, self.fs,
                                 fft_size=self.fftl)
        ap = pyworld.d4c(x, f0, time_axis, self.fs, fft_size=self.fftl)

        assert spc.shape == ap.shape
        return f0, spc, ap

    def synthesis(self, f0, spc, ap):
        """Synthesis re-synthesizes a speech waveform from:
        Parameters
        ----------
        f0 : array, shape (`T`)
            F0 sequence
        spc : array, shape (`T`, `dim`)
            Spectral envelope sequence
        ap: array, shape (`T`, `dim`)
            Aperiodicity sequence
        """

        return pyworld.synthesize(f0, spc, ap, self.fs, frame_period=self.shiftms)
    
    def synthesis_from_mcep(self, f0, mcep, ap, rmcep=None, alpha=0.42):
        """synthesis generates waveform from F0, mcep, aperiodicity
        Parameters
        ----------
        f0 : array, shape (`T`, `1`)
            array of F0 sequence
        mcep : array, shape (`T`, `dim`)
            array of mel-cepstrum sequence
        ap : array, shape (`T`, `fftlen / 2 + 1`) or (`T`, `dim_codeap`)
            array of aperiodicity or code aperiodicity
        rmcep : array, optional, shape (`T`, `dim`)
            array of reference mel-cepstrum sequence
            Default set to None
        alpha : int, optional
            Parameter of all-path transfer function
            Default set to 0.42
        Returns
        ----------
        wav: array,
            Synethesized waveform
        """

        if rmcep is not None:
            # power modification
            mcep = mod_power(mcep, rmcep, alpha=alpha)

        if ap.shape[1] < self.fftl // 2 + 1:
            # decode codeap to ap
            ap = pyworld.decode_aperiodicity(ap, self.fs, self.fftl)

        # mcep into spc
        spc = pysptk.mc2sp(mcep, alpha, self.fftl)

        # generate waveform using world vocoder with f0, spc, ap
        wav = pyworld.synthesize(f0, spc, ap,
                                 self.fs, frame_period=self.shiftms)

        return wav
    
    def mcep_from_spec(self, spc, dim=hparams.num_mcep, alpha=0.42):

        return pysptk.sp2mc(spc, dim, alpha)
    
    def spec_from_mcep(self, mcep, alpha=0.42):
        
        spc = pysptk.mc2sp(mcep, alpha, self.fftl)
        
        return spc