import numpy as np
import pyworld
import pysptk
import librosa
import os
import glob

import torch

from hparams import hparams

class Converter(object):
    
    def __init__(self, root_dir, speakers):
        self.root_dir = root_dir
        self.speakers = speakers
        self.norm_dict = self.normalizer_dict()
        self.world = World()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def normalizer_dict(self):
        d = {}
        for speaker in self.speakers:
            p = os.path.join(os.path.join(self.root_dir, speaker), speaker + "_norm.npz")
            try:
                stat_filepath = [fn for fn in glob.glob(p) if speaker in fn][0]
            except:
                raise Exception('====no match files!====')
            t = np.load(stat_filepath)
            d[speaker] = t

        return d
    
    def pitch_conversion(self, f0, source_speaker, target_speaker):
        if type(source_speaker) is int:
            source_speaker = self.speakers[source_speaker]
            target_speaker = self.speakers[target_speaker]
        mean_log_src = self.norm_dict[source_speaker]['log_f0s_mean']
        std_log_src = self.norm_dict[source_speaker]['log_f0s_std']

        mean_log_target = self.norm_dict[target_speaker]['log_f0s_mean']
        std_log_target = self.norm_dict[target_speaker]['log_f0s_std']

        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)
        return f0_converted

    def forward_process(self, x, speakername):
        if type(speakername) is int:
            speakername = self.speakers[speakername]
        mean = self.norm_dict[speakername]['mceps_mean']
        std = self.norm_dict[speakername]['mceps_std']
        mean = np.reshape(mean, [-1,1])
        std = np.reshape(std, [-1,1])
        x = (x - mean) / std
        return x

    def backward_process(self, x, speakername):
        if type(speakername) is int:
            speakername = self.speakers[speakername]
        mean = self.norm_dict[speakername]['mceps_mean']
        std = self.norm_dict[speakername]['mceps_std']
        mean = np.reshape(mean, [-1,1])
        std = np.reshape(std, [-1,1])
        x = x * std + mean
        return x
    
    def generate(self, generator, source_label, target_label, save_path, save_original=False):
        
        if type(source_label) is int:
            source_speaker = self.speakers[source_label]
            target_speaker = self.speakers[target_label]
        elif type(source_label) is str:
            source_speaker = source_label
            target_speaker = target_label
            source_label = self.speakers.index(source_speaker)
            target_label = self.speakers.index(target_speaker)

        print("Convert from '{}' to '{}'".format(source_speaker, target_speaker))

        
        data_dir = os.path.join(self.root_dir, source_speaker)
        file_path = os.path.join(data_dir, os.listdir(data_dir)[np.random.randint(0,100)])
        
        with torch.no_grad():
            
            wav, _ = librosa.load(file_path, sr=hparams.fs)
            if len(wav) < 100:
                return
            wav, _ = librosa.effects.trim(wav)
            wav = wav.astype(np.double)
            f0, spec, ap = self.world.analyze(wav)
            mcep = self.world.mcep_from_spec(spec)
            mcep = np.concatenate([mcep, np.zeros((128, hparams.num_mcep+1))])
            mcep = mcep.reshape(mcep.shape[0], mcep.shape[1], 1)
            mcep = mcep.transpose((2, 1, 0))
            
            source_label = torch.tensor(source_label, dtype=torch.long).view(1)
            target_label = torch.tensor(target_label, dtype=torch.long).view(1)
            source_label, target_label = source_label.to(self.device), target_label.to(self.device)
            
            convert_result = []
            
            for start_idx in range(0, mcep.shape[2] - 128 + 1, 128):

                seg = mcep[:, :, start_idx : start_idx+128]
                seg = self.forward_process(seg, source_speaker)
                seg = torch.FloatTensor(seg)
                seg = seg.view(1,1,seg.size(1),seg.size(2))
                
                seg = seg.to(self.device)

                outputs = generator(seg, source_label, target_label).data.cpu().numpy()
                outputs = np.squeeze(outputs)
                outputs = self.backward_process(outputs, target_speaker)
                convert_result.append(outputs)

            if len(convert_result) == 1:
                mcep_converted = np.array(convert_result)
            else:
                mcep_converted = np.concatenate(convert_result, axis=1)
            mcep_converted = mcep_converted.transpose((1,0))
            mcep_converted = mcep_converted[:f0.shape[0]]
            mcep_converted = np.ascontiguousarray(mcep_converted)
            f0_converted = self.pitch_conversion(f0, source_speaker, target_speaker)
            
            if save_original:
                librosa.output.write_wav(save_path+"_original.wav", wav, hparams.fs) 
            
            wav = self.world.synthesis_from_mcep(f0_converted, mcep_converted, ap)
            librosa.output.write_wav(save_path, wav, hparams.fs)     


    
    
    

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
            mcep = self.mod_power(mcep, rmcep, alpha=alpha)

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
    
    def mod_power(self, cvmcep, rmcep, alpha=0.42, irlen=1024):
        """Power modification based on inpulse responce
        Parameters
        ----------
        cvmcep : array, shape (`T`, `dim`)
            array of converted mel-cepstrum
        rmcep : array, shape (`T`, `dim`)
            array of reference mel-cepstrum
        alpha : float, optional
            All-path filter transfer function
            Default set to 0.42
        irlen : int, optional
            Length for IIR filter
            Default set to 1024
        Return
        ------
        modified_cvmcep : array, shape (`T`, `dim`)
            array of power modified converted mel-cepstrum
        """

        if rmcep.shape != cvmcep.shape:
            raise ValueError("The shapes of the converted and \
                             reference mel-cepstrum are different: \
                             {} / {}".format(cvmcep.shape, rmcep.shape))

        cv_e = pysptk.mc2e(cvmcep, alpha=alpha, irlen=irlen)
        r_e = pysptk.mc2e(rmcep, alpha=alpha, irlen=irlen)

        dpow = np.log(r_e / cv_e) / 2

        modified_cvmcep = np.copy(cvmcep)
        modified_cvmcep[:, 0] += dpow

        return modified_cvmcep