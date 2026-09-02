# -*- coding: utf-8 -*-
"""Standalone, framework-free extraction of v2e's (SensorsINI/v2e) DVS
simulator core, `EventEmulator`.

Source: v2ecore/emulator.py (authors: Tobi Delbruck, Yuhuang Hu, Zhe He),
paired with v2e_emulator_utils.py (extracted from v2ecore/emulator_utils.py).

What was stripped (plumbing/output only, no simulation-math change):
  - The three optional file-output writers (`AEDat2Output`, `AEDat4Output`,
    `DVSTextOutput`, and raw HDF5 dataset writing) and their constructor
    args (`dvs_h5`, `dvs_aedat2`, `dvs_aedat4`, `dvs_text`, `output_folder`,
    `prepare_storage()`). We consume events in-memory via
    `event_to_frame.events_to_frames()`, never write them to disk in any of
    v2e's own formats -- and `AEDat4Output` unconditionally imports
    `dv_processing`, a compiled binding for real iniVation event-camera
    hardware with no reason to install for a software-only simulated
    pipeline.
  - Debug/display: `_show()`, `show_dvs_model_state`, `save_dvs_model_state`,
    and the `screeninfo`-based window placement in `__init__`. This project
    trains headless (`MUJOCO_GL=egl` on the cluster) where OpenCV GUI
    windows have nothing to attach to; the original guards this at
    call-time, not import-time, but `screeninfo` and unused `cv2` GUI calls
    are simply dead weight here either way.
  - `record_single_pixel_states` (pickles per-pixel debug traces to disk)
    and `label_signal_noise` (tags each event signal-vs-noise for those same
    output writers) -- both unused without the writers above.
  - `atexit`-registered `cleanup()` -- had nothing left to clean up once the
    output writers were removed.

Everything else -- lin-log intensity mapping, per-pixel threshold mismatch,
IIR photoreceptor lowpass, leak current, shot noise, refractory period,
sub-frame event timestamp interpolation, center-surround (csdvs) and SCIDVS
adaptation -- is copied verbatim from the original repo. See
`generate_events()`'s docstring for the exact return contract this project's
own `v2e_events.py` (the actual RGB-to-event swap point) relies on.
"""
import logging
import math
import random
from typing import Optional

import torch

from v2e_emulator_utils import compute_event_map, compute_photoreceptor_noise_voltage
from v2e_emulator_utils import generate_shot_noise
from v2e_emulator_utils import lin_log
from v2e_emulator_utils import low_pass_filter
from v2e_emulator_utils import rescale_intensity_frame
from v2e_emulator_utils import subtract_leak_current

logger = logging.getLogger(__name__)


class EventEmulator(object):
    """compute events based on the input frame.
    - original author: Tobi Delbruck, Yuhuang Hu, Zhe He
    - contact: tobi@ini.uzh.ch
    """

    MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING = 1e-5

    # scidvs adaptation
    def scidvs_dvdt(self, v, tau=None):
        """
        Parameters
        ----------
            the input 'voltage',
        v:Tensor
            actually log intensity in base e units
        tau:Optional[Tensor]
            if None, tau is set internally

        Returns
        -------
        the time derivative of the signal
        """
        if tau is None:
            tau = EventEmulator.SCIDVS_TAU_S  # time constant for small signals = C/g
        efold = 1 / 0.7  # efold of sinh conductance in log_e units, based on 1/kappa
        dvdt = torch.div(1,tau) * torch.sinh(v / efold)
        return dvdt

    SCIDVS_GAIN: float = 2  # gain after highpass
    SCIDVS_TAU_S: float = .01  # small signal time constant in seconds
    SCIDVS_TAU_COV: float = 0.5  # each pixel has its own time constant. The tau's have log normal distribution with this sigma

    def __init__(
            self,
            pos_thres: float = 0.2,
            neg_thres: float = 0.2,
            sigma_thres: float = 0.03,
            cutoff_hz: float = 0.0,
            leak_rate_hz: float = 0.1,
            refractory_period_s: float = 0.0,
            shot_noise_rate_hz: float = 0.0,  # rate in hz of temporal noise events
            photoreceptor_noise: bool = False,
            leak_jitter_fraction: float = 0.1,
            noise_rate_cov_decades: float = 0.1,
            seed: int = 0,
            device: str = "cpu",
            cs_lambda_pixels: Optional[float] = None,
            cs_tau_p_ms: Optional[float] = None,
            hdr: bool = False,
            scidvs: bool = False,
    ):
        """
        Parameters
        ----------
        pos_thres: float, default 0.2
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.2
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float,
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: float
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz: float
            shot noise rate in Hz
        photoreceptor_noise: bool
            model photoreceptor noise to create the desired shot noise rate
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        device: str
            'cpu' or 'cuda'
        cs_lambda_pixels: float
            space constant of surround in pixels, or None to disable surround inhibition
        cs_tau_p_ms: float
            time constant of lowpass filter of surround in ms or 0 to make surround 'instantaneous'
        hdr: bool
            Treat input as HDR floating point logarithmic gray scale with 255 input scaled as ln(255)=5.5441
        scidvs: bool
            Simulate the high gain adaptive photoreceptor SCIDVS pixel
        """

        self.no_events_warning_count = 0
        logger.info(
            "ON/OFF log_e temporal contrast thresholds: "
            "{} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres))

        self.reset()
        self.t_previous = 0  # time of previous frame

        # torch device
        self.device = device

        # thresholds
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres

        # non-idealities
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.photoreceptor_noise = photoreceptor_noise
        self.photoreceptor_noise_vrms: Optional[float] = None
        self.photoreceptor_noise_arr: Optional[torch.Tensor] = None
        if photoreceptor_noise:
            if shot_noise_rate_hz == 0:
                raise ValueError(
                    'photoreceptor_noise is specified but shot_noise_rate_hz is 0; '
                    'set a finite rate of shot noise events per pixel')
            if cutoff_hz == 0:
                raise ValueError(
                    'photoreceptor_noise is specified but cutoff_hz is zero; '
                    'set a finite photoreceptor cutoff frequency')
            self.photoreceptor_noise_samples = []

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.SHOT_NOISE_INTEN_FACTOR = 0.25 # this factor models the slight increase of shot noise with intensity

        # generate jax key for random process
        if seed != 0:
            torch.manual_seed(seed)
            random.seed(seed)

        # event stats
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0

        # csdvs
        self.cs_steps_warning_printed = False
        self.cs_steps_taken = []
        self.cs_alpha_warning_printed = False
        self.cs_tau_p_ms = cs_tau_p_ms
        self.cs_lambda_pixels = cs_lambda_pixels
        self.cs_surround_frame: Optional[torch.Tensor] = None  # surround frame state
        self.csdvs_enabled = False  # flag to run center surround DVS emulation
        if self.cs_lambda_pixels is not None:
            self.csdvs_enabled = True
            # prepare kernels
            self.cs_tau_h_ms = 0 \
                if (self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) \
                else self.cs_tau_p_ms / (self.cs_lambda_pixels ** 2)
            self.cs_k_hh = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]], dtype=torch.float32).to(self.device)

        self.log_input = hdr
        if self.log_input:
            logger.info('Treating input as log-encoded HDR input')

        self.scidvs = scidvs
        if self.scidvs:
            logger.info('Modeling potential SCIDVS pixel with nonlinear CR highpass amplified log intensity')

    def cleanup(self):
        if len(self.cs_steps_taken) > 1:
            import numpy as np
            mean_staps = np.mean(self.cs_steps_taken)
            std_steps = np.std(self.cs_steps_taken)
            median_steps = np.median(self.cs_steps_taken)
            logger.info(
                f'CSDVS steps statistics: mean+std= {mean_staps:.0f} + {std_steps:.0f} (median= {median_steps:.0f})')

    def _init(self, first_frame_linear):
        """
        Parameters:
        ----------
        first_frame_linear: np.ndarray
            the first frame, used to initialize data structures
        """
        logger.debug(
            'initializing random temporal contrast thresholds '
            'from from base frame')
        self.diff_frame = None

        # take the variance of threshold into account.
        if self.sigma_thres > 0:
            self.pos_thres = torch.normal(
                self.pos_thres, self.sigma_thres,
                size=first_frame_linear.shape,
                dtype=torch.float32).to(self.device)
            self.pos_thres = torch.clamp(self.pos_thres, min=0.01)

            self.neg_thres = torch.normal(
                self.neg_thres, self.sigma_thres,
                size=first_frame_linear.shape,
                dtype=torch.float32).to(self.device)
            self.neg_thres = torch.clamp(self.neg_thres, min=0.01)

        self.pos_thres_pre_prob = torch.div(
            self.pos_thres_nominal, self.pos_thres)
        self.neg_thres_pre_prob = torch.div(
            self.neg_thres_nominal, self.neg_thres)

        if self.scidvs and EventEmulator.SCIDVS_TAU_COV > 0:
            self.scidvs_tau_arr = EventEmulator.SCIDVS_TAU_S * (
                torch.exp(torch.normal(0, EventEmulator.SCIDVS_TAU_COV, size=first_frame_linear.shape,
                                       dtype=torch.float32).to(self.device)))

        if self.leak_rate_hz > 0:
            self.noise_rate_array = torch.randn(
                first_frame_linear.shape, dtype=torch.float32,
                device=self.device)
            self.noise_rate_array = torch.exp(
                math.log(10) * self.noise_rate_cov_decades * self.noise_rate_array)

        if self.refractory_period_s > 0:
            self.timestamp_mem = torch.zeros(
                first_frame_linear.shape, dtype=torch.float32,
                device=self.device) - self.refractory_period_s

    def set_dvs_params(self, model: str):
        if model == 'clean':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.02
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.leak_jitter_fraction = 0
            self.noise_rate_cov_decades = 0
            self.shot_noise_rate_hz = 0
            self.refractory_period_s = 0
        elif model == 'noisy':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.05
            self.cutoff_hz = 30
            self.leak_rate_hz = 0.1
            self.shot_noise_rate_hz = 5.0
            self.refractory_period_s = 0
            self.leak_jitter_fraction = 0.1
            self.noise_rate_cov_decades = 0.1
        else:
            logger.warning(
                "dvs_params {} not known: "
                "Using commandline assigned options".format(model))

    def reset(self):
        '''resets so that next use will reinitialize the base frame
        '''
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0

        self.new_frame: Optional[torch.Tensor] = None
        self.log_new_frame: Optional[torch.Tensor] = None
        self.lp_log_frame: Optional[torch.Tensor] = None
        self.cs_surround_frame: Optional[torch.Tensor] = None
        self.c_minus_s_frame: Optional[torch.Tensor] = None
        self.base_log_frame: Optional[torch.Tensor] = None
        self.diff_frame: Optional[torch.Tensor] = None
        self.scidvs_highpass: Optional[torch.Tensor] = None
        self.scidvs_previous_photo: Optional[torch.Tensor] = None
        self.scidvs_tau_arr: Optional[torch.Tensor] = None

        self.frame_counter = 0

    def generate_events(self, new_frame, t_frame):
        """Compute events in new frame.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width], NOTE y is first dimension, like in matlab the column, x is 2nd dimension, i.e. row.
        t_frame: float
            timestamp of new frame in float seconds

        Returns
        -------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
            NOTE x,y, NOT y,x.
        """
        self.frame_counter += 1

        if t_frame < self.t_previous:
            raise ValueError(
                "this frame time={} must be later than "
                "previous frame time={}".format(t_frame, self.t_previous))

        delta_time = t_frame - self.t_previous

        if self.log_input and new_frame.dtype != torch.float32:
            logger.warning('log_frame is True but input frome is not np.float32 datatype')

        self.new_frame = torch.tensor(new_frame, dtype=torch.float64,
                                      device=self.device)
        self.log_new_frame = lin_log(self.new_frame) if not self.log_input else self.new_frame

        inten01 = None
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:
            inten01 = rescale_intensity_frame(self.new_frame.clone().detach())

        if self.base_log_frame is None:
            self.lp_log_frame = self.log_new_frame
            self.photoreceptor_noise_arr = torch.zeros_like(self.lp_log_frame)

        self.lp_log_frame = low_pass_filter(
            log_new_frame=self.log_new_frame,
            lp_log_frame=self.lp_log_frame,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cutoff_hz)

        if self.photoreceptor_noise and not self.base_log_frame is None:
            self.photoreceptor_noise_vrms = compute_photoreceptor_noise_voltage(
                shot_noise_rate_hz=self.shot_noise_rate_hz, f3db=self.cutoff_hz, sample_rate_hz=1 / delta_time,
                pos_thr=self.pos_thres_nominal, neg_thr=self.neg_thres_nominal, sigma_thr=self.sigma_thres)
            noise = self.photoreceptor_noise_vrms * torch.randn(self.log_new_frame.shape, dtype=torch.float32,
                                                                device=self.device)
            self.photoreceptor_noise_arr = low_pass_filter(noise, self.photoreceptor_noise_arr, None, delta_time,
                                                           self.cutoff_hz)
            self.photoreceptor_noise_samples.append(
                self.photoreceptor_noise_arr[0, 0].cpu().item())

        if self.csdvs_enabled:
            self._update_csdvs(delta_time)

        if self.base_log_frame is None:
            self._init(new_frame)
            if not self.csdvs_enabled:
                self.base_log_frame = self.lp_log_frame
            else:
                self.base_log_frame = self.lp_log_frame - self.cs_surround_frame
            return None  # on first input frame we just setup the state of all internal nodes of pixels

        if self.scidvs:
            if self.scidvs_highpass is None:
                self.scidvs_highpass = torch.zeros_like(self.lp_log_frame)
                self.scidvs_previous_photo = torch.clone(self.lp_log_frame).detach()
            self.scidvs_highpass += (self.lp_log_frame - self.scidvs_previous_photo) \
                                    - delta_time * self.scidvs_dvdt(self.scidvs_highpass,self.scidvs_tau_arr)
            self.scidvs_previous_photo = torch.clone(self.lp_log_frame)

        if self.leak_rate_hz > 0:
            self.base_log_frame = subtract_leak_current(
                base_log_frame=self.base_log_frame,
                leak_rate_hz=self.leak_rate_hz,
                delta_time=delta_time,
                pos_thres=self.pos_thres,
                leak_jitter_fraction=self.leak_jitter_fraction,
                noise_rate_array=self.noise_rate_array)

        photoreceptor = EventEmulator.SCIDVS_GAIN * self.scidvs_highpass if self.scidvs else self.lp_log_frame

        if not self.csdvs_enabled:
            self.diff_frame = photoreceptor + self.photoreceptor_noise_arr - self.base_log_frame
        else:
            self.c_minus_s_frame = photoreceptor + self.photoreceptor_noise_arr - self.cs_surround_frame
            self.diff_frame = self.c_minus_s_frame - self.base_log_frame

        pos_evts_frame, neg_evts_frame = compute_event_map(
            self.diff_frame, self.pos_thres, self.neg_thres)
        max_num_events_any_pixel = max(pos_evts_frame.max(),
                                       neg_evts_frame.max())
        max_num_events_any_pixel=max_num_events_any_pixel.cpu().numpy().item()
        if max_num_events_any_pixel > 100:
            logger.warning(f'Too many events generated for this frame: num_iter={max_num_events_any_pixel}>100 events')

        events = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        min_ts_steps=max_num_events_any_pixel if max_num_events_any_pixel>0 else 1
        ts_step = delta_time / min_ts_steps
        ts = torch.linspace(
            start=self.t_previous+ts_step,
            end=t_frame,
            steps=min_ts_steps, dtype=torch.float32, device=self.device)

        final_pos_evts_frame = torch.zeros(
            pos_evts_frame.shape, dtype=torch.int32, device=self.device)
        final_neg_evts_frame = torch.zeros(
            neg_evts_frame.shape, dtype=torch.int32, device=self.device)

        if max_num_events_any_pixel == 0 and self.no_events_warning_count<100:
            logger.warning(f'no signal events generated for frame #{self.frame_counter:,} at t={t_frame:.4f}s')
            self.no_events_warning_count+=1
        else:
            for i in range(max_num_events_any_pixel):
                pos_cord = (pos_evts_frame >= i + 1)
                neg_cord = (neg_evts_frame >= i + 1)

                if self.refractory_period_s > ts_step:
                    pos_time_since_last_spike = (
                            pos_cord * ts[i] - self.timestamp_mem)
                    neg_time_since_last_spike = (
                            neg_cord * ts[i] - self.timestamp_mem)

                    pos_cord = (
                            pos_time_since_last_spike > self.refractory_period_s)
                    neg_cord = (
                            neg_time_since_last_spike > self.refractory_period_s)

                    self.timestamp_mem = torch.where(
                        pos_cord, ts[i], self.timestamp_mem)
                    self.timestamp_mem = torch.where(
                        neg_cord, ts[i], self.timestamp_mem)

                final_pos_evts_frame += pos_cord
                final_neg_evts_frame += neg_cord

                pos_event_xy = pos_cord.nonzero(as_tuple=True)
                neg_event_xy = neg_cord.nonzero(as_tuple=True)

                events_curr_iter = self.get_event_list_from_coords(pos_event_xy, neg_event_xy, ts[i])

                if events_curr_iter is not None:
                    idx = torch.randperm(events_curr_iter.shape[0])
                    events_curr_iter = events_curr_iter[idx].view(events_curr_iter.size())
                    events=torch.cat((events,events_curr_iter))

        shot_on_cord, shot_off_cord = None, None

        if self.shot_noise_rate_hz > 0 and not self.photoreceptor_noise:
            shot_on_cord, shot_off_cord = generate_shot_noise(
                shot_noise_rate_hz=self.shot_noise_rate_hz,
                delta_time=delta_time,
                shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
                inten01=inten01,
                pos_thres_pre_prob=self.pos_thres_pre_prob,
                neg_thres_pre_prob=self.neg_thres_pre_prob)

            shot_on_xy = shot_on_cord.nonzero(as_tuple=True)
            shot_off_xy = shot_off_cord.nonzero(as_tuple=True)

            shot_noise_events = self.get_event_list_from_coords(shot_on_xy, shot_off_xy, ts[-1])

            if shot_noise_events is not None:
                events=torch.cat((events, shot_noise_events), dim=0)

        self.base_log_frame += final_pos_evts_frame * self.pos_thres
        self.base_log_frame -= final_neg_evts_frame * self.neg_thres

        if not self.photoreceptor_noise and self.shot_noise_rate_hz>0:
            self.base_log_frame[shot_on_xy]=self.lp_log_frame[shot_on_xy]
            self.base_log_frame[shot_off_xy]=self.lp_log_frame[shot_off_xy]

        if len(events) > 0:
            events = events.cpu().data.numpy()
            timestamps=events[:,0]
            import numpy as np
            if np.any(np.diff(timestamps)<0):
                idx=np.argwhere(np.diff(timestamps)<0)
                logger.warning(f'nonmonotonic timestamp(s) at indices {idx}')

        self.t_previous = t_frame
        if len(events) > 0:
            return events  # ndarray shape (N,4) where N is the number of events, rows are [t,x,y,p]
        else:
            return None

    def get_event_list_from_coords(self, pos_event_xy, neg_event_xy, ts):
        """ Gets event list from ON and OFF event coordinate lists.
        :param pos_event_xy: Tensor[2,n] where n is number of ON events, [0,n] are y addresses and [1,n] are x addresses
        :param neg_event_xy: Tensor[2,m] where m is number of ON events, [0,m] are y addresses and [1,m] are x addresses
        :param ts: the timestamp given to all events (scalar)
        :returns: Tensor[n+m,4] of AER [t, x, y, p]
        """
        num_pos_events = pos_event_xy[0].shape[0]
        num_neg_events = neg_event_xy[0].shape[0]
        num_events = num_pos_events + num_neg_events
        events_curr_iter=None
        if num_events > 0:
            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            events_curr_iter = torch.ones(
                (num_events, 4), dtype=torch.float32,
                device=self.device)
            events_curr_iter[:, 0] *= ts

            events_curr_iter[:num_pos_events, 1] = pos_event_xy[1]
            events_curr_iter[:num_pos_events, 2] = pos_event_xy[0]

            events_curr_iter[num_pos_events:, 1] = neg_event_xy[1]
            events_curr_iter[num_pos_events:, 2] = neg_event_xy[0]
            events_curr_iter[num_pos_events:, 3] = -1
        return events_curr_iter

    def _update_csdvs(self, delta_time):
        if self.cs_surround_frame is None:
            self.cs_surround_frame = self.lp_log_frame.clone().detach()
        else:
            abs_min_tau_p = 1e-9
            tau_p = abs_min_tau_p if (
                    self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) else self.cs_tau_p_ms * 1e-3
            tau_h = abs_min_tau_p / (self.cs_lambda_pixels ** 2) if (
                    self.cs_tau_h_ms is None or self.cs_tau_h_ms == 0) else self.cs_tau_h_ms * 1e-3
            min_tau = min(tau_p, tau_h)
            NUM_STEPS_PER_TAU = 5
            num_steps = int(math.ceil((delta_time / min_tau) * NUM_STEPS_PER_TAU))
            actual_delta_time = delta_time / num_steps
            if num_steps > 1000 and not self.cs_steps_warning_printed:
                if self.cs_tau_p_ms == 0:
                    logger.warning(
                        f'You set time constant cs_tau_p_ms to zero which set the minimum tau of {abs_min_tau_p}s')
                logger.warning(
                    f'CSDVS timestepping of diffuser could take up to {num_steps} '
                    f'steps per frame for Euler delta time {actual_delta_time:.3g}s; '
                    f'simulation of each frame will terminate when max change is smaller than {EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING}')
                self.cs_steps_warning_printed = True

            alpha_p = actual_delta_time / tau_p
            alpha_h = actual_delta_time / tau_h
            if alpha_p >= 1 or alpha_h >= 1:
                raise RuntimeError(
                    f'CSDVS update alpha (of IIR update) is too large; simulation would explode: '
                    f'alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}')
            if alpha_p > .25 or alpha_h > .25:
                logger.warning(
                    f'CSDVS update alpha (of IIR update) is too large; simulation will be inaccurate: '
                    f'alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}')
                self.cs_alpha_warning_printed = True
            p_ten = torch.unsqueeze(torch.unsqueeze(self.lp_log_frame, 0), 0)
            h_ten = torch.unsqueeze(torch.unsqueeze(self.cs_surround_frame, 0), 0)
            padding = torch.nn.ReplicationPad2d(1)
            max_change = 2 * EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING
            steps = 0
            while steps < num_steps and max_change > EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING:
                diff = p_ten - h_ten
                p_term = alpha_p * diff
                h_conv = torch.conv2d(padding(h_ten.float()), self.cs_k_hh.float())
                h_term = alpha_h * h_conv
                change_ten = p_term + h_term
                max_change = torch.max(
                    torch.abs(change_ten)).item()
                h_ten += change_ten
                steps += 1

            self.cs_steps_taken.append(steps)
            self.cs_surround_frame = torch.squeeze(h_ten)
