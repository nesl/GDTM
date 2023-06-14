
import math
import numpy as np
import os
import time
import sys

_meta_ = {
    'mss': 'MMW Demo',
    'dev': ('xWR14xx',),
    'ver': ('01.02.00.05', '02.01.00.04',),
    'cli': 'mmwDemo:/>',
    'seq': b'\x02\x01\x04\x03\x06\x05\x08\x07',
    'blk': 32,
    'aux': 921600,
    'ant': (4, 3),
    'app': {
        'rangeProfile':        ('plot_range_profile', 'capture_range_profile', 'monitor_activity', ),
        'noiseProfile':        ('plot_range_profile', ), 
        'detectedObjects':     ('plot_detected_objects', 'simple_cfar_clustering', ),
        'rangeAzimuthHeatMap': ('plot_range_azimuth_heat_map', ),
        'rangeDopplerHeatMap': ('plot_range_doppler_heat_map', )
    }
}

def get_conf(cfg):

    c = dict(cfg)
    p = {'loglin': float('nan'), 'fftcomp': float('nan'), 'rangebias': float('nan')}
    
    if '_comment_' in c:
        c.pop('_comment_', None)  # remove entry        
    
    if '_apps_' in c:
        _meta_['app'] = c['_apps_']
        c.pop('_apps_', None)  # remove entry

    if '_settings_' in c:
        
        rx_ant = int(c['_settings_']['rxAntennas'])
        tx_ant = int(c['_settings_']['txAntennas'])
        
        # common
        if c['channelCfg']['rxMask'] is None:
            c['channelCfg']['rxMask'] = 2**rx_ant - 1

        if c['channelCfg']['txMask'] is None:
            n = tx_ant
            if n == 1: n = 0
            else: n = 2 * n
            c['channelCfg']['txMask'] = 1 + n

        if c['channelCfg']['cascading'] is None:
            c['channelCfg']['cascading'] = 0  # always 0

        # range bias for post-processing
        if 'rangeBias' not in c['_settings_'] or c['_settings_']['rangeBias'] is None:
            c['_settings_']['rangeBias'] = 0
        
        # range bias for pre-processing
        if 'compRangeBiasAndRxChanPhase' in c:
            
            if c['compRangeBiasAndRxChanPhase']['rangeBias'] is None:
                c['compRangeBiasAndRxChanPhase']['rangeBias'] = c['_settings_']['rangeBias']
            
            if c['compRangeBiasAndRxChanPhase']['phaseBias'] is None or \
                type(c['compRangeBiasAndRxChanPhase']['phaseBias']) == list and \
                 len(c['compRangeBiasAndRxChanPhase']['phaseBias']) == 0:
                 c['compRangeBiasAndRxChanPhase']['phaseBias'] = [1, 0] * _meta_['ant'][0] * _meta_['ant'][1]

        # cli output
        if 'verbose' in c['_settings_'] and c['_settings_']['verbose'] is not None:
            verbose = c['_settings_']['verbose']
                       
        if c['dfeDataOutputMode']['type'] is None:
            c['dfeDataOutputMode']['type'] = 1  # legacy (no subframes)

        if c['adcCfg']['adcBits'] is None:
            c['adcCfg']['adcBits'] = 2  # 16 bit

        log_lin_scale = 1.0 / 512
        if num_tx_elev_antenna(c) == 1: log_lin_scale = log_lin_scale * 4.0 / 3  # MMWSDK-439

        fft_scale_comp_1d = fft_doppler_scale_compensation(32, num_range_bin(c))
        fft_scale_comp_2d = 1;                
        fft_scale_comp = fft_scale_comp_2d * fft_scale_comp_1d                
        
        p['log_lin'], p['fft_comp'], p['range_bias'] = log_lin_scale, fft_scale_comp, c['_settings_']['rangeBias']        
        
        c.pop('_settings_', None)  # remove entry
                
    return c, p

def show_config(cfg):  # simple print of resultant configuration
    
    info = 'Start frequency (GHz):    \t{}\n' + \
           'Slope (MHz/us):           \t{}\n' + \
           'Sampling rate (MS/s):     \t{:.2f}\n' + \
           'Sweep bandwidth (GHz):    \t{:.2f}\n' + \
           'Frame periodicity (ms):   \t{}\n' + \
           '\n' + \
           'Loops per frame:          \t{}\n' + \
           'Chirps per loop:          \t{}\n' + \
           'Samples per chirp:        \t{}\n' + \
           'Chirps per frame:         \t{}\n' + \
           'Samples per frame:        \t{}\n' + \
           'Receive antennas:         \t{}\n' + \
           '\n' + \
           'Azimuth antennas:         \t{}\n' + \
           'Elevation antennas:       \t{}\n' + \
           'Virtual antennas:         \t{}\n' + \
           'Azimuth resolution (°):   \t{:.1f}\n' + \
           '\n' + \
           'Range resolution (m):     \t{:.4f}\n' + \
           'Range bin (m):            \t{:.4f}\n' + \
           'Range depth (m):          \t{:.4f}\n' + \
           'Unambiguous range (m):    \t{:.4f}\n' + \
           'Range bins:               \t{}\n' + \
           '\n' + \
           'Doppler resolution (m/s): \t{:.4f}\n' + \
           'Maximum Doppler (m/s):    \t{:.4f}\n' + \
           'Doppler bins:             \t{}\n' + \
           ''

    info = info.format(
        
        cfg['profileCfg']['startFreq'],
        cfg['profileCfg']['freqSlope'],
        cfg['profileCfg']['sampleRate'] / 1000.0,
        bandwidth(cfg),        
        cfg['frameCfg']['periodicity'],

        cfg['frameCfg']['loops'],
        chirps_per_loop(cfg),
        samples_per_chirp(cfg),
        chirps_per_frame(cfg),
        samples_per_frame(cfg),
        num_rx_antenna(cfg),
        
        num_tx_azim_antenna(cfg),
        num_tx_elev_antenna(cfg),
        num_virtual_antenna(cfg),
        angular_resolution(cfg),
                
        range_resolution(cfg),
        range_bin(cfg),
        range_maximum(cfg),
        range_unambiguous(cfg),
        num_range_bin(cfg),
        
        doppler_resolution(cfg),
        doppler_maximum(cfg),
        num_doppler_bin(cfg),
        
    )

    print(file=sys.stderr, flush=True)
    print(info, file=sys.stderr, flush=True)


def hex2dec(value):
    """ 'ff' -> 255 ; 'af fe' -> (175, 254) ; ('af', 'fe) -> (175, 254) """
    if type(value) == str:
        value = value.strip()
        if ' ' not in value:
            return int(value, 16)
        else:
            return hex2dec(value.split(' '))
    else:
        return tuple(int(item, 16) for item in value)


def dec2hex(value, delim=''):
    """ 12648430 -> 'c0ffee' ; (255, 255) -> 'ffff' ; (256 * 256 - 1, 10) -> 'ffff0a' """
    if type(value) == int:
        s = hex(value)
        return '0' * (len(s) % 2) + s[2:]     
    else:       
        return delim.join(dec2hex(item, delim) for item in value) 


def dec2bit(value, bits=8):
    """ bits=8: 42 -> (False, True, False, True, False, True, False, False) """
    v = value % 2**bits
    seq = tuple(True if c == '1' else False for c in bin(v)[2:].zfill(bits)[::-1])
    if value - v > 0: seq = seq + dec2bit(value // 2**bits)
    return seq
 
 
def intify(value, base=16, size=2):
    if type(value) not in (list, tuple, bytes,):
        value = (value,)
    if (type(value) in (bytes,) and base == 16) or (type(value) in (list, tuple,)):
        return sum([item*((base**size)**i) for i, item in enumerate(value)])
    else:
        return sum([((item // 16)*base+(item % 16))*((base**size)**i) for i, item in enumerate(value)])


def split(value, size=2):
    return tuple(value[0 + i:size + i] for i in range(0, len(value), size))



def twos(value, bits):
    m = 2**(bits - 1)
    if value > m:
        value = value - 2*m
    return value


def pow2_ceil(x):
    if (x < 0): return 0
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


def q_to_dec(value, n):
    return value / (1 << n)


def dec_to_q(value, n):
    return int(value * (1 << n))


def q_to_db(value):
    return q_to_dec(value, 9) * 6

# ------------------------------------------------

def fft_range_scale_compensation(fft_min_size, fft_size):
    smin = (2.0**(math.ceil(math.log2(fft_min_size) / math.log2(4)-1))) / fft_min_size
    slin = (2.0**(math.ceil(math.log2(fft_size) / math.log2(4)-1))) / fft_size
    slin = slin / smin
    return slin


def fft_doppler_scale_compensation(fft_min_size, fft_size):
    slin = 1.0 * fft_min_size / fft_size
    return slin

# ------------------------------------------------

def num_tx_antenna(cfg, mask=(True, True, True)):
    b = dec2bit(cfg['channelCfg']['txMask'], 3)
    m = (True,) * (len(b) - len(mask)) + mask
    res = [digit if valid else 0 for digit, valid in zip(b, m)]
    return sum(res)


def num_tx_azim_antenna(cfg, mask=(True, False, True)):
    return num_tx_antenna(cfg, mask)


def num_tx_elev_antenna(cfg, mask=(False, True, False)):
    return num_tx_antenna(cfg, mask)


def num_rx_antenna(cfg):
    return sum(dec2bit(cfg['channelCfg']['rxMask'], 3))


def num_virtual_antenna(cfg):
    return num_tx_antenna(cfg) * num_rx_antenna(cfg)


def num_range_bin(cfg):
    return int(pow2_ceil(cfg['profileCfg']['adcSamples']))


def num_doppler_bin(cfg):
    return int(chirps_per_frame(cfg) / num_tx_antenna(cfg))


def num_angular_bin(cfg):
    return 64


def chirps_per_loop(cfg):
    if cfg['dfeDataOutputMode']['type'] == 1:
        return (cfg['frameCfg']['endIndex'] - cfg['frameCfg']['startIndex'] + 1)
    raise NotImplementedError('dfeDataOutputMode != 1')


def chirps_per_frame(cfg):
    return chirps_per_loop(cfg) * cfg['frameCfg']['loops']


def bandwidth(cfg):
    return 1.0 * cfg['profileCfg']['freqSlope'] * cfg['profileCfg']['adcSamples'] / cfg['profileCfg']['sampleRate']


def range_resolution(cfg):
    return range_maximum(cfg) / cfg['profileCfg']['adcSamples']


def range_bin(cfg):
    return range_maximum(cfg, 1.0) / num_range_bin(cfg)


def doppler_resolution(cfg):
    return 3e8 / (2 * cfg['profileCfg']['startFreq'] * 1e9 * (cfg['profileCfg']['idleTime'] + cfg['profileCfg']['rampEndTime']) * 1e-6 * chirps_per_frame(cfg))


def angular_resolution(cfg):
    n = num_rx_antenna(cfg) * num_tx_azim_antenna(cfg)
    if n == 1: return float('nan')
    return math.degrees(math.asin(2 / (num_rx_antenna(cfg) * num_tx_azim_antenna(cfg))))


def range_unambiguous(cfg):
    return range_maximum(cfg, 0.8)


def range_maximum(cfg, correction=1.0):
    return correction * 300 * cfg['profileCfg']['sampleRate'] / (2 * cfg['profileCfg']['freqSlope'] * 1e3)


def doppler_maximum(cfg):
    return doppler_resolution(cfg) * num_doppler_bin(cfg) / 2


def adc_sample_swap(cfg):
    return cfg['adcbufCfg']['sampleSwap']


def samples_per_chirp(cfg):
    return cfg['profileCfg']['adcSamples']


def samples_per_frame(cfg):
    return samples_per_chirp(cfg) * chirps_per_frame(cfg)