from easydict import EasyDict as edict

__C = edict()
# cfg is treated as a global variable.
# Need add "from config import cfg" for each file to use cfg.
cfg = __C

# parameters for MEF_SSIMc
__C.D1 = 0.01
__C.D2 = 0.03
__C.sigma_g = 0.2
__C.sigma_l = 0.2
__C.window_size = 8
__C.channels = 3
__C.adaptive_lum = True
__C.max_iter = 100
__C.cvg_t = 2e-6
__C.beta_inverse = 150
__C.plot_prg = 1
__C.write_qmap = 0
__C.write_progress = 0
__C.filename = None
__C.algorithm = None

# parameters for MS_SSIMc
__C.K = 3
__C.level = 3
__C.eps = 1e-9
__C.window_size_ms = 11
__C.beta_new = 10e6

# config
__C.use_cuda = True
__C.model = 'MEF_SSIMc'


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            # print(subkey)
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except ValueError:
            # handle the case when v is a string literal
            value = v
        assert isinstance(value, d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
