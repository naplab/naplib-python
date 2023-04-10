import struct
import numpy as np

from naplib import logger


def read_htk(filename, return_codes=False):
    """
    Read an HTK file.

    Parameters
    ----------
    filename: str, pathlike
        Path to file or filename to read (should end in .htk)

    Returns
    -------
    data : np.ndarray
        Data array in the file (time * channels)
    fs : int
        Sampling rate (Hz)
    type_code : int
        Type code of the file. Only returned if `return_codes=True`. See `voicebox's readhtk<http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_readhtk.html>`_ for details.
    data_type : str
        Data type of the file. Only returned if `return_codes=True`.

    Examples
    --------
    >>> from naplib.io import read_htk
    >>> data, fs = read_htk('example.htk')
    >>> data.shape
    (200000, 1)
    >>> fs
    2400
    >>> data, fs, tc, dt = read_htk('example.htk', return_codes=True)
    >>> tc
    8971
    >>> dt
    'PLP_D_A_0'


    Notes
    -----
    Not implemented types:
        CRC checking - files can have CRC, but it won't be checked for correctness
        VQ - Vector features are not implemented.
    """

    data = None
    n_samples = 0
    n_features = 0
    samp_period = 0
    basic_kind = None
    qualifiers = None
    endian = '>'

    with open(filename, "rb") as f:

        header = f.read(12)
        n_samples, samp_period, samp_size, type_code = struct.unpack(">iihh", header)
        if n_samples<0 or samp_period<0 or samp_size<0:
            endian = '<'
            n_samples, samp_period, samp_size, type_code = struct.unpack(endian+"iihh", header)
        basic_parameter = type_code & 0x3F
        
        fs = int(samp_period * 1e-4)

        if basic_parameter == 0:
            basic_kind = "WAVEFORM"
        elif basic_parameter == 1:
            basic_kind = "LPC"
        elif basic_parameter == 2:
            basic_kind = "LPREFC"
        elif basic_parameter == 3:
            basic_kind = "LPCEPSTRA"
        elif basic_parameter == 4:
            basic_kind = "LPDELCEP"
        elif basic_parameter == 5:
            basic_kind = "IREFC"
        elif basic_parameter == 6:
            basic_kind = "MFCC"
        elif basic_parameter == 7:
            basic_kind = "FBANK"
        elif basic_parameter == 8:
            basic_kind = "MELSPEC"
        elif basic_parameter == 9:
            basic_kind = "USER"
        elif basic_parameter == 10:
            basic_kind = "DISCRETE"
        elif basic_parameter == 11:
            basic_kind = "PLP"
        else:
            basic_kind = "ERROR"

        qualifiers = []
        if (type_code & 0o100) != 0:
            qualifiers.append("E")
        if (type_code & 0o200) != 0:
            qualifiers.append("N")
        if (type_code & 0o400) != 0:
            qualifiers.append("D")
        if (type_code & 0o1000) != 0:
            qualifiers.append("A")
        if (type_code & 0o2000) != 0:
            qualifiers.append("C")
        if (type_code & 0o4000) != 0:
            qualifiers.append("Z")
        if (type_code & 0o10000) != 0:
            qualifiers.append("K")
        if (type_code & 0o20000) != 0:
            qualifiers.append("0")
        if (type_code & 0o40000) != 0:
            qualifiers.append("V")
        if (type_code & 0o100000) != 0:
            qualifiers.append("T")

        if "C" in qualifiers or "V" in qualifiers or basic_kind == "IREFC" or basic_kind == "WAVEFORM":
            n_features = samp_size // 2
        else:
            n_features = samp_size // 4

        if "C" in qualifiers:
            n_samples -= 4

        if "V" in qualifiers:
            raise NotImplementedError("VQ is not implemented")

        data = []
        if basic_kind == "IREFC" or basic_kind == "WAVEFORM":
            for x in range(n_samples):
                s = f.read(samp_size)
                frame = []
                for v in range(n_features):
                    val = struct.unpack_from(endian+"h", s, v * 2)[0] / 32767.0
                    frame.append(val)
                data.append(frame)
        elif "C" in qualifiers:

            A = []
            s = f.read(n_features * 4)
            for x in range(n_features):
                A.append(struct.unpack_from(endian+"f", s, x * 4)[0])
            B = []
            s = f.read(n_features * 4)
            for x in range(n_features):
                B.append(struct.unpack_from(endian+"f", s, x * 4)[0])

            for x in range(n_samples):
                s = f.read(samp_size)
                frame = []
                for v in range(n_features):
                    frame.append((struct.unpack_from(endian+"h", s, v * 2)[0] + B[v]) / A[v])
                data.append(frame)
        else:
            for x in range(n_samples):
                s = f.read(samp_size)
                frame = []
                for v in range(n_features):
                    val = struct.unpack_from(endian+"f", s, v * 4)
                    frame.append(val[0])
                data.append(frame)
                
        if "K" in qualifiers:
            logger.warning("CRC checking not implememnted...")

        data = np.array(data)
            
        if return_codes:
            qualifiers = '_'.join(qualifiers)
            data_type = f'{basic_kind}_{qualifiers}'
            return data, fs, type_code, data_type
        else:
            return data, fs
