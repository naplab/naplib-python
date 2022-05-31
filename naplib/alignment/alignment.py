import os
from os.path import isfile, join
import re
import numpy as np

def get_phoneme_label_vector(phn_file, length, fs, befaft, mode='phonemes', return_label_lists=False):
    '''
    Creates a time-series vector of phoneme labels based on the output of the Penn-Phonetic Alignment
    procedure.
    
    Returns label vector as numpy array of shape (time, ) containing categorical labels.
    Assumes that the .phn files were made on wav files that did not have a befaft zero period.

    This function can be used instead of the ``get_label_vecs_from_files`` method
    inside the ``Aligner`` class if you only want a single file's label vector.

    Parameters
    ----------
    phn_file : string, path to .phn file
    length : int, length of stimulus (in samples) including befaft periods
    fs : int, sampling rate
    befaft : array or list of length 2, containing befaft time periods in seconds
    mode : string, either 'phonemes', or 'manner'. Any empty or 'sp' periods are labeled -1.
        If 'manner': labels correspond to ['plosive','fricative','nasal','sonorant']
    return_label_lists : bool, if True, returns a 2-tuple containing the labels as well as the list
        of possible labels, so that the index of a label in the list is the integer label assigned to it.
        
    Returns
    -------
    labels : np.array, shape (time,)
        Integer labels over time. Full length is befaft[0]*fs+length+befaft[1]*fs
        
    item_list : list
        List of phonemes (if mode=='phonemes') or manners of articulation (if mode=='manner')
        which were used. The index of a phoneme indicates the label it is assigned.
    '''

    pattern = r'[0-9]'
    phoneme_list = ['EH','K','S','L','AH','M','EY','SH','N','P','OY','T','OW','Z','W','D','B','V','IH','AA','R','AY','ER','AE','AO','NG','G','TH','IY','F','DH','HH','UH','CH','UW','AW','JH','Y','ZH']
    manner_list = ['plosive','fricative','nasal','sonorant']
    manner_dict = {'EH':'sonorant',
                    'K':'plosive',
                    'S':'fricative',
                    'L':'sonorant',
                    'AH':'sonorant',
                    'M':'nasal',
                    'EY':'sonorant',
                    'SH':'fricative',
                    'N':'nasal',
                    'P':'plosive',
                    'OY':'sonorant',
                    'T':'plosive',
                    'OW':'sonorant',
                    'Z':'fricative',
                    'W':'sonorant',
                    'D':'plosive',
                    'B':'plosive',
                    'V':'fricative',
                    'IH':'sonorant',
                    'AA':'sonorant',
                    'R':'sonorant',
                    'AY':'sonorant',
                    'ER':'sonorant',
                    'AE':'sonorant',
                    'AO':'sonorant',
                    'NG':'nasal',
                    'G':'plosive',
                    'TH':'fricative',
                    'IY':'sonorant',
                    'F':'fricative',
                    'DH':'fricative',
                    'HH':'fricative',
                    'UH':'sonorant',
                    'CH':'velar',
                    'UW':'sonorant',
                    'AW':'sonorant',
                    'JH':'velar',
                    'Y':'sonorant',
                    'ZH':'sonorant',
                    'sp':'none'}
    
    f = open(phn_file, 'r')
    Lines = [line.strip() for line in f.readlines()]
    f.close()

    labels = -1 * np.ones((length,))

    for line in Lines:
        phn = re.sub(pattern, '', line.split(' ')[-1])
        start_time, end_time = line.split(' ')[:2]
        start_time, end_time = float(start_time), float(end_time)
        start_time += befaft[0]
        end_time += befaft[0]
        if mode == 'phonemes':
            labels[round(start_time*fs):round(end_time*fs)] = phoneme_list.index(phn) if phn in phoneme_list else -1.0
        elif mode == 'manner':
            labels[round(start_time*fs):round(end_time*fs)] = manner_list.index(manner_dict[phn]) if manner_dict[phn] in manner_list else -1.0
        else:
            raise Exception(f"mode parameter must be either 'phonemes' or 'manner', but found {mode}")
        

    if return_label_lists:
        if mode == 'phonemes':
            return labels, phoneme_list
        elif mode == 'manner':
            return labels, manner_list
        else:
            raise Exception(f"mode parameter must be either 'phonemes' or 'manner', but found {mode}")
    return labels

def get_word_label_vector(wrd_file, length, fs, befaft, wrd_dict=None, wrd_files_dir=None, return_wrd_dict=False):
    '''
    Returns label vector as numpy array of shape (time, ) containing categorical labels.
    Assumes that the .wrd files were made on wav files that did not have a befaft zero period.

    This function can be used instead of the ``get_label_vecs_from_files`` method
    inside the ``Aligner`` class if you only want a single file's label vector.

    Parameters
    ----------
    wrd_file : string, path to .wrd file
    length : int
        length of stimulus (in samples) including befaft periods
    fs : int, sampling rate
    befaft : array or list of length 2
        containing befaft time periods in seconds
    wrd_dict : dict
        keys are words (capitalized) and values are integers which become the labels for each word
    wrd_files_dir : string
        Path to directory containing all the .wrd files to be included in the dictionary. This is
        ignored if wrd_dict is supplied, but otherwise, a new wrd_dict is created using the .wrd
        files in this directory. By default, this will make 'sp' (space) have a label of -1.
    return_wrd_dict : bool
        Whether or not to return the wrd_dict along with the labels as a tuple
        
    Returns
    -------
    labels : np.array, shape (time,)
        Integer labels over time. Full length is befaft[0]*fs+length+befaft[1]*fs
    wrd_dict : dict
        Word dictionary (word to integer) used to create the labels.
        Only returned if return_wrd_dict==True
    '''
    
    pattern = r'[0-9]'
    
    if wrd_dict is None and wrd_files_dir is None:
        raise Exception('Must provide either wrd_dict or wrd_files_dir')
    elif wrd_dict is None:
        wrd_dict = create_wrd_dict(wrd_files_dir)
    
    f = open(wrd_file, 'r')
    Lines = [line.strip() for line in f.readlines()]
    f.close()

    labels = -1 * np.ones((length,))

    for line in Lines:
        wrd = re.sub(pattern, '', line.split(' ')[-1])
        start_time, end_time = line.split(' ')[:2]
        start_time, end_time = float(start_time), float(end_time)
        start_time += befaft[0]
        end_time += befaft[0]
        try:
            labels[round(start_time*fs):round(end_time*fs)] = wrd_dict[wrd]
        
        except KeyError:
            raise KeyError(f"{wrd} is not a valid key in wrd_dict")

    if return_wrd_dict:
        return labels, wrd_dict

    return labels

def create_wrd_dict(wrd_files_dir, list_to_skip=[]):
    '''
    Create a new word to label dictionary, which can be passed to get_word_label_vector.
    
    Parameters
    ----------
    wrd_files_dir : string
        Path to directory containing all the .wrd files to be used
    list_to_skip : list of strings, default=[]
        Words which will not be added to the dictionary.
        
    Returns
    -------
    wrd_dict : dict
        Dictionary of word:int (key:value) pairs for all the words in the corpus
        of files in the directory.
    '''
    
    pattern = r'[0-9]'
    
    wrd_files = sorted([f for f in os.listdir(wrd_files_dir) if isfile(join(wrd_files_dir, f)) and f.endswith('.wrd')])
    
    wrd_dict = {}
    
    for wrd_file in wrd_files:
        f = open(join(wrd_files_dir, wrd_file), 'r')
        Lines = [line.strip() for line in f.readlines()]
        f.close()
    
        for line in Lines:
            wrd = re.sub(pattern, '', line.split(' ')[-1])
            if wrd == 'sp':
                wrd_dict[wrd] = -1
            elif wrd not in wrd_dict and wrd not in list_to_skip:
                wrd_dict[wrd] = len(wrd_dict)

    return wrd_dict
    