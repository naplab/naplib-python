import os
from os.path import isfile, join, isdir, dirname
import sys
import unicodedata
import string
import shutil
import subprocess
import warnings

import numpy as np
from scipy.io.wavfile import write as write_wavfile
from scipy.io.wavfile import read as read_wavfile
from scipy.signal import resample as scipy_resample

from .alignment_extras import create_wrd_dict, get_phoneme_label_vector, get_word_label_vector
from ..utils import _parse_outstruct_args
from ..data import Data
from .prosodylab_aligner import run_aligner


class Aligner():
    '''
    This class performs phoneme and word alignment using audio files
    and matching text files containing scripts. If words in the texts do not
    appear in the dict file, you will need to add them to a dict file and specify
    it as ``dictionary_file``.

    Please see the :ref:`alignment example notebooks <alignment examples>` for more detailed
    tutorials which show how to align text and audio data and analyze the output.

    Note
    ----
    Several extra packages are required to perform alignment. Please follow the
    installation instructions for ``HTK`` and ``sox`` for your system before
    using alignment. Additionally, you will need to install
    `pyyaml <https://pypi.org/project/PyYAML/>`_ as well as
    `TextGrid <https://pypi.org/project/TextGrid/>`_ for the Aligner to work. These are
    not required dependencies of naplib-python, so they must be installed separately.
    
    Parameters
    ----------
    output_dir : string, path-like
            Directory to put output files in, such as .phn, .wrd., and .TextGrid files.
    dictionary_file : string, path-like, optional
        Path to a dictionary file (e.g. eng.dict) which contains phonemes for
        all words in corpus. If not provided, will use the default eng.dict.
        For an example file, see
        `ProsodyLab's eng.dict <https://github.com/prosodylab/Prosodylab-Aligner/blob/master/eng.dict>`_ 
    tmp_dir : string, path-like, optional
        Directory to hold temporary files that are created. If not provided, creates
        a folder called `data_/` in the current working directory and uses that.
    verbose : int, default=1
            Integer specifying the level of verbosity of print statements and messages
            while the function is running. 0 indicates no output, and higher values
            generate more output.
    '''
    def __init__(self, output_dir, dictionary_file=None, tmp_dir=None, verbose=1):

        os.makedirs(output_dir, exist_ok=True)
        if tmp_dir is None:
            tmp_dir = 'data_/'
            # check if this folder already exists, in which case throw an error
            # so we don't overwrite a folder by default.
            if isdir(tmp_dir):
                raise ValueError(f'No tmp_dir was provided, but could not use '
                    'the default "data_/" because a folder with that name '
                    'already exists in the current path. Please remove that '
                    'directory or explicitly specify the tmp_dir parameter.')
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.filedir_ = dirname(__file__)
        if dictionary_file is None:
            dictionary_file = join(self.filedir_, 'prosodylab_aligner', 'eng.dict')
        self.dictionary_file = dictionary_file
        self.verbose = verbose

        try:
            import yaml
        except Exception as e:
            raise Exception('Missing package pyyaml which is required for alignment. Please '
                'install it with "pip install pyyaml"')
        try:
            import textgrid
        except Exception as e:
            raise Exception('Missing package TextGrid which is required for alignment. Please '
                'install it with "pip install TextGrid"')


    def _remove_nonword_characters_and_punctuation_and_capitalize(self, s):
        exclude = set(string.punctuation)
        exclude.remove("'")
        s = ''.join(ch for ch in s if ch not in exclude)
        # s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.upper()
        return s

    def _convert_text_to_ascii(self, name, root):
        new_name = name.replace('.txt', '.lab')
        new_folder = self.tmp_dir

        unicode_file = open(os.path.join(root, name))
        unicode_data = unicode_file.read() #.decode(input_codec)
        unicode_data = self._remove_nonword_characters_and_punctuation_and_capitalize(unicode_data)
        ascii_data = unicodedata.normalize('NFKD', unicode_data).encode('ascii','ignore')
        ascii_file = open(os.path.join(new_folder, new_name), 'wb')
        ascii_file.write(ascii_data)

    def align(self, data=None, name='name', sound='sound',
              soundf='soundf', transcript='transcript',
              dataf='dataf', length='length'):
        '''
        Perform alignment across a set of paired audio-text files stored
        in fields of a Data object. This function will create a set of
        .TextGrid files, as well as corresponding .phn and .wrd
        files in the output_dir which describe the timing of phonemes and
        words within each audio. These files can be used in conjunction
        with the other functions in `naplib.alignment`, such as
        ``get_phoneme_label_vector`` and ``get_word_label_vector``,
        which take these files as input. This function will automatically
        use ``naplib.alignment.get_phoneme_label_vector`` and
        ``naplib.alignment.get_word_label_vector`` to produce phoneme and
        word label vectors for each stimulus which can be placed into the
        Data object and further analyzed.

        This function is essentially equivalent to storing audio and text
        in directories and using ``Aligner.align_files`` followed by
        ``Aligner.get_label_vecs_from_files``.

        Parameters
        ----------
        data : Data instance
            Data object containing the data to align. It must contain the
            following fields. 
        name : string or list of strings, default='name'
            If a string, specifies a field of the Data which contains
            the name for each trial. Otherwise, a list of strings specifies
            the name for each trial.
        sound : string or list of np.ndarrays, default='sound'
            If a string, specifies a field of the Data which contains
            the sound waveform for each trial. Otherwise, a list of np.ndarrays
            specifies the waveform for each trial.
        soundf : string, integer, or list of integers, default='soundf'
            If a string, specifies a field of the Data which contains
            the sampling rate for each trial. Otherwise, a list of integers
            specifies the sampling rate for each trial, or a single integer gives the
            sampling rate for all trials.
        transcript : string or list of strings, default='transcript'
            If a string, specifies a field of the Data which contains
            the transcript text for each trial. Otherwise, a list of strings
            specifies the transcript text for each trial.
        dataf : string, integer, or list of integers, default='dataf'
            If a string, specifies a field of the Data which contains
            the desired sampling rate of the output. Otherwise, a list of integers
            specifies the Desired sampling rate of the output for each trial, or
            a single integer gives the desired sampling rate of the output
            for all trials.
        length : string or list of integers, default='length'
            If a string, specifies a field of the Data which contains
            the desired output length (in samples) for each trial. Otherwise,
            a list of integers specifies the desired output length (in samples)
            for each trial.

        Returns
        -------
        alignment_data: Data instance
            Data object containing all alignment information, with all the fields
            described by the return values below. 
        phn_labels : list of np.ndarrays
            Phoneme label vector for each trial. alignment_data['phn_labels'][i]
            is a np.ndarray of shape (time,) and sampling rate dataf[i].
        manner_labels : list of np.ndarrays
            Manner of articulation label vector for each trial.
            alignment_data['manner_labels'][i]
            is a np.ndarray of shape (time,) and sampling rate dataf[i].
        wrd_labels : list of np.ndarrays
            Word label vector for each trial. alignment_data['wrd_labels'][i]
            is a np.ndarray of shape (time,) and sampling rate dataf[i].
        phn_label_list : list of lists of strings
            Phoneme label list returned by ``naplib.alignment.get_phoneme_label_vector``,
            so alignment_data['phn_label_list'][i] is a list of phonemes, where the
            index of a given phoneme in the list encodes that phoneme's label in ``phn_labels``.
        manner_label_list : list of lists of strings
            Manner of articulation label list returned by ``naplib.alignment.get_phoneme_label_vector``,
            so alignment_data['manner_label_list'][i] is a list of manners, where the
            index of a given manner in the list encodes that manner's label in ``manner_labels``.
        wrd_dict : dict
            Dictionary of word:int (key:value) pairs for all the words in the corpus
            of files in the directory, created by ``naplib.alignment.create_wrd_dict``
            So, alignment_data['wrd_dict'][i] is a dictionary
            which maps a word to its integer value as it is represented in ``wrd_labels``.

        Note
        ----
        This function will produce the following files in the output_dir to aid in
        its running.

        | working directory
        | └── output_dir
        | │   └── trial1.phn
        | │   └── trial1.wrd
        | │   └── trial1.TextGrid
        | │   └── trial2.phn
        | │   └── trial2.wrd
        | │   └── trial2.TextGrid
        '''

        names, sounds, soundf, transcripts, dataf, lengths = _parse_outstruct_args(data,
                                                                                            name,
                                                                                            sound,
                                                                                            soundf,
                                                                                            transcript,
                                                                                            dataf,
                                                                                            length, allow_strings_without_outstruct=False)

        # Write sounds to wav files in tmp folder and text to .txt files
        audio_dir = join(self.tmp_dir, 'tmp_sounds')
        os.makedirs(audio_dir, exist_ok=False)
        text_dir = join(self.tmp_dir, 'tmp_text')
        os.makedirs(text_dir, exist_ok=False)
        for name, soundwave, soundf_, script in zip(names, sounds, soundf, transcripts):
            fname_wav = join(audio_dir, f'{name}.wav')
            write_wavfile(fname_wav, int(soundf_), soundwave)
            fname_txt = join(text_dir, f'{name}.txt')
            with open(fname_txt, "w") as text_file:
                n = text_file.write(script)
                text_file.close()

        # Align text and audio from files
        self.align_files(audio_dir, text_dir, names=names)
        
        shutil.rmtree(audio_dir, ignore_errors=True)
        shutil.rmtree(text_dir, ignore_errors=True)

        # Get the label vectors from the alignment files
        return self.get_label_vecs_from_files(data=data, name=names,
                                  dataf=dataf, length=lengths,
                                  befaft=np.array([0, 0]))


    def align_files(self, audio_dir, text_dir, names=None):
        '''
        Perform alignment across a set of paired audio-text files stored
        in directories. This function will create a set of .TextGrid files,
        as well as corresponding .phn and .wrd
        files in the output_dir which describe the timing of phonemes and
        words within each audio. These files can be used in conjunction
        with the other functions in `naplib.alignment`, such as
        ``get_phoneme_label_vector`` and ``get_word_label_vector``,
        which take these files as input.

        Parameters
        ----------
        audio_dir : string, path-like
            Directory containing audio files (.wav).
        text_dir : string, path-like
            Directory containing text files (.txt) with matching names
            to the files in ``audio_dir``.
        names : list of strings, optional
            List of names (without file-type) which specify a subset of files within
            .the audio_dir and text_dir to process.

        Note
        ----
        The directory structure containing audios and matching text
        files must be correct in order to properly perform alignment.
        See below for what the directory layout should look like
        before running this function.

        | working directory
        | ├── audio_dir
        | │   ├── file1.wav
        | │   ├── file2.wav
        | └── text_dir
        | │   └── file1.txt
        | │   └── file2.txt 

        After running this function, the directory layout will look
        like this:

        | working directory
        | ├── audio_dir
        | │   ├── file1.wav
        | │   ├── file2.wav
        | └── text_dir
        | │   └── file1.txt
        | │   └── file2.txt
        | └── output_dir
        | │   └── file1.phn
        | │   └── file1.wrd
        | │   └── file1.TextGrid
        | │   └── file2.phn
        | │   └── file2.wrd
        | │   └── file2.TextGrid
        '''        
        import textgrid

        if names is not None and not isinstance(names, list):
            raise TypeError(f'names argument must be a list, or None, but got {type(names)}')

        if self.verbose >= 1:
            print(f'Resampling audio and putting in {self.tmp_dir} directory...')

        resample_path = join(self.filedir_, 'resample.sh')

        # resample the audios to 16000 and put them in the tmp data folder
        # if sox is installed use that, otherwise use scipy
        try:
            wavefilepath_ = join(self.filedir_, 'test.wav')
            subprocess.run(['sox', wavefilepath_, wavefilepath_], check=True, capture_output=True)
            os.system(f'{resample_path} -s 16000 -r {audio_dir} -w {self.tmp_dir}')
        except (OSError, subprocess.SubprocessError, subprocess.CalledProcessError):
            warnings.warn('Could not find sox. Using scipy to resample and save .wav files instead')
            # don't have sox, so use scipy instead
            wavfiles = [fname_ for fname_ in os.listdir(audio_dir) if fname_.endswith(".wav")]
            for wavfile_ in wavfiles:
                old_fs, wavdata = read_wavfile(join(audio_dir, wavfile_))
                if old_fs == 16000:
                    write_wavfile(join(self.tmp_dir, wavfile_), 16000, wavdata)
                else:
                    wavdata = scipy_resample(wavdata, int(16000. / old_fs))
                    write_wavfile(join(self.tmp_dir, wavfile_), 16000, wavdata)
            

        if self.verbose >= 1:
            print(f'Converting text files to ascii in {self.tmp_dir} directory...')

        for root, dirs, files in os.walk(text_dir, topdown=False):
            for name in files:
                if '.txt' in name:
                    self._convert_text_to_ascii(name, root)

        if self.verbose >= 1:
            print('Performing alignment...')

        # perform alignment using ProsodyLab-Aligner
        eng_zip_file = join(self.filedir_, 'prosodylab_aligner', 'eng.zip')
        run_aligner(align=self.tmp_dir, dictionary=[self.dictionary_file], read=eng_zip_file)

        if self.verbose >= 1:
            print(f'Converting .TextGrid files to .phn and .wrd in {self.output_dir}')

        # Convert textgrid files to .phn and .wrd files in output_dir
        for root, dirs, files in os.walk(self.tmp_dir, topdown=False):
            for name in files:
                if '.TextGrid' in name:

                    if names is not None and name.split('.TextGrid')[0] not in names:
                        continue

                    # copy TextGrid file to output_dir so they are saved
                    os.system(f'cp {join(root, name)} {join(self.output_dir, name)}')

                    new_phn_name = name.replace('.TextGrid', '.phn')
                    new_wrd_name = name.replace('.TextGrid', '.wrd')

                    tg = textgrid.TextGrid.fromFile(join(root, name))
                    phones = tg[0]
                    words = tg[1]

                    # write phn file

                    phn_file = open(os.path.join(self.output_dir, new_phn_name), 'w')

                    for phone_seg in phones:
                        if phone_seg.mark == "":
                            phone_seg.mark = "sp"
                        if phone_seg.mark != "sil":
                            print(f"{phone_seg.minTime} {phone_seg.maxTime} {phone_seg.mark}", file=phn_file)

                    phn_file.close()

                    # write wrd file

                    wrd_file = open(os.path.join(self.output_dir, new_wrd_name), 'w')

                    for word_seg in words:
                        if word_seg.mark != "sil":
                            print(f"{word_seg.minTime} {word_seg.maxTime} {word_seg.mark}", file=wrd_file)

                    wrd_file.close()


        if self.verbose >= 1:
            print('Finished creating alignment files.')

    def get_label_vecs_from_files(self, data=None, name='name',
                                  dataf='dataf', length='length',
                                  befaft='befaft'):
        '''

        Parameters
        ----------
        data : Data instance
            Data object containing the data to align. It must contain the
            following fields. 
        name : string or list of strings, default='name'
            If a string, specifies a field of the Data which contains
            the name for each trial. Otherwise, a list of strings specifies
            the name for each trial.
        dataf : string, integer, or list of integers, default='dataf'
            If a string, specifies a field of the Data which contains
            the desired sampling rate of the output. Otherwise, a list of integers
            specifies the Desired sampling rate of the output for each trial, or
            a single integer gives the desired sampling rate of the output
            for all trials.
        length : string or list of integers, default='length'
            If a string, specifies a field of the Data which contains
            the desired output length (in samples) for each trial. Otherwise,
            a list of integers specifies the desired output length (in samples)
            for each trial.
        befaft : string or list of np.ndarrays, or a single np.ndarray, default='befaft'
            If a string, specifies a field of the Data which contains
            the before and after time (in sec) for each trial. Otherwise,
            a list should contain the befaft period for each trial, and a single
            np.ndarray of length 2 specifies the befaft period for all trials. For
            example, befaft=np.array([0.5, 0.5]) indicates that for each trial, the
            wav file which was used to produce the alignment is 0.5 seconds shorter
            at the beginning and 0.5 seconds shorter at the end than the desired
            output length.

        Returns
        -------
        alignment_data: Data instance
            Data object containing all alignment information, with all the fields
            described by the return values below. 
        phn_labels : list of np.ndarrays
            Phoneme label vector for each trial. alignment_data['phn_labels'][i]
            is a np.ndarray of shape (time,) and sampling rate dataf[i].
        manner_labels : list of np.ndarrays
            Manner of articulation label vector for each trial.
            alignment_data['manner_labels'][i]
            is a np.ndarray of shape (time,) and sampling rate dataf[i].
        wrd_labels : list of np.ndarrays
            Word label vector for each trial. alignment_data['wrd_labels'][i]
            is a np.ndarray of shape (time,) and sampling rate dataf[i].
        phn_label_list : list of lists of strings
            Phoneme label list returned by ``naplib.alignment.get_phoneme_label_vector``,
            so alignment_data['phn_label_list'][i] is a list of phonemes, where the
            index of a given phoneme in the list encodes that phoneme's label in ``phn_labels``.
        manner_label_list : list of lists of strings
            Manner of articulation label list returned by ``naplib.alignment.get_phoneme_label_vector``,
            so alignment_data['manner_label_list'][i] is a list of manners, where the
            index of a given manner in the list encodes that manner's label in ``manner_labels``.
        wrd_dict : dict
            Dictionary of word:int (key:value) pairs for all the words in the corpus
            of files in the directory, created by ``naplib.alignment.create_wrd_dict``
            So, alignment_data['wrd_dict'][i] is a dictionary
            which maps a word to its integer value as it is represented in ``wrd_labels``.

        Note
        ----
        This function requires that the following files ALREADY exist in the aligner's
        output_dir.

        | working directory
        | └── output_dir
        | │   └── trial1.phn
        | │   └── trial1.wrd
        | │   └── trial1.TextGrid
        | │   └── trial2.phn
        | │   └── trial2.wrd
        | │   └── trial2.TextGrid
        '''
        names, dataf, lengths, befafts = _parse_outstruct_args(data, name, dataf, length, befaft, allow_strings_without_outstruct=False)

        for len_ in lengths:
            if not isinstance(len_, int):
                raise TypeError(f'Each length must be an integer but found {type(len_)}')

        wrd_dict = create_wrd_dict(self.output_dir)

        alignment_results = []

        if self.verbose >= 1:
            print(f'Creating label vectors for phonemes, manner of articulation, and words.')

        for n in range(len(names)):

            this_trial_result = {}
    
            # filenames for the .phn and .wrd files
            filename_phn = join(self.output_dir, f'{names[n]}.phn')
            filename_wrd = join(self.output_dir, f'{names[n]}.wrd')
            
            # desired length of the output label vector
            length = lengths[n]
            
            # sampling rate of our data
            fs = dataf[n]
            
            # before-after period for our data is 0 since we are using a Data object where the
            # durations of sound and output should already be matched
            befaft = befafts[n]
            
            
            # compute label vectors for phonemes, manner of articulation, and words, for this trial
            label_vec_phn, phn_label_list = get_phoneme_label_vector(filename_phn, length, fs, befaft, return_label_lists=True)
            label_vec_manner, manner_label_list = get_phoneme_label_vector(filename_phn, length, fs, befaft, mode='manner', return_label_lists=True)
            label_vec_wrd = get_word_label_vector(filename_wrd, length, fs, befaft, wrd_dict=wrd_dict)

            this_trial_result['phn_labels'] = label_vec_phn
            this_trial_result['manner_labels'] = label_vec_manner
            this_trial_result['wrd_labels'] = label_vec_wrd
            this_trial_result['phn_label_list'] = phn_label_list
            this_trial_result['manner_label_list'] = manner_label_list
            this_trial_result['wrd_dict'] = wrd_dict

            alignment_results.append(this_trial_result)
            
        # Add the computed label vectors to the Data
        return Data(alignment_results, strict=False)


