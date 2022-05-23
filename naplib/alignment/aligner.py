import os
from os.path import isfile, join, isdir, dirname
import sys
import unicodedata
import string

from ..utils import _parse_outstruct_args


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
    '''
    def __init__(self, output_dir, dictionary_file=None, tmp_dir=None):

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
            dictionary_file = join(self.filedir_, 'eng.dict')
        self.dictionary_file = dictionary_file

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

    def align_from_files(self, audio_dir, text_dir):
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

        print(f'Resampling audio and putting in {self.tmp_dir} directory...')

        resample_path = join(self.filedir_, 'resample.sh')

        # resample the audios to 16000 and put them in the tmp data folder
        os.system(f'{resample_path} -s 16000 -r {audio_dir} -w {self.tmp_dir}')

        print(f'Converting text files to ascii in {self.tmp_dir} directory...')

        for root, dirs, files in os.walk(text_dir, topdown=False):
            for name in files:
                if '.txt' in name:
                    self._convert_text_to_ascii(name, root)

        print('Performing alignment...')

        # perform alignment using ProsodyLab-Aligner
        sys.path.insert(1, self.filedir_)
        prosodylab_main_file = join(self.filedir_, 'prosodylab_aligner/__main__.py')
        eng_zip_file = join(self.filedir_, 'eng.zip')
        os.system(f'python3 {prosodylab_main_file} -a {self.tmp_dir} -d {self.dictionary_file} -r {eng_zip_file}')
        sys.path.remove(self.filedir_)

        print(f'Converting .TextGrid files to .phn and .wrd in {self.output_dir}')

        # Convert textgrid files to .phn and .wrd files in output_dir
        for root, dirs, files in os.walk(self.tmp_dir, topdown=False):
            for name in files:
                if '.TextGrid' in name:

                    # print(f'looking at {os.path.join(root, name)}')

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


        print('All done!')

    def align(self, outstruct=None, name='name', sound='sound',
              soundf='soundf', transcript='transcript',
              dataf='dataf'):
        '''
        Perform alignment across a set of paired audio-text files stored
        in fields of an OutStruct. This function will create a set of
        .TextGrid files, as well as corresponding .phn and .wrd
        files in the output_dir which describe the timing of phonemes and
        words within each audio. These files can be used in conjunction
        with the other functions in `naplib.alignment`, such as
        ``get_phoneme_label_vector`` and ``get_word_label_vector``,
        which take these files as input. This function will automatically
        use ``naplib.alignment.get_phoneme_label_vector`` and
        ``naplib.alignment.get_word_label_vector`` to produce phoneme and
        word label vectors for each stimulus which can be placed into the
        OutStruct and further analyzed.

        Parameters
        ----------
        outstruct : naplib.OutStruct
            OutStruct containing the data to align. It must contain the
            following fields. 
        name : string or list of strings, default='name'
            If a string, specifies a field of the outstruct which contains
            the name for each trial. Otherwise, a list of strings specifies
            the name for each trial.
        sound : string or list of np.ndarrays, default='sound'
            If a string, specifies a field of the outstruct which contains
            the sound waveform for each trial. Otherwise, a list of np.ndarrays
            specifies the waveform for each trial.
        soundf : string, integer, or list of integers
            If a string, specifies a field of the outstruct which contains
            the sampling rate for each trial. Otherwise, a list of integers
            specifies the sampling rate for each trial, or a single integer gives the
            sampling rate for all trials.
        transcript : string or list of strings
            If a string, specifies a field of the outstruct which contains
            the transcript text for each trial. Otherwise, a list of strings
            specifies the transcript text for each trial.
        dataf : string, integer, or list of integers
            If a string, specifies a field of the outstruct which contains
            the desired sampling rate of the output. Otherwise, a list of integers
            specifies the Desired sampling rate of the output for each trial, or
            a single integer gives the desired sampling rate of the output
            for all trials.
        length : string or list of integers
            If a string, specifies a field of the outstruct which contains
            the desired output length (in samples) for each trial. Otherwise,
            a list of integers specifies the desired output length (in samples)
            for each trial.

        Returns
        -------
        phoneme_labels : list of np.ndarrays
            Phoneme label vector for each trial. phoneme_labels[i] is a np.ndarray
            of shape (time,) and sampling rate 

        Note
        ----
        After running this function, several files will be created and put into
        the output_dir. If the `name` field of each trial is "trial1", "trial2",
        etc, then the output_dir will look like this.

        | working directory
        | └── output_dir
        | │   └── trial1.phn
        | │   └── trial1.wrd
        | │   └── trial1.TextGrid
        | │   └── trial2.phn
        | │   └── trial2.wrd
        | │   └── trial2.TextGrid
        '''
        import textgrid

        print(f'Resampling audio and putting in {self.tmp_dir} directory...')

        resample_path = join(self.filedir_, 'resample.sh')

        # resample the audios to 16000 and put them in the tmp data folder
        os.system(f'{resample_path} -s 16000 -r {audio_dir} -w {self.tmp_dir}')

        print(f'Converting text files to ascii in {self.tmp_dir} directory...')

        for root, dirs, files in os.walk(text_dir, topdown=False):
            for name in files:
                if '.txt' in name:
                    self._convert_text_to_ascii(name, root)

        print('Performing alignment...')

        # perform alignment using ProsodyLab-Aligner
        sys.path.insert(1, self.filedir_)
        prosodylab_main_file = join(self.filedir_, 'prosodylab_aligner/__main__.py')
        eng_zip_file = join(self.filedir_, 'eng.zip')
        os.system(f'python3 {prosodylab_main_file} -a {self.tmp_dir} -d {self.dictionary_file} -r {eng_zip_file}')
        sys.path.remove(self.filedir_)

        print(f'Converting .TextGrid files to .phn and .wrd in {output_dir}')

        os.makedirs(output_dir, exist_ok=True)

        # Convert textgrid files to .phn and .wrd files in output_dir
        for root, dirs, files in os.walk(self.tmp_dir, topdown=False):
            for name in files:
                if '.TextGrid' in name:

                    # print(f'looking at {os.path.join(root, name)}')

                    # copy TextGrid file to output_dir so they are saved
                    os.system(f'cp {join(root, name)} {join(output_dir, name)}')

                    new_phn_name = name.replace('.TextGrid', '.phn')
                    new_wrd_name = name.replace('.TextGrid', '.wrd')

                    tg = textgrid.TextGrid.fromFile(join(root, name))
                    phones = tg[0]
                    words = tg[1]

                    # write phn file

                    phn_file = open(os.path.join(output_dir, new_phn_name), 'w')

                    for phone_seg in phones:
                        if phone_seg.mark == "":
                            phone_seg.mark = "sp"
                        if phone_seg.mark != "sil":
                            print(f"{phone_seg.minTime} {phone_seg.maxTime} {phone_seg.mark}", file=phn_file)

                    phn_file.close()

                    # write wrd file

                    wrd_file = open(os.path.join(output_dir, new_wrd_name), 'w')

                    for word_seg in words:
                        if word_seg.mark != "sil":
                            print(f"{word_seg.minTime} {word_seg.maxTime} {word_seg.mark}", file=wrd_file)

                    wrd_file.close()


        print('All done!')
