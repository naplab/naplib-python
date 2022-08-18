Features
========

.. currentmodule:: naplib.features

The following functions and classes are useful for extracting
features from acoustic signals, such as time-frequency representations
and linguistic features.

Auditory Spectrogram
--------------------

.. autofunction:: auditory_spectrogram

Aligner for Phonemes and Words
------------------------------

.. autoclass:: Aligner
	:members:

Phoneme Labels from Phonetic Alignment File
-------------------------------------------

.. autofunction:: get_phoneme_label_vector

Word Labels from Word Alignment File
------------------------------------

.. autofunction:: get_word_label_vector

Build Word Dictionary from Set of Files
---------------------------------------

.. autofunction:: create_wrd_dict