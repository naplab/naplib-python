Changelog
=========

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |MajorFeature| replace:: :raw-html:`<font color="green">[Major Feature]</font>`
.. |Feature| replace:: :raw-html:`<font color="green">[Feature]</font>`
.. |Efficiency| replace:: :raw-html:`<font color="blue">[Efficiency]</font>`
.. |Enhancement| replace:: :raw-html:`<font color="blue">[Enhancement]</font>`
.. |Fix| replace:: :raw-html:`<font color="red">[Fix]</font>`
.. |API| replace:: :raw-html:`<font color="DarkOrange">[API]</font>`

Change tags (adopted from `sklearn <https://scikit-learn.org/stable/whats_new/v0.23.html>`_):

- |MajorFeature| : something big that you couldn’t do before. 

- |Feature| : something that you couldn’t do before.

- |Efficiency| : an existing feature now may not require as much computation or memory.

- |Enhancement| : a miscellaneous minor improvement.

- |Fix| : something that previously didn’t work as documentated – or according to reasonable expectations – should now work.

- |API| : you will need to change your code to have the same effect in the future; or a feature will be removed in the future.

Version 1.3.0
-------------
- |Feature| : Added the function ``peak_rate`` to ``naplib.features`` which can be used to extract peak rate events from an acoustic stimulus. See documentation for details.
- |Feature| : Added the function ``resample_categorical`` to ``naplib.array_ops`` which can be used to resample categorical data.
- |Feature| : Added the function ``forward_fill`` to ``naplib.array_ops`` for fast forward-filling of nan values in an array.
- |Enhancement| : Added the parameter ``average`` to ``naplib.stats.responsive_ttest`` to enable more robust t-test statistics when there are enough trials. See documentation for details.

Version 1.2.0
-------------
- |Enhancement| : Added the parameter ``makedirs`` to ``naplib.io.save`` to enable the automatic creation of directories in the path for the filename provided.
- |Feature| : Added the function ``shift_label_onsets`` to ``naplib.segmentation`` which can be used to shift label vectors such as to segment data by word centers when only word onsets are aligned.

Version 1.1.0
-------------
- |API| |FIX| : Added a ``pre_post`` argument to ``stats.responsive_ttest`` to allow greater flexibility to how the responsiveness test is conducted. Also fixed a minor issue with how the test was being computed.
- |Enhancement| : Added the function ``visualization.eeg_locs`` to load the EEG channel locations which can be used with MNE ``plot_topomap`` to plot EEG channel data on a scalp.

Version 1.0.0
-------------
- |API| : Renamed ``stats.fratio`` to ``stats.discriminability`` and added Wilks' Lambda f-statistic as a method for computing disciminability.
- |API| : Changed all plotting function names to snake case. See API reference for details.
- |Efficiency| : Several large enhancements to the computation speed of several preprocessing steps.
- |Feature| : Added ``io.load_cnd`` function to load Continuous-event Neural Data (CND) format.

Version 0.3.0
-------------

- |Fix| : The ``auditory_spectrogram`` function was giving incorrect results when the input sampling rate was not a multiple of 2. For consistency with the Matlab wav2aud function's output, the function now resamples audio to 16k sampling rate before computing the auditory spectrogram, which ensures that the output is in the correct frequency range.
- |Fix| : Multiple miscellaneous fixes for the ``process_ieeg`` pipeline edge cases.

Version 0.2.0
-------------

- |Efficiency| : The major functionality of ``filterbank_hilbert`` has been significantly optimized and put into a new function called ``filter_hilbert`` which reduces memory usage by averaging the output over center frequencies before returning it to the user, utilizing up to 50x less memory.
- |API| : The API of filtering functions within the ``preprocessing`` module has changed to support inplace operations or other API changes that may change their output compared to v0.1.10 by changing default values of arguments. This includes ``filter_line_noise`` 
- |MajorFeature| : New module called ``naplab`` containing preprocessing pipelines and tools used by the Neural Acoustic Processing Lab (NAPLab) for processing raw neural data.


Version 0.1.10
--------------

- |Fix| : Changed the ``features.get_wrd_dict`` function to create a dictionary which does not use the value 0 for any words, which fixes an issue when performing word alignment where one word in the transcript might be assigned the value of 0, which would be masked by the 0's which indicate spacing.


Version 0.1.9
-------------

- |Feature| : Added the ``kdeplot`` function to ``naplib.visualization`` which plots kernel density and histograms jointly, and for multiple distributions at once.
- |Enhancement| : Expanded the functionality of ``naplib.visualization.shadederrorplot`` to allow computing the confidence interval using percentiles (such as 95% confidence interval), and to allow plotting the median or the mean at each time point.
- |API| : All visualization functions (except the default case of ``hierarchicalclusterplot`` given its multi-axis nature) now return the axes on which the data were plotted.


Version 0.1.8
-------------

- |Feature| : Added the ability to read an HTK file with ``naplib.io.read_htk``.
- |Enhancement| : Expanded the English phonetic dictionary file used by ``features.Aligner`` to include more words.
- |Fix| : Creating an empty ``naplib.Data`` object by initializing it with no arguments no longer raises an exception. This allows you to create an empty Data object and then build it up easily from a blank starting point.

Version 0.1.7
-------------

- |Fix| : Fix issue where ``stats.responsive_ttest`` to allow customization of the time periods to compare between before and after stimulus onset to test for stimulus responsiveness. Also fix a minor issue where p-values where not properly corrected for multiple tests.

Version 0.1.6
-------------

- |Fix| : Fix issue where ``stats.responsive_ttest`` was not comparing the correct values against each other to find responsive electrodes.

Version 0.1.5
-------------

- |MajorFeature| : Added a function for performing t-tests while controlling for categorical or continuous features, like subject identity, in the stats module as ``stats.ttest``.
- |Feature| : Added a function ``naplib.concat`` for concatenating Data objects over trials or over fields.

Version 0.1.4
-------------

- |Feature| : Added a sample dataset which can be loaded with ``io.load_speech_task_data``. This dataset contains simulated intracranial EEG recordings from a speech-listening task. The example notebooks on the documentation now utilize this dataset for all iEEG analysis.
- |API| : The ``import_outstruct`` function has been renamed ``import_data`` to better imply that a Data object is returned.
- |Efficiency| : The ``import_data`` function can now optionally use h5py under the hood, rather than hdf5storage, which makes loading large Data objects from MATLAB significantly faster.
- |Feature| : Added the ``export_data`` function which can be used to export Data objects as MATLAB-compatible (.mat) files, the same file structures which are read in by the ``import_data`` function.
- |Enhancement| : Added the ability to pass format strings (such as 'r--' to indicate red, dashed lines) to ``visualization.shadederrorplot`` so that the API matches that of matplotlib's Axes.plot.
- |Feature| : Added the ``filter_line_noise`` function which performs notch filtering with a linear-phase filter.
- |Feature| : Added the ``phase_amplitude_extract`` function uses the Hilbert Transform on a filterbank to extact phase and amplitude of broad frequency bands.
- |Feature| : Added the ``filterbank_hilbert`` function uses the Hilbert Transform on a filterbank to extact phase and amplitude of each filterbank output.


Version 0.1.2
-------------

- |Fix| : Fixed issue where data files required to properly use ``features.auditory_spectrogram`` and ``features.Aligner`` were not being included in the pip-installable package.
- |Fix| : Changed ``preprocessing.normalize`` function to properly allow ``axis=None`` to specify normalizing by global statistics, and updated the documentation accordingly.


Version 0.1.1
-------------

- |Feature| : Added Butterworth filtering to the ``preprocessing`` module as ``filter_butter``.
- |Feature| : Added frequency response plotting to the ``visualization`` module as ``freq_response``, which complements the Butterworth filter method by allowing a user to plot the filter used by that function.
- |API| : Changed the name of the OutStruct data structure to be called ``Data``, since this more accurately reflects what is stored in it, and OutStruct was a name created for internal use previously. This changes the API for all functions that previously took an OutStruct, since they now use the keyword argument ``data=data`` to input a Data object, and the field to be extracted is typically specified with ``field=field``.


Version 0.1.0
-------------

- |MajorFeature| : We’re happy to announce the first major version of ``naplib-python``. The package is pip-installable and contains a wide variety of methods for neural-acoustic data analysis.

