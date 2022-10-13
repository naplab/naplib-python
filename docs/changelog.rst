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

- |MajorFeature| : We’re happy to announce the first major stable version of ``naplib-python``. The package is pip-installable and contains a wide variety of methods for neural-acoustic data analysis.

