Install
=======

.. _pipAnchor:

pip installation instructions
-----------------------------

Below we assume you have the default Python3 environment already configured on
your computer and you intend to install ``naplib-python`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip3`` (the Python3 package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip3`` first.

Install the current release of ``naplib`` with ``pip3``::

    $ pip3 install naplib

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip3 install --upgrade naplib

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip3 install --user naplib

Alternatively, you can manually download ``naplib-python`` from
`GitHub <https://github.com/naplab/naplib-python>`_  or
`PyPI <https://pypi.org/project/naplib/>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip3 install -e .

This will install ``naplib`` and the required dependencies.

.. _dependencyAnchor:

Required Dependencies
---------------------

``naplib-python`` requires the following packages:

- matplotlib>=3.1.0
- numpy>=1.15.0
- scipy>=1.5.0
- pandas>=1.0.0
- statsmodels>=0.13.0
- hdf5storage>=0.1.1
- scikit-learn
- mne

