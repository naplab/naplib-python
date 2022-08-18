# naplib-python
Tools and functions for neural acoustic data processing and analysis in python. The documentation can be acccessed at the link below. It contains the API reference as well as example notebooks.

- [**Documentation**](https://naplib-python.readthedocs.io/en/latest/index.html)
- [**Examples**](https://naplib-python.readthedocs.io/en/latest/examples/index.html)
- [**License**](https://naplib-python.readthedocs.io/en/latest/license.html)

## Installation

To install or update this package through pip, run the following command:

```bash
pip install git+https://github.com/naplab/naplib-python.git
```

## API

The basic data structure for storing neural recording data is the OutStruct, which contains neural recordings and other variables associated with different trials/stimuli. Examples of loading and using this data structure can be found in the documentation and the docs/examples/ folder.

### OutStruct Data Structure Schematic

<p align="center">
  <img width=650 src="docs/figures/naplib-python-outstruct-figure.png" />
</p>

## Contributions

naplib-python is built by the [Neural Acoustic Processing Lab](http://naplab.ee.columbia.edu/) at Columbia University. We primarily use it for processing neural data coming from electrocorticography (ECoG) and electroencephalography (EEG) along with paired audio stimuli in order to study the auditory cortex. You are free to use the software according to its license, and we welcome contributions if you would like to propose changes, additions, or fixes. See our [**contribution guide**](https://naplib-python.readthedocs.io/en/latest/contributing.html) for more details.
