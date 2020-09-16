# WIP - Obvious Open Mining Format (OOMF)
The Open Mining Format (OMF) is an attempt at standardizing the way mining (and more general geoscience data) is serialized and managed.
Goals shared ont he [OMF read the docs page](https://omf.readthedocs.io/en/latest/):

>* The goal of Open Mining Format is to standardize data formats across the mining community and promote collaboration
>* The goal of the API library is to provide a well-documented, object-based interface for serializing OMF files

OMF is based around the [properties](https://github.com/seequent/properties) project developed by Seequent, and naturally Seequent is one of the biggest supporters of OMF. 

This is a fantastic idea. Sharing data, geologic models, variograms etc across commercial software platforms and open source tools can be challenging. The OMF library Python API is does not lack functionality but it is not the most approachable way to open, manipulate or save data when working in python. The goal of OOMF, is simply make it easier to work with this file format in a more obvious and pythonic way.