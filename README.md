# Obvious Open Mining Format (OOMF)

The Open Mining Format (OMF) is an attempt at standardizing the way mining (and more general geoscience data) is serialized and managed.
Goals shared ont he [OMF read the docs page](https://omf.readthedocs.io/en/latest/):

> - The goal of Open Mining Format is to standardize data formats across the mining community and promote collaboration
> - The goal of the API library is to provide a well-documented, object-based interface for serializing OMF files

OMF is based around the [properties](https://github.com/seequent/properties) project developed by Seequent, and naturally Seequent is one of the biggest supporters of OMF.

This is a fantastic idea. Sharing data, geologic models, variograms etc across commercial software platforms and open source tools can be challenging. The OMF library Python API does not lack functionality but it is simply diffcult to use. OOMF is an attempt to provide a wrapper to the OMF API and making this format a bit more approachable.
