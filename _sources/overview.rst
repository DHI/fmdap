.. _overview:

Overview
========

FMDAp has the following modules: 

* `PFS <api.html#pfs>`_
* `Diagnostic outputs <api.html#module-fmdap.diagnostic_output>`_ 
* `Ensemble output <api.html#ensembleoutput>`_ 
* `AR1 <api.html#module-fmdap.AR1>`_
* `Spatial <api.html#module-fmdap.spatial>`_


Key features
------------

`PFS files <api.html#pfs>`_

* Get a dictionary with all DA settings 
* Get DataFrame of measurements
* Get DataFrame of model errors
* Get DataFrame of diagnostic outputs
* Check if points are inside domain, find nearest cell centers

See `PFS notebook <https://nbviewer.jupyter.org/github/DHI/fmdap/blob/main/notebooks/Pfs_file.ipynb>`_ for examples of use.

`Diagnostic outputs <api.html#module-fmdap.diagnostic_output>`_ 

* Get summary statistics
* Plot timeseries
* Plot histogram
* Plot ecdf
* Get analysis, forecast, result, increments and innovations

`Ensemble output <api.html#ensembleoutput>`_ 

* xx

`AR1 <api.html#module-fmdap.AR1>`_

* Estimate AR(1) half-life from data
* Simulate AR(1) process

See `Analyze forcing data notebook <https://nbviewer.jupyter.org/github/DHI/fmdap/blob/main/notebooks/Analyze_forcing_data.ipynb>`_ for examples of use.

`Spatial <api.html#module-fmdap.spatial>`_


.. image:: https://raw.githubusercontent.com/DHI/fmdap/main/images/spatial_correlation.png

* Get pair-wise distances and correlation coefficient from Dfsu file
* Fit Gaussian distance function to the above

See `Analyze forcing data notebook <https://nbviewer.jupyter.org/github/DHI/fmdap/blob/main/notebooks/Analyze_forcing_data.ipynb>`_ for examples of use.