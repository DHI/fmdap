.. _diagnostic_output:

Diagnostic Output
=================


Main Properties
---------------

The data in a diagnostics file can be categorized as 

0. *non*-update step
1. update-step: *before* update 
2. update-step: *after* update (analysis)

The Diagnostic object therefore has the following *result* properties:

* **forecast** (0 and 1)
* **forecast_at_update** (1)
* **analysis** (2)
* **result** (0 and 2)

and the additional data properties:

* **innovation** (y-Hx measurement-minus-model at 0, 1 and 2)
* **increment** (2-1; the update itself)

All these properties are also diagnostic objects with 
similar functionality as the diagnostic object: min(), max(), 
std(), plot(), hist(), ... 


Main Properties for files without updates
-----------------------------------------

If the diagnostic files stem from a **non-DA** simulation 
it will only contain *non*-update steps (0). The object will therefore 
only contain:

* **forecast** 
* **result** 
* **innovation** (y-Hx measurement-minus-model)

And `forecast` = `result`. 



Diagnostic Collection
---------------------

It's often convenient to load all diagnostic files from a simulation 
using FMDAp's DiagnosticCollection. It can done with the  




API
---
.. automodule:: fmdap.diagnostic_output
	:members:
	:inherited-members:

.. automodule:: fmdap.diagnostic_collection
	:members:
	:inherited-members: