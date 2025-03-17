.. _vision:

Vision
======



Design principles
-----------------
Like `MIKE IO <https://github.com/DHI/mikeio>`_ and `ModelSkill <https://github.com/DHI/modelskill>`_, FMDAp should be: 

* Easy to use
* Easy to install
* Easy to get started
* Open Source​
* Easy to collaborate​
* Reproducible
* Easy access to new features


Functional objectives
===================== 

FMDAp should cover the following **user stories**: 


Is my DA scheme performing well? 
--------------------------------
As a user, I want to detect...

* model instabilities
* non-physical behavior
* DA over-fitting (big difference between forecast and analysis)
* DA under-fitting (little impact of assimilation)
* non-Gaussian behavior (e.g. biases)
* auto-correlation in innovation/increments
* observation offset/biases


Visualization
-------------
As a user, I want to plot...

* observations on map showing the model domain
* diagnostic positions on a map showing the model domain
* multiple timeseries together, interactively

As a user, I want to 

* make all the *static* plots for my report/paper
* make *interactive* plots for...
    - troubleshooting
    - DA calibration
    - DA assessment (exploratory; setup phase)



Input/settings (pre-processing)
-------------------------------
As a user, I want to... 

* create MEASUREMENTS section from xlsx file 
* create MODEL_ERROR section from xlsx file 
* check if measurements are inside domain, otherwise find nearest point inside
* Estimate AR(1) half life from observations or forcing data
* Estimate spatial correlation length scales 


Automatic reporting
-------------------
After doing a DA simulation, I would like to get a **report** with:

* summary of DA settings 
* map showing domain and observations
* timeline showing simulation and observation coverage
* summary skill table
* summary bar chart
* summary scatter plot
* For each observation: 
    - info/stats

