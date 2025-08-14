# nxcals-methods

NXCALS data extraction, analysis, and visualization API

Liam O'Shaughnessy (Princeton '26, physics) under the supervision of Amaury Beeckman - CERN Beams Department Operations Group, Proton Synchrotron - 2025 CERN Summer Student Program.

In this project, my objective was to furnish a number of methods that would allow for easy handling of NXCALS data.

For accelerators such as the Proton Synchrotron at CERN, beam data (e.g. emittance, bunch length, etc.) is accessible through NXCALS, a logging system based on Hadoop Big Data Technologies that uses a Spark cluster. To track beam performance, identify problems, and test proposed solutions, accessing this data is critical. Hence, it is of interest to have reliable methods for fetching and processing this data.

The methods contained in nxcalsExtractors.py fit more or less into three groups. First, extractNxcalsData and its helper methods retrieve any number of desired fields from a property, and return a Pandas DataFrame for a specific user of these fields over some time window. Second, getNxcalsStats and its helper methods return statistical data, such as time-averaged values, for any field for any user. Finally, the plotting methods allow for the visualization of raw scalar data, time-averaged array data, and field vs. trace for particular instants. With these tools, analysis of hours of beam data should be faster and more automated.
