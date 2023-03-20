# MMSS

Python Implementation for the MMSS System: Storytelling Simulation for Decision-Making Support to Mitigate Misinformation on Social Media. With extended analytical capacity and novel representation of the diffusion model. For instance, Polarization and Societal Acceptance are now modeled along with Misinformation and Factual Information. The system is also now tested on a novel dataset with the novel representation, called PEGYPT dataset:
(https://github.com/Ahmed-Abouzeid/PEGYPT)
# Visualization

An example of COVID-19 social network mitigation story (Green nodes: true info, Red nodes: Misinformation) created by MMSS. The visualization is based on the Gephi (https://gephi.org/) software while the visualization temporal meta-data is generated by the paper proposed coloring graph algorithm..
<img src="social_network.gif" width="600" height="600"/>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running the code on default parameters on covid-19- 100 users
1- Unzip the data.zip file first before running the code.

2- execute 'python main.py' on terminal or run the script from an IDE.

3- after execution, to visualize results, run the Gephi software on the generated temporal meta-data in the graphs folder.


