# Critically Assess Burgess's Concentric Model through Agent-based Model

This model adopts basic assumptions in Burgess's Concentric Model and attempts to model the spatial distribution of different income groups in the city in ABM to complement Burgess's theory. At every time step, each agent will check if they are happy on the patch (income >= housing_price; density-sensitivity > density; random.random() <= jobs_prob) and move to a satisfactory neighbor if unhappy. 

PeopleAgent
- category (“poor”, “middle”, “rich”)
- income (1, 2, 3)
- density-sensitivity (5, 3, 1)
- happy (True, False) 

PatchAgent 
- layout (“land_price”, “jobs_probs”, “density”)
- density (number of people on patch) 
- land_price (1, 2, 3)
- housing_price (land_price / density)
- jobs_prob

Parameters 
- Frames per Second (Number of frames to change per second)
- Number of Agents (from 90 to 900 and changes 9 for each step)
- Layout Choices (“land_price”, “jobs_probs”, “density”) 



