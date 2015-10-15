##Agent based percolation model of innovation

*Nathan Goldschlag*

*January 15, 2015*

*Version 1.2*


The files in this repository execute and analyze the output of an agent-based percolation model of the innovation process. The model extends the percolation framework in Silverberg and Verspagen 2007. The model incorporates patents which can: 1) provide additional resources to innovators to invest in R&D and 2) block other firms from exploring the technology space.

To execute the model, modify the tests to run in main(). Below is a list of valid simulation test names. 
['test1a', 'test1b', 'test2a', 'test2b', 'test3a', 'test3b', 'test4a', 'test4b', 'typical', 'microsim']

ABSTRACT:
A model of the innovation process is proposed as a method of understanding the effects of patents in both promoting and stifling innovative search. Traditional economic models of the innovation process and patenting struggle to capture the interdependence of different types of technologies and the path dependence of innovative search. This research addresses this gap by developing an agent-based computational model that directly incorporates path dependence, localized search, uncertainty, and the interdependence of technological innovations. The model is capable of replicating several stylized facts including the skewed distribution of innovation value as well as the temporal clustering of radical innovations. Simulation results are used to investigate how monopoly power, complexity of the technology space, patent breadth, and patent duration affect innovative activity. The results suggest that monopoly power in varying degrees can substitute for patent protection. Patents can improve innovative performance when firms have less monopoly power and the technology space is difficult to navigate. However, when the technology space is less difficult to navigate innovative performance significantly improves without patents. Likewise, when the technology space is simple increased patent breadth and duration stifle innovative search.

Pseudo code:
For each run:
  Create lattice
  Create firms
  For each step:
    Shuffle firms
    For each firm:
      Get new firm position
        Evaluate relative heights 
        If doPatents:
          Evaluate ratio of blocked cells
        Get position to perform R&D on
          Randomly select cell from firm’s local search neighborhood
        If resistance for that position > 0:
          Apply R&D budget to cell
          If new cell resistance < 0:
            Set cell as discovered, value = 1
              If doPatent:
                Firm potentially patents discovered cell
            If cell connected to a cell with value = 2:
              Set cell value = 2
              Cycle through connected cells (chainReaction)
    Calculate profits for each firm, update R&D budgets
    Store data for step
  Store data for run


License: This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. Included libraries are subject to their own licenses.
