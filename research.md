# Models to choose from:
Cellular automata or agent-based models.

# Case study: How was the racial segregation agent-based model developed?
The racial segregation agent-based model was developed by economist Thomas Schelling in the 1970s, well before the advent of modern computational tools. Schelling wanted to understand how small individual preferences can lead to a larger, unanticipated systemic outcome. Here is a general outline of how the Schelling segregation model was developed and how it works:

### **Development of Schelling’s Segregation Model**

#### **1. Identifying the Research Question**
Schelling wanted to explore how racial segregation in neighborhoods could occur even when individuals do not have an explicit preference for living in a neighborhood dominated by their own race.

#### **2. Setting Up the Model**
Schelling set up a grid (like a chess board) where each cell represented a household. The households were of two types, representing two racial groups. A certain fraction of the grid was left vacant.

#### **3. Defining the Rules**
The core of the model was the definition of a satisfaction rule: a household would move if less than a certain proportion of its neighbors were of the same type. If the threshold was not met, the household would move to a vacant cell.

#### **4. Simulation Through Iteration**
Schelling initially conducted the simulations manually, iterating over the grid and moving unsatisfied households to vacant spots until a stable state (where all households met their satisfaction threshold) was reached.

#### **5. Observing the Emergent Phenomena**
Even with a moderate preference for having neighbors of the same type, high levels of segregation emerged. This showed how moderate individual preferences could lead to a highly segregated society, an emergent phenomenon not anticipated looking at individual preferences.

#### **6. Variations of the Model**
Later, with the advancement of computational tools, this model was simulated using computer programs, allowing for the exploration of larger grids and different configurations more efficiently. Researchers introduced variations to the original model by changing the shapes of neighborhoods, adding more groups, varying the thresholds dynamically, etc.

### **Characteristics of the Model**

- **Agent-Based**: The model is agent-based, with each agent representing a household having a simple rule based on the type of its neighboring agents.
- **Bottom-Up Approach**: The model uses a bottom-up approach, where the global phenomenon of segregation emerges from individual preferences.
- **Thresholds and Preferences**: The key parameters in the model are the thresholds and preferences of the agents which dictate their decision to move.
- **Emergent Properties**: The model showcases how a complex societal pattern can emerge from simple rules at the individual level.

### **Further Analysis**
Post development, the model has been analyzed from various angles, including statistical analysis to identify the point of phase transition, studying the final configurations, and the speed of segregation.

This model remains a seminal work in the field of agent-based modeling, showcasing the power of this approach in understanding complex social phenomena through relatively simple rules governing individual behaviors.
