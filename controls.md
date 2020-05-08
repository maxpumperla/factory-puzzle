# Controls

- Tables are independent agents in the sense of RL.
- Tables can either stay where they are, move up, down, left or right.
- Some of these movements might be illegal given the current state. The RL algorithm is 
supposed to learn and avoid illegal patterns.
- We define a "rail" as the trajectory of a movable shuttle. 
Each rail has precisely one shuttle, which we call the rail shuttle.
- If a table is next to a rail and its rail shuttle bears no table, the table can "order" the
shuttle to enter the rail. By this we mean that the shuttle will move next to the table and the
table will then board this shuttle.
- Once a table is on a shuttle the same controls as before apply (four basic directions and halting).
In particular, i.e. the table can either legally move along the rail (on its shuttle) or leave
the rail again if an adjacent node is free.
- These controls allows us to model the whole scenario with precisely one agent type.
