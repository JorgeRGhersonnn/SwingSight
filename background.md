# Golf Ball Simulation 

The ultimate goal of this simulation is to use Monte Carlo simulations to predict the landing radius of a golf ball. This simulation takes into consideration a non-ideal scenario, 
where drag forces and lift forces (Magnus Effect) are taken into consideration. 

## Equations of Motion

Consider a golf ball of mass $m$ with a cross-sectional area $A$ that experiences projectile motion when hit by a golf club. To model the trajectory of said ball, we begin by Newton's Second Law, 

\begin{equation}
    \sum_i \vec{F} = \frac{\partial \vec{p}}{\partial t}
\end{equation}