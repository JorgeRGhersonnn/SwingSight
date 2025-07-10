# Golf Ball Simulation 

The ultimate goal of this simulation is to use Monte Carlo simulations to predict the landing radius of a golf ball. This simulation takes into consideration a non-ideal scenario, 
where drag forces and lift forces (Magnus Effect) are taken into consideration. 

## Equations of Motion

Consider a golf ball of mass $m$ with a cross-sectional area $A$ that experiences projectile motion when hit by a golf club. To model the trajectory of said ball, we begin by Newton's Second Law, 

$$
\begin{equation}
    \sum_i \vec{F}_i = \frac{\partial \vec{p}}{\partial t},
\end{equation}
$$

where $\vec{F}$ represents the Force vector, and $\vec{p}$ represents the ball's momentum over time. Now, we assume that the ball's mass is invariant throughout its trajectory, so it is independent of time. Since $\vec{p} = m\vec{v}$, the right-hand-side becomes

$$
\begin{equation}
    \frac{\partial \vec{p}}{\partial t} = \frac{\partial}{\partial t}\left( m\vec{v}\right) = m \frac{\partial \vec{v}}{\partial t}.
\end{equation}
$$

Assuming air drag is not negligible and a possibility for spinning (i.e., magnus effect): 

$$
\begin{equation}
    \vec{F}_\text{grav} + \vec{F}_\text{drag} + \vec{F}_\text{magnus} = m \frac{\partial \vec{v}}{\partial t}
\end{equation}
$$

Gravitational forces only act on the $z$-axis. Air drag is affected by the air's density $\rho$, the cross-sectional area of the ball, its velocity $\vec{v}$, its magnitude $\mid \vec{v} \mid$, and a drag coefficient $C_d$ which opposes the ball's motion such that, 

$$
\begin{equation}
    \vec{F}_\text{drag} = -\frac{1}{2}\rho C_d A \vec{v} \mid \vec{v} \mid.
\end{equation}
$$(drag)

Similarly, the magnus effect (spinning) arises from a lift coefficient $C_L$ and the unit vector of the angular velocity $\hat{\omega}$, which determines the direction of the spin, such that:

$$
\begin{equation}
    \vec{F}_\text{magnus} = -\frac{1}{2}\rho C_L A \mid \vec{v}\mid^2 \hat{\omega}\times \hat{v}.
\end{equation}
$$(magnus)

By the fact that $\hat{\omega} = \frac{\vec{\omega}}{\mid \vec{\omega}\mid}$, 

$$
\begin{equation}
    \vec{F}_\text{magnus} = -\frac{1}{2}\rho C_L A \mid \vec{v}\mid^2 \left( \frac{\vec{\omega}\times \vec{v}}{\mid \vec{\omega}\mid \mid \vec{v}\mid}\right).
\end{equation}
$$

Therefore, we have the trajectory of the golf ball modelled by 

$$
\begin{equation}
    \vec{F}_\text{grav} - \frac{1}{2}\rho C_d A \vec{v} \mid \vec{v} \mid -\frac{1}{2}\rho C_L A \mid \vec{v}\mid^2 \left( \frac{\vec{\omega}\times \vec{v}}{\mid \vec{\omega}\mid \mid \vec{v}\mid}\right) 
    = m \frac{\partial \vec{v}}{\partial t}.
\end{equation}
$$


## Monte Carlo Implementation

