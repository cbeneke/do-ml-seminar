# Problem Statement

- NIF - Neural Implicit Flow Implementation
  - By Shaowou Pan, Steven Brunton, Nathan Kutz

- High dimensional spatio-temporal dynamics encoded in low-dimensional subspace
  - Modeling, Characterisation, Controlling relies on dimensionality reduction for real-time usage
  - PDEs modeling directly are computational challenging
  - Current approach: (Linear) Singular Value Decomposition [SVD] (NonLinear) Convolutional Autoencoders [CAE]
  - Lack ability to represent complexity required for examples
- Examples
  - Calculating the sea surface temperature over time given a few measurement points
  - Turbulence modeling with on a moving object (e.g. wings of a bird)

# New Approach
- Close to a Hypernetwork structure, feeding coordinates into shape-net, time and measurements (mu) into parameter net
- Parameter net learns the weights and biases for Shapenet for specific points
- During inference all inputs are used
- efficient, nonlinear dimensionality reduction
- Interpretable Representations

# Contributions
- Paper mentioned performance is suboptimal (5 days training compared to a few hours) ~> Optimisation
- Current approaches are bad with chaotic systems
- NIF shows 40% better performance, and doesn’t suffer problems with chaotic

# Contributions - Performance
- Implementation in Tensorflow, plan to port to PyTorch
- Paper mentions strong improvement with SIREN layer, but uses own, slimmed down implementation
- Validating if using upstream performs better
- Paper leaves distributed learning to for future work
  - Looking into what I can achieve here - potential strong speedup in learning phase

# Contributions - Evaluation
- Additionally compare to other Hypernetworks, LoRA, etc
- Compare again with results from own Implementation
