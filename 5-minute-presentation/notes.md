# Intro
- NIF - Neural Implicit Flow Implementation
  - By Shaowou Pan, Steven Brunton, Nathan Kutz from University of Washington

# Problem Statement
- What is it about?
  - High dimensional spatio-temporal dynamics
    - Examples
      - Calculating the sea surface temperature over time given just a few measurement points
      - Turbulence modeling with on a moving object (e.g. wings of a bird)
    - In Theory these dynamics can often be encoded in a low-dimensional subspace
    - But Modeling, Characterisation, Controlling these using engineering applications relies on dimensionality reduction to allow real-time usage
  - One approach: direct modeling with partial differential equations [PDEs]
    - is computational challenging - Algorithms 40 years ago perform better than current implementations direct PDEs
  - Current state of the art
    - (Linear) Singular Value Decomposition [SVD]
    - (NonLinear) Convolutional Autoencoders [CAE]
    - Lack ability to represent complexity required for examples mentioned before in a good way

# New Approach
- Authors of this paper proposing
- New model, close to a Hypernetwork structure
  - Feeding coordinates into shape-net
  - Time and measurements (mu) into parameter net
- Parameter net learns the weights and biases for ShapeNet given specific points in space
- During inference all inputs are used
  - NIF models a function "u" for the whole time and space
- Results
  - efficient, nonlinear dimensionality reduction
  - Interpretable Representations

# Contributions
- Remember the results
  - Current approaches are bad with chaotic systems
  - NIF shows 40% better performance
  - Doesn’t suffer problems with chaotic
  - Thus scaling efficiently to complex spatio-temporal datasets
- Paper mentioned training performance is suboptimal (5 days training compared to a few hours) ~> Optimisation
- Additionally I want to re-evaluate results

# Contributions - Performance
- "What I mean by Performance Tuning"
  - Implementation in Tensorflow, plan to port to PyTorch to allow utilisation of both eco-system
- Paper mentions strong improvement with SIREN layer
  - but uses own, slimmed down implementation
  - Validating if using upstream performs better
- Paper leaves distributed learning for future work
  - I believe strong potential speedup in learning phase
  - Looking into what I can achieve here

# Contributions - Evaluation
- Additionally to existing with Singular Value Decomposition and Convolutional Autoencoders compare to other Hypernetworks, LoRA, and other.
- Compare again with results from own Implementation
