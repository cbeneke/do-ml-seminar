Source Code for my Submission in the WS2024 Discrete Optimization and Machine Learning Seminar.

Neural Implicit Flow (NIF)[1] was proposed as a powerful approach for representing continuous functions in various domains, particularly for spatio-temporal data modeled
by PDEs. This paper presents a comparative study of three different implementations of NIFs: an upstream reference implementation, a PyTorch-based approach, and a TensorFlow
Functional API design. We evaluate these implementations on both simple periodic and complex high-frequency wave functions, analyzing their performance, convergence
characteristics, and implementation trade-offs. Our results demonstrate that while all implementations successfully modeled the target functions, they exhibited different
strengths in terms of convergence speed, accuracy, and code maintainability. The PyTorch implementation with Adam Optimiser showed superior performance for high-frequency
cases, achieving up to 4x better loss values compared to the baseline.

[1]: https://arxiv.org/pdf/2204.03216
