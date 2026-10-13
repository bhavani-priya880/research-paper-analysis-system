"""
Enriched arXiv sample dataset with 20 distinct, realistic papers per domain (120 total papers).
Ensures diverse terminology across:
1. Computer Science (transformers, vision, graphs, RL, systems, compilers)
2. Mathematics (optimization, PDEs, algebraic geometry, number theory, probability, topology)
3. Physics (quantum computing, relativity, high-energy particles, condensed matter, astrophysics)
4. Statistics (covariance estimation, Bayesian MCMC, causal inference, high-dimensional hypothesis testing)
5. Quantitative Biology (AlphaFold protein folding, single-cell RNA, neurobiology, synthetic biology)
6. Electrical Engineering (MIMO 5G/6G, MRI reconstruction, control systems, radar signal processing)
"""

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_PAPERS = {
    "Computer Science": [
        ("Attention Is All You Need for Transformer Networks", "Recurrent and convolutional neural networks are standard for translation. We propose the Transformer based solely on self-attention mechanisms with multi-head attention.", "cs.CL"),
        ("Deep Residual Learning for Image Recognition", "Deeper neural networks are harder to optimize. We present residual learning frameworks that reformulate layers as learning residual functions with reference to inputs on ImageNet.", "cs.CV"),
        ("Semi-Supervised Classification with Graph Convolutional Networks", "We present a scalable approach for semi-supervised learning on graph-structured data using localized spectral graph convolutions.", "cs.LG"),
        ("Mastering the Game of Go with Deep Neural Networks and Monte Carlo Tree Search", "We introduce AlphaGo combining value networks to evaluate board positions and policy networks to select moves trained via reinforcement learning.", "cs.AI"),
        ("Practical Byzantine Fault Tolerance for Replicated Services", "We describe a new Byzantine fault-tolerant algorithm for state machine replication that survives malicious attacks in asynchronous environments.", "cs.DC"),
        ("BERT Pretraining of Deep Bidirectional Transformers for Language Understanding", "We introduce BERT which pretrains deep bidirectional representations by conditioning on left and right context in all layers via masked language models.", "cs.CL"),
        ("YOLO You Only Look Once Unified Real-Time Object Detection", "We frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities using deep convolutional backbones.", "cs.CV"),
        ("Generative Adversarial Nets for Unsupervised Representation Learning", "We propose generative adversarial networks where a generative model captures data distribution while a discriminative model estimates probability of sample authenticity.", "cs.LG"),
        ("Asynchronous Methods for Deep Reinforcement Learning", "We propose asynchronous gradient descent for continuous control where actor-critic agents parallelize experience collection across multiple worker threads.", "cs.AI"),
        ("Raft In Search of an Understandable Consensus Algorithm", "Raft decomposes consensus into leader election, log replication, and safety guarantees with formal state invariants across distributed clusters.", "cs.DC"),
        ("RoBERTa A Robustly Optimized BERT Pretraining Approach", "We present a replication study of BERT showing that careful hyperparameter tuning, longer training, larger batch sizes, and dynamic masking improve performance.", "cs.CL"),
        ("Mask R-CNN for Instance Segmentation", "We extend Faster R-CNN by adding a branch for predicting segmentation masks in parallel with the existing bounding box recognition branch.", "cs.CV"),
        ("Adam A Method for Stochastic Optimization in Deep Networks", "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions with adaptive learning rates based on moments.", "cs.LG"),
        ("Compiler Optimization for Domain Specific Hardware Accelerators", "We present polyhedral compilation techniques targeting systolic arrays and specialized tensor processing units with loop tiling and vectorization.", "cs.AR"),
        ("Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge", "We formalize zk-SNARK protocols for verifying arbitrary NP computations with logarithmic proof size and sub-second verification times.", "cs.CR"),
        ("Proximal Policy Optimization Algorithms for Reinforcement Learning", "We propose a novel family of policy gradient methods that clip probability ratios to prevent destructively large policy updates in continuous control.", "cs.AI"),
        ("An Image is Worth 16x16 Words Transformers for Image Recognition at Scale", "We show that pure transformers applied directly to sequences of image patches perform exceptionally well on computer vision classification tasks.", "cs.CV"),
        ("Self-Supervised Learning of Graph Embeddings via Contrastive Pretraining", "We introduce a graph contrastive learning framework that maximizes agreement between augmented topological graph views via mutual information.", "cs.LG"),
        ("Static Analysis and Symbolic Execution for Smart Contract Vulnerability Detection", "We build a symbolic virtual execution engine to detect reentrancy bugs, integer overflows, and unhandled exceptions in EVM bytecode.", "cs.SE"),
        ("Deep Speech 2 End-to-End Speech Recognition in English and Mandarin", "We describe an end-to-end deep learning system that handles noisy speech with convolutional layers, bidirectional recurrent cells, and CTC loss.", "cs.CL")
    ],
    "Mathematics": [
        ("Convergence Analysis of Stochastic Gradient Descent in Non-Convex Hilbert Spaces", "We establish non-asymptotic convergence guarantees for stochastic gradient algorithms under Polyak-Lojasiewicz inequality and relaxed variance bounds.", "math.OC"),
        ("Uniform Geometric Ergodicity of Markov Semigroups on Riemannian Manifolds", "We prove spectral gap estimates and log-Sobolev inequalities for diffusion operators on compact Riemannian manifolds with bounded Ricci curvature.", "math.PR"),
        ("Hodge Classes and Étale Cohomology Rings on Projective Algebraic Varieties", "We study motivic decomposition and cycle classes on smooth projective varieties over finite fields using crystalline cohomology and p-adic Hodge theory.", "math.AG"),
        ("Existence and Regularity of Weak Solutions to Quasilinear Elliptic Equations", "We prove Harnack inequalities and Hölder continuity for quasilinear divergence-form elliptic operators with singular drift vector fields.", "math.AP"),
        ("Log-Concavity of the Characteristic Polynomial of Matroids and Chow Rings", "We establish Hodge-Riemann relations for matroid intersection theory, resolving Welsh's conjecture on log-concavity of independent sets.", "math.CO"),
        ("Subconvexity Bounds for Automorphic L-Functions and Kuznetsov Trace Formulas", "We prove subconvexity bounds on the critical line for GL(2) automorphic L-functions using spectral decomposition of the hyperbolic Laplacian.", "math.NT"),
        ("Symplectic Floer Homology and Lagrangian Intersections in Cotangent Bundles", "We construct filtered Floer homology groups for exact Lagrangian submanifolds, proving the Arnold conjecture for compact symplectic manifolds.", "math.SG"),
        ("Monge-Ampere Equations and Calabi-Yau Metrics on Complex Kähler Manifolds", "We establish a priori C2 estimates and uniform Laplacian bounds for complex Monge-Ampère equations on compact Kähler manifolds.", "math.DG"),
        ("Quasi-Periodic Solutions to Non-Linear Schrödinger Equations via KAM Theory", "We prove persistence of invariant tori for Hamiltonian infinite-dimensional partial differential equations with Dirichlet boundary conditions.", "math.DS"),
        ("Classification of Finite Simple Groups and Modular Representation Theory", "We investigate blocks of defect zero and Harish-Chandra induction for finite groups of Lie type over algebraically closed fields.", "math.RT"),
        ("Global Well-Posedness of Navier-Stokes Equations in Critical Besov Spaces", "We establish global existence of mild solutions to 3D incompressible Navier-Stokes equations with small initial data in negative Besov regularity.", "math.AP"),
        ("Duality Theory for Non-Convex Semidefinite Programming and Rank Relaxations", "We derive exact penalty functions and zero duality gap conditions for rank-constrained semidefinite optimization problems.", "math.OC"),
        ("Extreme Value Theory for Long-Memory Stationary Gaussian Sequences", "We prove limit theorems for normalized maxima of Gaussian processes with fractional Brownian covariance kernels.", "math.PR"),
        ("Intersection Theory on Deligne-Mumford Moduli Stacks of Stable Curves", "We compute Mumford tautological classes and intersection numbers of psi-classes on Deligne-Mumford compactifications of moduli spaces.", "math.AG"),
        ("Ramsey Numbers and Hypergraph Regularity Lemmas", "We establish exponential lower bounds for multicolor hypergraph Ramsey numbers using probabilistic construction and quasirandomness.", "math.CO"),
        ("Zeros of the Riemann Zeta Function on the Critical Strip and Pair Correlation", "We examine Montgomery's pair correlation conjecture and deduce spacing distributions of non-trivial zeros under Riemann Hypothesis.", "math.NT"),
        ("Derived Categories of Coherent Sheaves and Fourier-Mukai Transforms", "We prove autoequivalence classifications for derived categories of K3 surfaces using lattice theory and bridgeland stability conditions.", "math.AG"),
        ("Free Boundary Problems for Minimal Surfaces and Obstacle Problems", "We analyze the regularity of free boundaries for two-phase elliptic variational inequalities using monotonicity formulas.", "math.AP"),
        ("Optimal Transport Distances and Ricci Curvature Bounds on Metric Measure Spaces", "We characterize synthetic Ricci curvature lower bounds via convexity of Boltzmann entropy along Wasserstein geodesics.", "math.MG"),
        ("Asymptotic Expansion of Jacobi Polynomials and Orthogonal Ensembles", "We derive Riemann-Hilbert asymptotic expansions for orthogonal polynomials with varying exponential weights on the real line.", "math.CA")
    ],
    "Physics": [
        ("Continuous Quantum Error Correction on Superconducting Surface Code Qubits", "We demonstrate fault-tolerant quantum error correction suppressing phase-flip and bit-flip errors exponentially in a planar superconducting processor.", "quant-ph"),
        ("Observation of Gravitational Waves from Binary Black Hole Coalescence", "LIGO detectors recorded a gravitational wave transient consistent with general relativistic predictions for a binary black hole merger.", "gr-qc"),
        ("Majorana Zero Modes in Topological Semiconductor Superconductor Nanowires", "We demonstrate zero-bias conductance peaks in indium antimonide nanowires in proximity to aluminum superconductors, confirming Majorana bound states.", "cond-mat.supr-con"),
        ("Precise Measurement of the Higgs Boson Mass and Decay Branching Fractions", "ATLAS and CMS experiments report combined measurements of Higgs boson couplings to vector bosons and fermions at 13 TeV collision energy.", "hep-ex"),
        ("Cosmological Parameters from Planck Cosmic Microwave Background Anisotropy", "Measurements of temperature and polarization power spectra from Planck satellite confirm six-parameter Lambda-CDM cosmological concordance.", "astro-ph.CO"),
        ("Holographic Duality Between Anti-de Sitter Gravity and Superconformal Field Theories", "We calculate entanglement entropy and Wilson loops in AdS/CFT correspondence connecting N=4 Yang-Mills theory with string theory on AdS5 x S5.", "hep-th"),
        ("Bose-Einstein Condensation in Dilute Alkali Atomic Gases", "We achieve Bose-Einstein condensation of rubidium atoms cooled below 170 nanokelvin using laser cooling and magnetic evaporative traps.", "cond-mat.quant-gas"),
        ("Neutrino Oscillation Parameters and Mass Hierarchy in Long-Baseline Detectors", "We measure muon neutrino disappearance and electron neutrino appearance probabilities, determining atmospheric mass splitting and mixing angles.", "hep-ex"),
        ("Thermodynamics of Kerr-Newman Black Holes and Hawking Radiation Flux", "We calculate quantum radiation emission spectra from rotating charged black hole horizons using quantum field theory in curved spacetime.", "gr-qc"),
        ("Fractional Quantum Hall Effect and Non-Abelian Anyon Statistics", "We probe fractional charge and braiding statistics of quasiparticles at Landau level filling factor 5/2 in high-mobility GaAs heterostructures.", "cond-mat.mes-hall"),
        ("Dark Matter Annihilation Cross Sections from Galactic Center Gamma-Ray Flux", "Fermi-LAT observations place upper bounds on thermal relic WIMP cross sections from diffuse gamma-ray excess searches in the Galactic center.", "astro-ph.HE"),
        ("Quantum Teleportation of Discrete and Continuous Variable States", "We demonstrate high-fidelity quantum teleportation of photonic qubits across a free-space optical link using Einstein-Podolsky-Rosen entanglement.", "quant-ph"),
        ("Non-Perturbative QCD Lattice Simulations of Nucleon Axial Form Factors", "We compute nucleon axial charge and electromagnetic radii using 2+1 flavor lattice quantum chromodynamics at physical pion mass.", "hep-lat"),
        ("Exoplanet Atmosphere Transmission Spectroscopy with James Webb Space Telescope", "JWST NIRSpec observations reveal carbon dioxide and water vapor absorption features in the transmission spectrum of transiting hot Jupiter planets.", "astro-ph.EP"),
        ("High-Temperature Superconductivity in Cuprate and Nickelate Thin Films", "We investigate d-wave pairing symmetry and antiferromagnetic spin fluctuations in infinite-layer nickelate superconductors under epitaxial strain.", "cond-mat.supr-con"),
        ("Perturbative Unification of Gauge Couplings in Supersymmetric Grand Unified Theories", "We compute two-loop renormalization group equations for SU(5) and SO(10) SUSY GUT models consistent with precision electroweak LEP data.", "hep-ph"),
        ("Stochastic Gravitational Wave Background from Pulsar Timing Arrays", "NANOGrav reports evidence for an isotropic stochastic background of nanohertz gravitational waves consistent with supermassive black hole binaries.", "astro-ph.CO"),
        ("Laser Interferometric Squeezed Vacuum Injection Beyond Standard Quantum Limit", "We inject frequency-dependent squeezed vacuum states into optical resonators to suppress quantum radiation pressure and shot noise simultaneously.", "quant-ph"),
        ("Non-Equilibrium Phase Transitions and Critical Quenches in Quantum Spin Chains", "We study dynamical quantum phase transitions following quantum quenches in transverse-field Ising models using tensor network approximations.", "cond-mat.stat-mech"),
        ("Quark-Gluon Plasma Hydrodynamics in Ultra-Relativistic Heavy Ion Collisions", "We analyze elliptic flow and shear viscosity to entropy density ratios in relativistic hydrodynamic simulations of gold-gold collisions at RHIC.", "nucl-th")
    ],
    "Statistics": [
        ("High-Dimensional Covariance Matrix Estimation via Adaptive Thresholding", "We investigate sparse covariance estimators when dimension p exceeds sample size n. We prove consistency in operator norm and minimax convergence rates.", "stat.ME"),
        ("Posterior Contraction Rates in Nonparametric Bayesian Regression Models", "We establish Bernstein-von Mises theorems for Gaussian process priors in functional linear regression, proving asymptotic coverage of credible sets.", "stat.TH"),
        ("Doubly Robust Estimation of Causal Treatment Effects in Cohort Studies", "We propose an augmented inverse propensity weighted estimator combining generalized additive propensity models with outcome regression.", "stat.AP"),
        ("Hamiltonian Monte Carlo on Riemannian Manifolds for High-Dimensional Posterior Sampling", "We utilize Riemannian geometry and Fisher information metrics to navigate complex posterior distributions with varying local curvature in MCMC.", "stat.CO"),
        ("Sharp Generalization Error Bounds for Deep Neural Networks via Rademacher Complexity", "We derive generalization bounds for multi-layer neural networks using localized Rademacher complexity and spectral norm capacity control.", "stat.ML"),
        ("Exact Post-Selection Inference and Polyhedral Conditioning for the Lasso", "We develop exact hypothesis testing methods for linear regression after variable selection via the Lasso by conditioning on polyhedral selection sets.", "stat.ME"),
        ("Empirical Bayes Methods for Large-Scale Simultaneous Hypothesis Testing", "We formulate local false discovery rate estimators using empirical Bayes non-parametric mixture deconvolution over z-score distributions.", "stat.ME"),
        ("Optimal Transport Distances for Two-Sample Testing in Multivariate Spaces", "We introduce permutation tests based on sliced Wasserstein distances to evaluate whether two high-dimensional distributions share identical density.", "stat.TH"),
        ("Variational Inference with Normalizing Flows for Complex Bayesian Posteriors", "We combine mean-field variational inference with invertible neural networks to generate flexible non-Gaussian posterior approximations.", "stat.CO"),
        ("Uniform Convergence and Concentration of Measure for U-Statistics in Metric Spaces", "We derive exponential concentration inequalities for degenerate U-statistics over Hoeffding decompositions in unbounded random variables.", "stat.TH"),
        ("Longitudinal Mixed-Effects Models for Irregularly Sampled Biomarker Trajectories", "We develop stochastic expectation-maximization algorithms for linear mixed models with non-ignorable missingness and autoregressive correlation.", "stat.AP"),
        ("Sparse Principal Component Analysis via Orthogonal Dictionary Learning", "We formulate sparse PCA as a regularized matrix factorization problem solved via alternating minimization with convergence guarantees.", "stat.ML"),
        ("Bayesian Non-Parametric Dirichlet Process Mixture Models for Density Estimation", "We derive Gibbs sampling routines for stick-breaking Dirichlet process mixtures and prove L1 posterior consistency for smooth density targets.", "stat.TH"),
        ("False Discovery Rate Control Under Arbitrary Covariance and Dependence", "We prove that Benjamini-Hochberg procedures control false discovery rates under positive regression dependency from a subset of null hypotheses.", "stat.ME"),
        ("Conformal Prediction Sets with Finite-Sample Valid Coverage Guarantees", "We construct distribution-free prediction intervals with exact marginal coverage guarantees using split conformal inference on holdout calibration data.", "stat.ML"),
        ("Spatial Point Process Modeling for Clustered Environmental Sensor Networks", "We fit log-Gaussian Cox processes with Matérn covariance structures to predict pollutant concentrations using integrated nested Laplace approximations.", "stat.AP"),
        ("Spectral Clustering and Minimax Error Rates in Stochastic Block Models", "We establish sharp threshold bounds for community detection in sparse network graphs using eigenvectors of the normalized graph Laplacian.", "stat.TH"),
        ("Robust Estimation of Location and Scatter Using Minimum Covariance Determinant", "We evaluate high-breakdown affine-equivariant estimators for multivariate location and scatter in the presence of severe asymmetric outliers.", "stat.ME"),
        ("Survival Analysis with Time-Varying Covariates and High-Dimensional Penalization", "We extend Cox proportional hazards models with elastic net regularization to identify genomic markers associated with patient survival times.", "stat.AP"),
        ("Sequential Monte Carlo Particle Filtering for Non-Linear State Space Models", "We introduce adaptive resampling schemes for particle filters in non-Gaussian dynamical tracking systems to alleviate particle degeneracy.", "stat.CO")
    ],
    "Quantitative Biology": [
        ("Atomic Accuracy Protein Structure Prediction with AlphaFold", "AlphaFold uses spatial graph networks and evolutionary attention to predict 3D coordinates of proteins directly from amino acid primary sequences.", "q-bio.BM"),
        ("Single-Cell RNA Sequencing and Trajectory Inference in Hematopoiesis", "We map cellular differentiation trajectories during bone marrow stem cell maturation using single-cell transcriptomics and diffusion pseudotime.", "q-bio.GN"),
        ("Spike-Timing-Dependent Plasticity in Recurrent Biological Neural Circuits", "We model biophysical Hodgkin-Huxley cortical neurons to determine how dendritic spine remodeling controls long-term potentiation and depression.", "q-bio.NC"),
        ("Stochastic Bursting and Master Equations in Gene Regulatory Networks", "We formulate chemical master equations to model transcriptional bursting, promoter switching, and mRNA synthesis validated by smFISH imaging.", "q-bio.MN"),
        ("Constraint-Based Genome-Scale Metabolic Flux Analysis in Microbial Systems", "Using 13C metabolic flux balance analysis, we identify enzymatic knock-out targets in Escherichia coli to maximize biosynthetic biofuel yields.", "q-bio.SC"),
        ("Compartmental Epidemic Transmission on Scale-Free Demographic Contact Networks", "We formulate SIR epidemic models on heterogeneous scale-free contact networks and compute analytical percolation thresholds for pathogen spread.", "q-bio.PE"),
        ("Cryo-EM Reconstruction of Ribosomal Subunit Translation Intermediates", "Cryo-electron microscopy single-particle analysis reveals high-resolution conformational states of the eukaryotic 80S ribosome during elongation.", "q-bio.BM"),
        ("CRISPR-Cas9 Off-Target Cleavage Prediction Using Deep Convolutional Networks", "We train sequence models on genome-wide cleavage assays to predict guide RNA off-target binding affinities and chromatin accessibility effects.", "q-bio.GN"),
        ("Neural Population Dynamics and Low-Dimensional Latent Manifolds in Motor Cortex", "High-density multi-electrode recordings during reaching tasks reveal that motor cortical activity reflects rotational dynamical trajectories.", "q-bio.NC"),
        ("Oscillatory Dynamics and Entrainment in Circadian Gene Clock Networks", "We analyze non-linear limit cycles in transcriptional-translational feedback loops governing molecular circadian rhythms in mammalian cells.", "q-bio.MN"),
        ("Cell Membrane Permeability and All-Atom Molecular Dynamics of Ion Channels", "We simulate voltage-gated potassium ion channels in lipid bilayers using all-atom molecular dynamics to resolve voltage-sensing gate motions.", "q-bio.BM"),
        ("Evolutionary Fitness Landscapes and Epistatic Interactions in Viral Genomes", "Deep mutational scanning of viral surface glycoproteins maps how higher-order epistatic interactions dictate antibody escape mutations.", "q-bio.PE"),
        ("High-Throughput Spatial Transcriptomics in Mammalian Brain Slices", "We combine multiplexed in situ hybridization with spatial indexing to resolve cell-type specific gene expression in intact neocortical tissue.", "q-bio.GN"),
        ("Synaptic Neurotransmitter Release Kinetics and Vesicle Pool Depletion", "We model calcium-triggered exocytosis at presynaptic active zones to predict short-term synaptic depression during high-frequency firing.", "q-bio.NC"),
        ("Synthetic Gene Circuit Construction for Programmable Cell Therapy", "We engineer mammalian synthetic Notch receptor circuits that activate chimeric antigen receptor expression strictly upon multi-antigen sensing.", "q-bio.SC"),
        ("Biomechanics of Flagellar Locomotion in Viscous Microfluidic Environments", "We measure hydrodynamics and flagellar propulsion of swimming microorganisms in low Reynolds number fluidic chambers using high-speed tracking.", "q-bio.CB"),
        ("Deep Learning for Quantitative Fluorescence Microscopy Cell Segmentation", "We train residual U-Net architectures to segment touching cell nuclei in multi-channel confocal fluorescence microscopy imaging datasets.", "q-bio.QM"),
        ("Phylogenetic Tree Inference from Whole-Genome Sequence Alignments", "We implement maximum likelihood phylogenetic algorithms with site-specific substitution rates to reconstruct ancient speciation branching.", "q-bio.PE"),
        ("Co-Translational Protein Folding and Ribosome Stalling Mechanics", "Optical tweezer force spectroscopy measures folding forces exerted by nascent polypeptide chains emerging from the ribosomal exit tunnel.", "q-bio.BM"),
        ("Tumor Microenvironment Clonal Evolution and Drug Resistance Mutations", "Whole-exome longitudinal sequencing of glioblastoma biopsies tracks subclonal evolutionary selection under alkylating chemotherapy agents.", "q-bio.GN")
    ],
    "Electrical Engineering": [
        ("Massive MIMO Channel Estimation and Signal Detection Using Deep Learning", "Millimeter-wave 5G massive MIMO requires accurate CSI. We introduce deep convolutional neural networks for compressed channel reconstruction.", "eess.SP"),
        ("Deep Physics-Guided Unrolled Networks for Accelerated MRI Reconstruction", "Accelerated MRI reconstructs images from undersampled k-space data. We formulate an unrolled iterative network integrating coil sensitivity maps.", "eess.IV"),
        ("Model Predictive Control with Lyapunov Guarantees for Autonomous Vehicles", "We design an integrated MPC and reinforcement learning controller for high-speed vehicle trajectory tracking under uncertain tire friction.", "eess.SY"),
        ("Conformer Transducer Models for Real-Time Edge Hardware Speech Recognition", "We optimize 8-bit quantized conformer neural architectures for streaming automatic speech recognition on battery-powered microcontrollers.", "eess.AS"),
        ("Compressed Sensing Synthetic Aperture Radar Imaging via ADMM Optimization", "We formulate radar image formation as an L1-regularized sparse inverse problem solved via alternating direction method of multipliers.", "eess.SP"),
        ("Decentralized Frequency Control for Low-Inertia Power Grids with Wind Energy", "We introduce robust H-infinity distributed control for battery energy storage systems to counter frequency volatility in low-inertia grids.", "eess.SY"),
        ("Deep Super-Resolution of Optical Satellite Imagery Using Attention GANs", "We design generative adversarial networks with multi-scale residual attention blocks to upscale low-resolution multispectral satellite images.", "eess.IV"),
        ("End-to-End Beamforming Optimization in Reconfigurable Intelligent Surfaces", "We formulate mutual coupling and phase-shift optimization for wireless communication networks assisted by reconfigurable intelligent metasurfaces.", "eess.SP"),
        ("Acoustic Echo Cancellation Using Kalman Filtering and Deep Neural Post-Filters", "We combine partitioned-block frequency-domain adaptive Kalman filtering with deep recurrent neural networks for nonlinear acoustic echo removal.", "eess.AS"),
        ("Fault Detection and Resilient Microgrid State Estimation via Graph Neural Nets", "We apply spatio-temporal graph neural networks to phasor measurement unit telemetry to isolate power transmission line faults in sub-cycle time.", "eess.SY"),
        ("Hyperspectral Image Denoising via Non-Local Low-Rank Tensor Decomposition", "We propose tensor subspace representation with non-local spatial regularization to remove mixed stripe and Gaussian noise from hyperspectral bands.", "eess.IV"),
        ("Orthogonal Time Frequency Space Modulation for High-Mobility Vehicular Wireless", "OTFS modulation multiplexes information symbols in the delay-Doppler domain. We construct low-complexity equalization algorithms for high Doppler.", "eess.SP"),
        ("Adaptive Cruise Control and Platoon String Stability in Connected Vehicles", "We develop distributed sliding mode controllers for connected vehicle platoons that guarantee string stability under wireless packet loss.", "eess.SY"),
        ("Deep Audio Source Separation for Multi-Speaker Mixtures with TasNet", "We implement time-domain audio separation networks with dilated convolutional encoders to isolate speech signals in challenging reverberant rooms.", "eess.AS"),
        ("Phase Retrieval in Optical Diffraction Tomography Using Deep Image Priors", "We solve holographic phase retrieval inverse problems by utilizing untrained convolutional networks as structural regularizers.", "eess.IV"),
        ("Energy-Efficient Resource Allocation in Cell-Free Massive MIMO Networks", "We optimize downlink power control and access point selection in cell-free networks using fractional programming and successive convex approximations.", "eess.SP"),
        ("Nonlinear Observer Design for Lithium-Ion Battery State-of-Charge Estimation", "We construct unscented Kalman filters combined with equivalent circuit electrochemical models to estimate state-of-charge under dynamic driving cycles.", "eess.SY"),
        ("Deep Video Frame Interpolation via Bi-Directional Optical Flow Synthesis", "We propose an end-to-end convolutional network with warp-and-blend architecture to generate intermediate video frames at high frame rates.", "eess.IV"),
        ("Speech Enhancement in Non-Stationary Acoustic Noise Using Spectral Masking", "We train dual-path recurrent neural networks on magnitude and complex spectra to restore speech intelligibility in cockpit environments.", "eess.AS"),
        ("Distributed Voltage Regulation in Active Distribution Feeders with Solar PV", "We propose decentralized peer-to-peer reactive power optimization to mitigate overvoltage conditions caused by rooftop solar photovoltaic generation.", "eess.SY")
    ]
}

records = []
paper_idx = 1
for domain, papers in DOMAIN_PAPERS.items():
    for title, summary, cat_code in papers:
        records.append({
            "id": f"arxiv_{paper_idx:04d}",
            "title": title,
            "summary": summary,
            "summary_word_count": len(summary.split()),
            "category": domain,
            "category_code": cat_code,
            "published_date": "2023-05-15",
            "updated_date": "2023-06-01",
            "authors": f"Lead Author et al. ({domain})",
            "first_author": "Lead Author"
        })
        paper_idx += 1

df = pd.DataFrame(records)
sample_csv_path = DATA_DIR / "sample_arxiv_metadata.csv"
df.to_csv(sample_csv_path, index=False)
print(f"Enriched dataset generated: {len(df)} papers across {len(DOMAIN_PAPERS)} domains saved to {sample_csv_path}")
