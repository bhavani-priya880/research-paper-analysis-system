"""
Generates representative arXiv dataset and sample academic PDFs for local testing.
Matches the sumitm004/arxiv-scientific-research-papers-dataset schema:
id, title, summary, summary_word_count, category, category_code, published_date, updated_date, authors, first_author
"""

import os
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PAPERS_DIR = DATA_DIR / "sample_papers"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Curated scientific research papers across domains
PAPERS_DATA = [
    # Computer Science
    {
        "id": "2301.0001",
        "title": "Attention Is All You Need for Scalable Transformer Architectures",
        "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
        "category": "Computation and Language",
        "category_code": "cs.CL",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar",
        "first_author": "Ashish Vaswani"
    },
    {
        "id": "2301.0002",
        "title": "Deep Residual Learning for Image Recognition and Vision Benchmarks",
        "summary": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs. We evaluate these residual networks on ImageNet classification and COCO object detection benchmarks.",
        "category": "Computer Vision and Pattern Recognition",
        "category_code": "cs.CV",
        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
        "first_author": "Kaiming He"
    },
    {
        "id": "2301.0003",
        "title": "Graph Convolutional Networks for Semi-Supervised Node Classification",
        "summary": "We present a scalable approach for semi-supervised learning on graph-structured data. Our method is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges.",
        "category": "Machine Learning",
        "category_code": "cs.LG",
        "authors": "Thomas N. Kipf, Max Welling",
        "first_author": "Thomas N. Kipf"
    },
    {
        "id": "2301.0004",
        "title": "Mastering the Game of Go with Deep Neural Networks and Tree Search",
        "summary": "The game of Go has long been viewed as the most challenging of classic games for artificial intelligence. We introduce AlphaGo, which evaluates board positions using value networks and selects moves using policy networks. These deep neural networks are trained through supervised learning from human expert moves and reinforcement learning from self-play games.",
        "category": "Artificial Intelligence",
        "category_code": "cs.AI",
        "authors": "David Silver, Aja Huang, Chris J. Maddison",
        "first_author": "David Silver"
    },
    {
        "id": "2301.0005",
        "title": "Distributed Consensus Algorithms in Byzantine Fault Tolerant Systems",
        "summary": "We propose a novel practical Byzantine fault-tolerant consensus algorithm for high-throughput distributed state machine replication. Our protocol guarantees safety under arbitrary asynchronous network partitions and provides liveness under bounded message transmission delays with minimal communication complexity.",
        "category": "Distributed Computing",
        "category_code": "cs.DC",
        "authors": "Leslie Lamport, Robert Shostak",
        "first_author": "Leslie Lamport"
    },
    {
        "id": "2301.0006",
        "title": "Self-Supervised Pretraining for Code Representation and Synthesis",
        "summary": "We introduce CodeBERT, a bimodal pre-trained model for natural language and programming languages. CodeBERT captures the semantic connection between code comments and code implementations via masked language modeling and replaced token detection objectives across multiple languages.",
        "category": "Software Engineering",
        "category_code": "cs.SE",
        "authors": "Zhangyin Feng, Daya Guo, Duyu Tang",
        "first_author": "Zhangyin Feng"
    },

    # Mathematics
    {
        "id": "2301.0010",
        "title": "On the Convergence Rates of Stochastic Gradient Descent in Non-Convex Optimization",
        "summary": "We analyze the non-asymptotic convergence rate of stochastic gradient descent (SGD) for smooth non-convex objective functions under relaxed variance conditions. We prove that adaptive step sizes achieve minimax optimal convergence rates to stationary points in finite dimensional Hilbert spaces with high probability.",
        "category": "Optimization and Control",
        "category_code": "math.OC",
        "authors": "Jean-Pierre Aubin, Hedy Attouch",
        "first_author": "Jean-Pierre Aubin"
    },
    {
        "id": "2301.0011",
        "title": "Ergodic Properties of Markov Chains on Compact Riemannian Manifolds",
        "summary": "This paper establishes uniform geometric ergodicity for a class of continuous-time Markov jump processes on compact Riemannian manifolds. Using logarithmic Sobolev inequalities and Poincaré inequalities, we derive explicit bounds on spectral gaps and convergence in total variation distance.",
        "category": "Probability",
        "category_code": "math.PR",
        "authors": "Cedric Villani, Michel Ledoux",
        "first_author": "Cedric Villani"
    },
    {
        "id": "2301.0012",
        "title": "Hodge Theory and Cohomology of Algebraic Varieties over Finite Fields",
        "summary": "We investigate algebraic cycles and motives on smooth projective varieties. By constructing explicit resolution of singularities and applying p-adic Hodge theory, we establish relations between étale cohomology groups and crystalline cohomology rings across positive characteristic fibrations.",
        "category": "Algebraic Geometry",
        "category_code": "math.AG",
        "authors": "Alexander Grothendieck, Pierre Deligne",
        "first_author": "Alexander Grothendieck"
    },
    {
        "id": "2301.0013",
        "title": "Regularity and Boundary Blow-Up of Solutions to Non-Linear Elliptic Equations",
        "summary": "We investigate weak solutions to quasilinear elliptic partial differential equations with singular absorption terms. We prove existence and uniqueness theorems using variational techniques, Sobolev embeddings, and maximum principles on non-smooth Lipschitz domains.",
        "category": "Analysis of PDEs",
        "category_code": "math.AP",
        "authors": "Luis Caffarelli, Xavier Cabre",
        "first_author": "Luis Caffarelli"
    },
    {
        "id": "2301.0014",
        "title": "Combinatorial Invariants of Matroid Complexes and Chow Rings",
        "summary": "We prove the log-concavity of the coefficients of the characteristic polynomial of matroids using intersection theory on toric varieties. We construct the Chow ring of a matroid and establish the hard Lefschetz theorem and Hodge-Riemann relations for matroid polytopes.",
        "category": "Combinatorics",
        "category_code": "math.CO",
        "authors": "June Huh, Karim Adiprasito, Eric Katz",
        "first_author": "June Huh"
    },
    {
        "id": "2301.0015",
        "title": "L-Functions and Equidistribution of Hecke Eigenvalues in Analytic Number Theory",
        "summary": "We establish subconvexity bounds for automorphic L-functions associated with GL(2) Maass forms. By analyzing spectral decomposition of the hyperbolic Laplacian and applying Kuznetsov trace formulas, we derive explicit error terms for the Sato-Tate distribution of Hecke eigenvalues.",
        "category": "Number Theory",
        "category_code": "math.NT",
        "authors": "Henryk Iwaniec, Peter Sarnak",
        "first_author": "Henryk Iwaniec"
    },

    # Physics
    {
        "id": "2301.0020",
        "title": "Experimental Realization of Fault-Tolerant Quantum Error Correction Codes",
        "summary": "Quantum processors must implement error correction to achieve fault-tolerant computation. We demonstrate continuous quantum error correction on a superconducting qubit processor using the surface code architecture. We show that logical qubits exhibit exponential suppression of bit-flip and phase-flip error rates.",
        "category": "Quantum Physics",
        "category_code": "quant-ph",
        "authors": "John M. Martinis, Charles Neill",
        "first_author": "John M. Martinis"
    },
    {
        "id": "2301.0021",
        "title": "Observation of Gravitational Waves from a Binary Black Hole Coalescence",
        "summary": "On September 14, 2015 at 09:50:45 UTC, the two detectors of the Laser Interferometer Gravitational-Wave Observatory observed a transient gravitational-wave signal. The signal matches the waveform predicted by general relativity for the inspiral and merger of a pair of black holes and the ringdown of the resulting single black hole.",
        "category": "General Relativity and Quantum Cosmology",
        "category_code": "gr-qc",
        "authors": "B. P. Abbott, R. Abbott, T. D. Abbott",
        "first_author": "B. P. Abbott"
    },
    {
        "id": "2301.0022",
        "title": "Topological Superconductivity and Majorana Fermions in Condensed Matter Systems",
        "summary": "We study the topological phase transition in one-dimensional semiconductor nanowires with strong spin-orbit coupling in proximity to an s-wave superconductor. We compute the topological invariant and demonstrate the emergence of localized zero-energy Majorana bound states at the wire boundaries.",
        "category": "Condensed Matter - Superconductivity",
        "category_code": "cond-mat.supr-con",
        "authors": "Roman M. Lutchyn, Jay D. Sau, S. Das Sarma",
        "first_author": "Roman M. Lutchyn"
    },
    {
        "id": "2301.0023",
        "title": "Measurement of the Higgs Boson Mass and Couplings at the Large Hadron Collider",
        "summary": "A measurement of the mass of the Higgs boson is presented based on the combined analysis of the diphoton and four-lepton decay channels. The data correspond to an integrated luminosity of proton-proton collisions recorded by the ATLAS and CMS experiments at the CERN Large Hadron Collider.",
        "category": "High Energy Physics - Experiment",
        "category_code": "hep-ex",
        "authors": "G. Aad, B. Abbott, D. C. Abbott",
        "first_author": "G. Aad"
    },
    {
        "id": "2301.0024",
        "title": "Cosmological Constraints from Cosmic Microwave Background Anisotropies",
        "summary": "We present cosmological parameter constraints from the Planck satellite measurements of temperature and polarization anisotropies in the cosmic microwave background. The results are in robust agreement with the standard six-parameter spatially flat Lambda-CDM cosmological model.",
        "category": "Astrophysics - Cosmology",
        "category_code": "astro-ph.CO",
        "authors": "N. Aghanim, Y. Akrami, M. Ashdown",
        "first_author": "N. Aghanim"
    },
    {
        "id": "2301.0025",
        "title": "Holographic Duals of Conformal Field Theories and Quantum Gravity in Anti-de Sitter Space",
        "summary": "We explore the Anti-de Sitter conformal field theory (AdS/CFT) correspondence relating string theory on AdS5 x S5 to four-dimensional N=4 supersymmetric Yang-Mills theory. We calculate Wilson loop expectation values, correlation functions, and entanglement entropy across black hole horizons.",
        "category": "High Energy Physics - Theory",
        "category_code": "hep-th",
        "authors": "Juan Maldacena, Edward Witten",
        "first_author": "Juan Maldacena"
    },

    # Statistics
    {
        "id": "2301.0030",
        "title": "High-Dimensional Covariance Matrix Estimation and False Discovery Rates",
        "summary": "We investigate high-dimensional covariance estimation when the dimension p exceeds the sample size n. We propose a penalized likelihood estimator with adaptive thresholding. We prove consistency in operator norm and establish asymptotic control over false discovery rates under sparsity constraints.",
        "category": "Methodology",
        "category_code": "stat.ME",
        "authors": "Jianqing Fan, Yingying Fan",
        "first_author": "Jianqing Fan"
    },
    {
        "id": "2301.0031",
        "title": "Asymptotic Normality and Minimax Bounds in Non-Parametric Bayesian Inference",
        "summary": "We study posterior contraction rates for infinite-dimensional parameter spaces under Gaussian process and Dirichlet process priors. We establish Bernstein-von Mises theorems for functional linear models, proving asymptotic normality and coverage of Bayesian credible sets.",
        "category": "Theory",
        "category_code": "stat.TH",
        "authors": "Aad van der Vaart, Subhashis Ghosal",
        "first_author": "Aad van der Vaart"
    },
    {
        "id": "2301.0032",
        "title": "Causal Inference Using Propensity Score Matching in Observational Cohort Studies",
        "summary": "Estimating treatment effects in observational data is prone to confounding bias. We introduce a doubly robust estimator combining generalized additive propensity score matching with outcome regression models. We evaluate finite sample performance using bootstrap resampling.",
        "category": "Applications",
        "category_code": "stat.AP",
        "authors": "Donald B. Rubin, Paul R. Rosenbaum",
        "first_author": "Donald B. Rubin"
    },
    {
        "id": "2301.0033",
        "title": "Markov Chain Monte Carlo Sampling via Hamiltonian Dynamics with Riemannian Geometry",
        "summary": "We propose Riemannian manifold Hamiltonian Monte Carlo (RM-HMC) to sample from target distributions with complex curvature and high correlation. By defining the kinetic energy metric via the Fisher information matrix, the sampler navigates non-linear manifolds efficiently.",
        "category": "Computation",
        "category_code": "stat.CO",
        "authors": "Mark Girolami, Ben Calderhead",
        "first_author": "Mark Girolami"
    },
    {
        "id": "2301.0034",
        "title": "Generalization Error Bounds for Deep Neural Networks via Rademacher Complexity",
        "summary": "We derive sharp generalization bounds for deep neural network classifiers using localized Rademacher complexity and norm-based capacity control. We prove that overparameterized networks generalize well when weight matrices maintain bounded spectral norms across hidden layers.",
        "category": "Machine Learning",
        "category_code": "stat.ML",
        "authors": "Peter L. Bartlett, Dylan J. Foster",
        "first_author": "Peter L. Bartlett"
    },
    {
        "id": "2301.0035",
        "title": "Selective Inference and Post-Selection Confidence Intervals for Lasso Models",
        "summary": "We develop exact post-selection inference methods for linear models after selection by the Lasso. By conditioning on the selection event in polyhedral geometry, we construct valid p-values and confidence intervals that maintain exact nominal coverage conditional on selected active sets.",
        "category": "Methodology",
        "category_code": "stat.ME",
        "authors": "Ryan J. Tibshirani, Jonathan Taylor",
        "first_author": "Ryan J. Tibshirani"
    },

    # Quantitative Biology
    {
        "id": "2301.0040",
        "title": "Highly Accurate Protein Structure Prediction with AlphaFold Deep Learning",
        "summary": "Proteins are essential to life, and understanding their structure can facilitate a mechanistic understanding of their function. We present AlphaFold, an AI system that predicts three-dimensional protein structures with atomic accuracy directly from their primary amino acid sequence using spatial graph networks and evolutionary profile attention.",
        "category": "Biomolecules",
        "category_code": "q-bio.BM",
        "authors": "John Jumper, Richard Evans, Demis Hassabis",
        "first_author": "John Jumper"
    },
    {
        "id": "2301.0041",
        "title": "Single-Cell RNA Sequencing and Cell Lineage Trajectory Inference in Tissue Regeneration",
        "summary": "Single-cell transcriptome profiling enables unbiased identification of cellular heterogeneity. We apply single-cell RNA sequencing (scRNA-seq) to reconstruct dynamic developmental trajectories during stem cell differentiation and neural crest migration using diffusion pseudo-time mapping.",
        "category": "Genomics",
        "category_code": "q-bio.GN",
        "authors": "Aviv Regev, Sarah A. Teichmann",
        "first_author": "Aviv Regev"
    },
    {
        "id": "2301.0042",
        "title": "Spike-Timing-Dependent Plasticity and Synaptic Dynamics in Biological Neural Networks",
        "summary": "We investigate spike-timing-dependent plasticity (STDP) in recurrent cortical neural circuits. We simulate biophysically realistic Hodgkin-Huxley neuron networks to elucidate how dendritic spine remodeling and neurotransmitter diffusion coordinate long-term potentiation and depression.",
        "category": "Neurons and Cognition",
        "category_code": "q-bio.NC",
        "authors": "Wulfram Gerstner, Henry Markram",
        "first_author": "Wulfram Gerstner"
    },
    {
        "id": "2301.0043",
        "title": "Stochastic Modeling of Gene Regulatory Networks and Transcriptional Bursts",
        "summary": "Transcriptional activity in individual cells occurs in episodic bursts. We formulate a continuous-time master equation to model stochastic promoter switching, mRNA synthesis, and enzyme degradation. We validate model predictions against single-molecule fluorescence in situ hybridization data.",
        "category": "Molecular Networks",
        "category_code": "q-bio.MN",
        "authors": "Michael B. Elowitz, Stanislas Leibler",
        "first_author": "Michael B. Elowitz"
    },
    {
        "id": "2301.0044",
        "title": "Metabolic Flux Analysis and Kinetic Modeling in Microbial Synthetic Biology",
        "summary": "We present a constraint-based genome-scale metabolic network reconstruction of engineered Escherichia coli for biofuel synthesis. Using 13C metabolic flux analysis and flux balance optimization, we identify rate-limiting enzymatic pathways and knock-out targets to maximize lipid yield.",
        "category": "Subcellular Processes",
        "category_code": "q-bio.SC",
        "authors": "Bernhard O. Palsson, Daniel Segre",
        "first_author": "Bernhard O. Palsson"
    },
    {
        "id": "2301.0045",
        "title": "Epidemic Spread Dynamics on Complex Heterogeneous Contact Networks",
        "summary": "We develop a compartmental susceptible-infected-recovered (SIR) model on scale-free contact networks with power-law degree distributions. We calculate analytical percolation thresholds for disease transmission and evaluate vaccination strategies across clustered demographic communities.",
        "category": "Populations and Evolution",
        "category_code": "q-bio.PE",
        "authors": "Mark E. J. Newman, Romualdo Pastor-Satorras",
        "first_author": "Mark E. J. Newman"
    },

    # Electrical Engineering
    {
        "id": "2301.0050",
        "title": "Deep Learning for Channel Estimation and Signal Detection in Massive MIMO 5G Systems",
        "summary": "Millimeter-wave massive MIMO systems require accurate channel state information (CSI) under limited feedback bandwidth. We propose a deep neural network architecture for compressed channel estimation and multi-user signal detection. Our approach outperforms conventional linear minimum mean-square error (LMMSE) detectors.",
        "category": "Signal Processing",
        "category_code": "eess.SP",
        "authors": "Hao Ye, Geoffrey Ye Li, Biing-Hwang Juang",
        "first_author": "Hao Ye"
    },
    {
        "id": "2301.0051",
        "title": "End-to-End Deep Learning for High-Resolution Magnetic Resonance Image Reconstruction",
        "summary": "Accelerated magnetic resonance imaging (MRI) reduces scan time by under-sampling k-space data. We introduce a physics-guided deep unrolled neural network that integrates parallel imaging physics into convolutional reconstruction cascades, achieving state-of-the-art structural similarity.",
        "category": "Image and Video Processing",
        "category_code": "eess.IV",
        "authors": "Florian Knoll, Kerstin Hammernik",
        "first_author": "Florian Knoll"
    },
    {
        "id": "2301.0052",
        "title": "Model Predictive Control and Reinforcement Learning for Autonomous Vehicle Trajectories",
        "summary": "We design an integrated model predictive control (MPC) and deep reinforcement learning framework for autonomous vehicle path tracking and emergency collision avoidance under non-linear tire-road friction dynamics. Stability guarantees are proven via Lyapunov barrier certificates.",
        "category": "Systems and Control",
        "category_code": "eess.SY",
        "authors": "Alberto Bemporad, Francesco Borrelli",
        "first_author": "Alberto Bemporad"
    },
    {
        "id": "2301.0053",
        "title": "End-to-End Neural Speech Recognition and Keyword Spotting on Edge Hardware",
        "summary": "We propose a lightweight conformer transducer model for real-time speech recognition on embedded microcontroller platforms. Using 8-bit post-training quantization and pruning, the model achieves low word error rate with sub-10ms inference latency.",
        "category": "Audio and Speech Processing",
        "category_code": "eess.AS",
        "authors": "Anmol Gulati, James Qin, Chung-Cheng Chiu",
        "first_author": "Anmol Gulati"
    },
    {
        "id": "2301.0054",
        "title": "Compressed Sensing Radar Imaging via Sparsity-Aware Optimization",
        "summary": "Synthetic aperture radar (SAR) imaging faces trade-offs between pulse repetition frequency and cross-range resolution. We formulate SAR image formation as a sparse inverse problem solved via alternating direction method of multipliers (ADMM) with total variation regularization.",
        "category": "Signal Processing",
        "category_code": "eess.SP",
        "authors": "Volkan Cevher, Marco Duarte",
        "first_author": "Volkan Cevher"
    },
    {
        "id": "2301.0055",
        "title": "Robust Decentralized Power Grid Frequency Regulation under Renewable Energy Volatility",
        "summary": "High penetration of inverter-based renewable resources reduces power system inertia and increases frequency fluctuations. We propose a distributed H-infinity robust control strategy for battery energy storage systems to maintain grid frequency stability under stochastic wind generation.",
        "category": "Systems and Control",
        "category_code": "eess.SY",
        "authors": "Florian Dorfler, Francesco Bullo",
        "first_author": "Florian Dorfler"
    }
]

# Replicate and synthesize slight variants to create a solid 144-sample dataset for fast training
all_records = []
for repetition in range(4):
    for p in PAPERS_DATA:
        rec = dict(p)
        rec["id"] = f"{p['id']}_{repetition}"
        rec["summary_word_count"] = len(p["summary"].split())
        rec["published_date"] = f"2023-0{repetition+1}-15"
        rec["updated_date"] = f"2023-0{repetition+1}-20"
        all_records.append(rec)

df = pd.DataFrame(all_records)
sample_csv_path = DATA_DIR / "sample_arxiv_metadata.csv"
df.to_csv(sample_csv_path, index=False)
print(f"Generated {len(df)} sample records in {sample_csv_path}")

# Now generate realistic sample PDFs across all 6 domains in sample_papers/
def create_sample_pdf(title, abstract, category, filename):
    pdf_path = SAMPLE_PAPERS_DIR / filename
    doc = fitz.open()
    page = doc.new_page()

    # Layout structured academic paper text
    y = 50
    # Title
    rect_title = fitz.Rect(50, y, 550, y + 45)
    page.insert_textbox(rect_title, title, fontsize=14, fontname="helv", color=(0.1, 0.1, 0.3))
    y += 55

    # Authors & metadata
    authors = f"Research Group in {category} - Scientific Preprint"
    rect_auth = fitz.Rect(50, y, 550, y + 25)
    page.insert_textbox(rect_auth, authors, fontsize=10, fontname="times-italic", color=(0.3, 0.3, 0.3))
    y += 35

    # Abstract heading
    rect_abs_head = fitz.Rect(50, y, 550, y + 20)
    page.insert_textbox(rect_abs_head, "Abstract", fontsize=11, fontname="helv", color=(0, 0, 0))
    y += 25

    # Abstract body
    rect_abs = fitz.Rect(50, y, 550, y + 110)
    page.insert_textbox(rect_abs, abstract, fontsize=9.5, fontname="times-roman", color=(0.1, 0.1, 0.1))
    y += 120

    # Section 1. Introduction
    rect_sec1 = fitz.Rect(50, y, 550, y + 20)
    page.insert_textbox(rect_sec1, "1. Introduction", fontsize=11, fontname="helv", color=(0, 0, 0))
    y += 25

    intro_text = (
        f"In recent years, the scientific domain of {category} has witnessed significant transformations. "
        "Theoretical models and empirical validation have motivated advanced mathematical formulations and "
        "computational pipelines. In this work, we rigorously formulate the proposed architecture, establish "
        "fundamental convergence properties, and demonstrate superior benchmark results compared to legacy techniques. "
        "The remainder of this paper is structured as follows: Section 2 reviews related work, Section 3 details the "
        "proposed method, Section 4 reports empirical experiments, and Section 5 concludes with prospective research."
    )
    rect_intro = fitz.Rect(50, y, 550, y + 100)
    page.insert_textbox(rect_intro, intro_text, fontsize=9.5, fontname="times-roman", color=(0.15, 0.15, 0.15))
    y += 110

    # Section 2. Methodology & Mathematical Formulation
    rect_sec2 = fitz.Rect(50, y, 550, y + 20)
    page.insert_textbox(rect_sec2, "2. Methodology and Formal Analysis", fontsize=11, fontname="helv", color=(0, 0, 0))
    y += 25

    method_text = (
        "Let X denote the observed input space and let Y represent the target variable manifold. "
        "We optimize the parameterized objective function L(theta) = E [ loss(f(X; theta), Y) ] + lambda * R(theta). "
        "Under standard Lipschitz continuity assumptions on the gradient operator nabla L(theta), the stochastic "
        "optimization sequence converges to an epsilon-stationary point in O(1 / epsilon^2) iterations. "
        "Experimental evaluations verify that the proposed model achieves robust generalization across diverse tests."
    )
    rect_method = fitz.Rect(50, y, 550, y + 100)
    page.insert_textbox(rect_method, method_text, fontsize=9.5, fontname="times-roman", color=(0.15, 0.15, 0.15))

    doc.save(pdf_path)
    doc.close()
    print(f"Created sample PDF: {pdf_path.name}")

# Generate 6 sample PDFs, one per primary domain
sample_spec = [
    ("Attention Is All You Need for Scalable Transformer Architectures", PAPERS_DATA[0]["summary"], "Computer Science", "paper_cs_transformers.pdf"),
    ("On the Convergence Rates of Stochastic Gradient Descent in Non-Convex Optimization", PAPERS_DATA[6]["summary"], "Mathematics", "paper_math_optimization.pdf"),
    ("Experimental Realization of Fault-Tolerant Quantum Error Correction Codes", PAPERS_DATA[12]["summary"], "Physics", "paper_physics_quantum.pdf"),
    ("High-Dimensional Covariance Matrix Estimation and False Discovery Rates", PAPERS_DATA[18]["summary"], "Statistics", "paper_stat_covariance.pdf"),
    ("Highly Accurate Protein Structure Prediction with AlphaFold Deep Learning", PAPERS_DATA[24]["summary"], "Quantitative Biology", "paper_qbio_alphafold.pdf"),
    ("Deep Learning for Channel Estimation and Signal Detection in Massive MIMO 5G Systems", PAPERS_DATA[30]["summary"], "Electrical Engineering", "paper_eess_5g_mimo.pdf"),
]

for title, abs_text, cat, fname in sample_spec:
    create_sample_pdf(title, abs_text, cat, fname)

print("Sample data and PDFs created successfully!")
