# Jupyter Notebook Analysis: Isolated Word Recognition using HMM-GMMs

This document provides a detailed analysis of the notebook `Lab3_ASR_cluely.ipynb`, which implements an Automatic Speech Recognition (ASR) system for isolated digit recognition using Hidden Markov Models with Gaussian Mixture Models (HMM-GMMs).

---

## Overview

This notebook demonstrates a complete ASR pipeline for recognizing spoken digits (0-9) using:
- **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction
- **HMM-GMM (Hidden Markov Model with Gaussian Mixture Model)** for acoustic modeling
- **Maximum Likelihood** for classification

---

## Cell-by-Cell Analysis

### Cell 1 (Markdown)
**Functional Summary:** Title cell introducing the notebook's purpose.

**Theoretical Context:**
- **What is happening?** This is a header indicating that the notebook focuses on **Isolated Word Recognition**, meaning each audio sample contains exactly one word (digit) with silence before and after.
- **Why Isolated Word Recognition?** This is simpler than continuous speech recognition because:
  - Word boundaries are known (start/end of audio)
  - No need for word segmentation
  - Ideal for learning ASR fundamentals

---

### Cell 2 (Code - Package Installation)
**Functional Summary:** Installing the `hmmlearn` library, which provides implementations of Hidden Markov Models.

**Theoretical Context:**
- **What is happening?** The notebook uses `!pip install hmmlearn` to install the required library for HMM modeling.
- **Why hmmlearn?** This library provides:
  - Efficient HMM implementations (Gaussian HMM, GMM-HMM)
  - Training algorithms (Baum-Welch/EM algorithm)
  - Scoring/inference methods (Forward algorithm, Viterbi decoding)

---

### Cell 3 (Code - Import)
**Functional Summary:** Importing the HMM module from hmmlearn.

**Theoretical Context:**
- **What is happening?** Importing `hmm` from `hmmlearn` to access HMM model classes.
- **Model/Algorithm Deep Dive:** The `hmmlearn.hmm` module provides:
  - **GaussianHMM**: States emit observations from a Gaussian distribution
  - **GMMHMM**: States emit observations from a Gaussian Mixture Model (more flexible than single Gaussian)
  - These models can capture the temporal dynamics of speech through state transitions

---

### Cell 4 (Code - Training Function Definition)
**Functional Summary:** Defines the `train_GMMHMM` function that creates and trains one HMM-GMM model per digit class.

**Theoretical Context:**

**What is happening?**
1. For each digit (label), a separate HMM-GMM model is created
2. Each model has 5 states and 6 Gaussian mixture components per state
3. The transition matrix enforces a left-to-right topology
4. Models are trained using the Expectation-Maximization (EM) algorithm

**Model/Algorithm Deep Dive:**

**Hidden Markov Model (HMM):**
- **Purpose:** Model temporal sequences with hidden (latent) states
- **Speech Application:** Different parts of a spoken word correspond to different acoustic states
- **Key Components:**
  - **States (5 states):** Represent different phases of pronouncing a digit
    - States 1-3: Main pronunciation phases
    - State 4: Transition/ending state
    - State 5: Final absorbing state
  - **Transition Matrix:** Defines probability of moving from state i to state j
    ```python
    # Left-to-right topology (can only stay or move forward)
    [[1/3, 1/3, 1/3, 0,   0  ],   # State 1 → {1,2,3}
     [0,   1/3, 1/3, 1/3, 0  ],   # State 2 → {2,3,4}
     [0,   0,   1/3, 1/3, 1/3],   # State 3 → {3,4,5}
     [0,   0,   0,   0.5, 0.5],   # State 4 → {4,5}
     [0,   0,   0,   0,   1  ]]   # State 5 → {5} (absorbing)
    ```
  - **Start Probability:** `[0.5, 0.5, 0, 0, 0]` - always start in state 1 or 2

**Gaussian Mixture Model (GMM):**
- **Purpose:** Model the probability distribution of observations (MFCC features) in each state
- **Why GMM instead of single Gaussian?** 
  - Speech features are not unimodal (e.g., different speakers, accents)
  - 6 mixture components can capture multiple "modes" of pronunciation
  - More expressive than single Gaussian, less prone to overfitting than one Gaussian per sample

**Training (EM Algorithm):**
- **Expectation step:** Given current model parameters, compute probability of each state sequence
- **Maximization step:** Update model parameters (transition probs, GMM params) to maximize likelihood
- **Iterations:** `n_iter=10` runs 10 EM iterations

**Covariance Type (`diag`):**
- Assumes diagonal covariance matrices (features are independent)
- Reduces parameters from O(d²) to O(d) where d = 13 MFCC coefficients
- Trade-off: Efficiency vs. modeling feature correlations

**Why Left-to-Right Topology?**
- Speech is inherently temporal and unidirectional (you can't "un-say" something)
- Prevents unrealistic backward transitions
- Represents the natural progression: beginning → middle → end

**Strengths:**
- Well-suited for temporal data like speech
- Probabilistic framework handles variability
- Efficient training and inference

**Limitations:**
- Assumes feature frames are independent given the state (Markov assumption)
- Cannot model long-term dependencies (addressed by modern RNNs/Transformers)
- Requires careful topology design

---

### Cell 5 (Markdown)
**Functional Summary:** Section header for MFCC extraction.

---

### Cell 6 (Raw/Instructions)
**Functional Summary:** Instructions for implementing MFCC extraction with specific parameters.

**Theoretical Context:** Specifies hyperparameters:
- **Hop length:** 25ms (standard for speech; balances time resolution vs. computation)
- **n_fft:** 2048 samples (frequency resolution)
- **n_mfcc:** 13 coefficients (standard; captures phonetic information efficiently)

---

### Cell 7 (Code - MFCC Extraction Function)
**Functional Summary:** Implements MFCC feature extraction from audio files.

**Theoretical Context:**

**What is happening?**
1. Load audio file using `librosa`
2. Convert hop length from seconds to samples: `0.025 * sr`
3. Extract 13 MFCCs using FFT with 2048 samples
4. Transpose to shape `(time_frames, 13)` - each row is one time frame

**Model/Algorithm Deep Dive - MFCC:**

**Purpose:** Convert raw audio waveform into a compact representation that captures phonetic content while discarding speaker-specific and noise characteristics.

**Pipeline:**
1. **Pre-emphasis:** Boost high frequencies (speech has more energy in low frequencies)
2. **Framing:** Divide signal into overlapping frames (25ms windows)
3. **Windowing:** Apply Hamming window to reduce edge effects
4. **FFT:** Convert to frequency domain (2048-point FFT)
5. **Mel Filter Bank:** Apply triangular filters spaced on Mel scale
   - Mel scale mimics human perception (we're better at discriminating low frequencies)
6. **Log:** Take logarithm (human perception is logarithmic)
7. **DCT:** Discrete Cosine Transform to decorrelate coefficients
8. **Select coefficients:** Keep first 13 (most information, less redundancy)

**Why MFCC for Speech?**
- **Compact:** 13 numbers per frame vs. thousands in raw audio
- **Perceptually motivated:** Mel scale matches human hearing
- **Discriminative:** Different phonemes have distinct MFCC patterns
- **Robust:** Log operation reduces sensitivity to volume variations

**Why these parameters?**
- **25ms hop:** Standard for speech; balances time resolution (catch rapid changes) and stationarity assumption (speech is quasi-stationary over short periods)
- **2048 n_fft:** High frequency resolution; captures harmonic structure
- **13 coefficients:** Industry standard; empirically found optimal for speech recognition

**Strengths:**
- Efficient, compact representation
- Works well for phoneme/word recognition
- Decades of proven success in ASR

**Limitations:**
- Lossy (discards phase information)
- Assumes stationarity within frames
- Not optimal for noisy environments (modern systems use learned features)

---

### Cell 8 (Markdown)
**Functional Summary:** Section header for building training data.

---

### Cell 9 (Code - Instructions)
**Functional Summary:** NULL cell with instructions (not executed).

**Theoretical Context:** Outlines data preparation steps:
1. Download dataset from GitHub
2. List all `.wav` files
3. Extract labels from filenames
4. Split 70/30 train/test
5. Save to CSV files

---

### Cell 10 (Raw/Instructions)
**Functional Summary:** Instructions for the `build_data` function.

---

### Cell 11 (Code - Dataset Download)
**Functional Summary:** Clones the Free Spoken Digit Dataset from GitHub.

**Theoretical Context:**
- **What is happening?** Using `git clone` to download a public dataset of spoken digits
- **Dataset:** Contains recordings of digits 0-9 spoken by multiple speakers
- **Why this dataset?** 
  - Free and publicly available
  - Well-structured (filenames encode labels)
  - Ideal for learning HMM-based ASR

---

### Cell 12 (Code - List Audio Files)
**Functional Summary:** Uses `glob` to find all `.wav` files in the dataset directory.

**Theoretical Context:**
- **What is happening?** Creating a list of all audio file paths
- **Pattern matching:** `*.wav` matches all WAV files in the recordings directory
- **Why glob?** Efficient file pattern matching; handles directory traversal

---

### Cell 13 (Code - Train/Test Split)
**Functional Summary:** Splits the dataset into training (70%) and test (30%) sets and saves them to CSV files.

**Theoretical Context:**

**What is happening?**
1. Shuffle file list randomly to avoid bias
2. Calculate split point (70% of total files)
3. Write train files to `train_audio_liste.csv`
4. Write test files to `test_audio_liste.csv`

**Why Train/Test Split?**
- **Training set:** Used to learn model parameters (HMM transition probs, GMM means/covariances)
- **Test set:** Evaluates generalization to unseen data
- **70/30 ratio:** Common split; balances having enough training data vs. reliable test metrics
- **Why shuffle?** Ensures random distribution of speakers/digits across sets; prevents ordering bias

**Why save to CSV?**
- Reproducibility: Same split can be reused
- Separation of concerns: Data preparation separate from model training

---

### Cell 14 (Code - Build Data Function)
**Functional Summary:** Defines `build_data` function that extracts MFCCs for all audio files and organizes them by label.

**Theoretical Context:**

**What is happening?**
1. Initialize a `defaultdict(list)` to store features by label
2. For each audio file:
   - Extract label from filename (e.g., `"0_jackson_0.wav"` → label is `"0"`)
   - Extract MFCCs using `extract_mfcc`
   - Append MFCC array to the corresponding label's list
3. Return dictionary: `{label: [mfcc_array1, mfcc_array2, ...]}`

**Why this structure?**
- **Dictionary by label:** Each digit class gets its own model, so we need features grouped by label
- **List of arrays:** Each utterance has variable length (different speakers say digits at different speeds)
- **Benefits:**
  - Organized for per-class training
  - Handles variable-length sequences
  - Easy to access all examples of a specific digit

**Label Extraction:**
- Filename format: `{digit}_{speaker}_{repetition}.wav`
- `split('_')[0]` extracts the first part (digit)
- **Why from filename?** Dataset convention; labels are encoded in filenames

---

### Cell 15 (Markdown)
**Functional Summary:** Section header for model training.

---

### Cell 16 (Code - Load and Train)
**Functional Summary:** Loads training file paths, builds the dataset, trains HMM-GMM models for all digits, and prints completion messages.

**Theoretical Context:**

**What is happening?**
1. **Load file paths:** Read CSV to get list of training audio files
2. **Build dataset:** Call `build_data` to extract MFCCs and organize by label
3. **Train models:** Call `train_GMMHMM` to create and train 10 models (one per digit)
4. **Output:** Dictionary `hmmModels` with keys `'0'` to `'9'`, values are trained GMMHMM objects

**Why one model per digit?**
- **Discriminative approach:** Each digit has unique acoustic characteristics
- **Classification via scoring:** For unknown audio, compute likelihood under each model; highest likelihood wins
- **Alternative approach:** Single model with 50 states (5 per digit) + forced alignment
  - More complex, requires sequence-level labels
  - This approach is simpler for isolated word recognition

**Training Process (under the hood):**
For each digit's model:
1. **Concatenate all utterances:** `np.vstack(trainData)` creates one big array
2. **Track lengths:** `length` array tells HMM where one utterance ends and next begins
3. **EM algorithm:** 
   - Initialize GMM parameters randomly (or using k-means)
   - Iterate:
     - E-step: Compute state occupancy probabilities using Forward-Backward algorithm
     - M-step: Update GMM means, covariances, transition probs to maximize likelihood
   - Converge after 10 iterations

**Why multiple utterances?**
- Variability: Different speakers, speeds, accents
- Generalization: Model learns average characteristics and variance
- More data → better parameter estimates

**Warnings about n_fft:**
- Some audio files are shorter than 2048 samples
- Librosa pads with zeros (acceptable but increases frequency resolution unnecessarily)
- Not critical for this task

---

### Cell 17 (Markdown)
**Functional Summary:** Section header for evaluation.

---

### Cell 18 (Code - Evaluation)
**Functional Summary:** Loads test data, performs inference using trained models, and calculates recognition accuracy.

**Theoretical Context:**

**What is happening?**
1. **Load test set:** Read CSV and build test dataset (MFCCs organized by true label)
2. **For each true label:**
   - Take first test example: `feature[0]`
   - **Score against all 10 models:** Compute log-likelihood for each model
   - **Predict:** Choose model with highest log-likelihood
   - **Compare:** Check if prediction matches true label
3. **Calculate accuracy:** `score_cnt / total_labels * 100`

**Model/Algorithm Deep Dive - Inference/Scoring:**

**Scoring (`model.score(feature)`):**
- Computes **log-likelihood** of the observation sequence given the model
- Uses **Forward algorithm:**
  - Dynamic programming algorithm
  - Computes probability of observation sequence by summing over all possible state sequences
  - Complexity: O(T × N²) where T = sequence length, N = number of states
  
**Why Log-Likelihood?**
- Probabilities are very small (product of many values < 1)
- Logs convert products to sums: `log(a × b) = log(a) + log(b)`
- Prevents numerical underflow
- Higher (less negative) log-likelihood = better fit

**Classification:**
- **Generative approach:** Model P(X|digit) for each digit
- **Decision rule:** `predicted_digit = argmax_digit P(X|digit)`
- This is equivalent to Maximum Likelihood classification
- **Assumption:** All digits equally likely a priori (could incorporate priors with Bayes' rule)

**Why only test one example per label?**
- Code limitation: `feature[0]` only tests first sample
- **Should be:** Loop over all test examples for each label
- Current accuracy (100%) is likely inflated due to small test set

**Strengths of this approach:**
- Principled probabilistic framework
- No need for explicit feature engineering beyond MFCCs
- Handles variable-length sequences naturally

**Limitations:**
- **Independence assumption:** Assumes MFCC frames are independent given state (not true; consecutive frames are correlated)
- **Limited context:** HMMs have limited ability to model long-range dependencies
- **Generative vs. Discriminative:** Modern systems use discriminative models (DNNs) that directly model P(digit|X)
- **Data efficiency:** Requires substantial training data per class
- **Speaker adaptation:** No mechanism to adapt to new speakers (would need MAP adaptation or more data)

**Why 100% accuracy?**
- Very simple task (isolated digits, 10 classes)
- Clean dataset (studio recordings)
- Limited test set (testing only one sample per digit)
- With full test set, accuracy would be lower (~90-95% typical for GMM-HMM on this dataset)

---

### Cell 19 (Empty Code Cell)
**Functional Summary:** Placeholder cell, no code.

---

## Summary and Key Takeaways

### Pipeline Overview:
1. **Feature Extraction:** Audio → MFCC (compact, perceptually motivated representation)
2. **Modeling:** One HMM-GMM per digit class (temporal dynamics + acoustic variability)
3. **Training:** EM algorithm learns parameters from labeled data
4. **Inference:** Forward algorithm scores test audio against all models; pick best

### Why HMM-GMM for ASR?
- **Temporal modeling:** HMMs capture sequential nature of speech
- **Variability:** GMMs handle speaker/pronunciation variations
- **Statistical framework:** Principled probabilistic inference
- **Industry proven:** Dominated ASR before deep learning (still used in hybrid systems)

### Strengths:
- Works well for clean, isolated word recognition
- Interpretable (states correspond to phone/subphone units)
- Efficient training and inference
- Modest data requirements compared to deep learning

### Limitations and Modern Alternatives:
- **Independence assumption:** Frames are actually correlated (addressed by RNNs)
- **Feature engineering:** MFCCs are hand-crafted (DNNs learn features end-to-end from raw audio)
- **Generative approach:** Modern systems use discriminative models (CTC, attention-based seq2seq)
- **Scalability:** Continuous/large-vocabulary speech needs language models, pronunciation dictionaries
- **Noisy conditions:** Struggles with background noise (modern: robust features, denoising)

### Extensions and Improvements:
1. **Full test set evaluation:** Loop over all test samples
2. **Confusion matrix:** Analyze which digits are confused
3. **Speaker-independent splits:** Ensure train/test have different speakers
4. **Language model:** Add bigram/trigram probs for digit sequences
5. **Delta features:** Add MFCC derivatives (Δ, ΔΔ) to capture dynamics
6. **Advanced topologies:** Parallel paths, skip states for pronunciation variants
7. **Deep learning:** Replace GMMs with DNNs (hybrid DNN-HMM) or use end-to-end models

### Debugging Tips:
- **Low accuracy:** Check that labels are correctly extracted from filenames
- **NaN/Inf errors:** Ensure no silent/zero audio files; check covariance regularization
- **Memory issues:** Reduce n_mix or batch training
- **Convergence:** Increase n_iter, initialize with k-means

### Mathematical Intuition:

**HMM:**
- **Forward probability α_t(i):** Probability of observing x_1...x_t and being in state i at time t
- **Recursion:** `α_t(j) = Σ_i [α_{t-1}(i) × a_{ij}] × b_j(x_t)`
  - Sum over previous states, weighted by transition prob, multiplied by emission prob

**GMM:**
- **Mixture:** `p(x|state) = Σ_k w_k × N(x; μ_k, Σ_k)`
  - Weighted sum of K Gaussians
  - Each Gaussian represents a "mode" of the feature distribution
  
**EM:**
- **E-step:** `γ_t(i) = P(state=i at time t | X, model)`
  - Compute soft assignments of frames to states
- **M-step:** Update parameters to maximize expected log-likelihood
  - Transition probs: `a_{ij} ∝ Σ_t ξ_t(i,j)` (expected state-to-state transitions)
  - GMM params: Weighted MLE using γ_t(i) as weights

