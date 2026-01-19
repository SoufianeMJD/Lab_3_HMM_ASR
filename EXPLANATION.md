# Lab 3: Isolated Word Recognition using HMM-GMMs - Complete Beginner's Guide

Welcome! This document will walk you through every step of this notebook as if you've never programmed before. Think of this project like teaching a computer to recognize spoken numbers (0-9) - similar to how voice assistants understand "Hey Siri" or "OK Google."

---

## Overview: What Are We Building?

Imagine you say the word "five" into a microphone. This project teaches the computer to:
1. Listen to your voice (audio recording)
2. Break it down into patterns (features)
3. Compare those patterns to what it learned from training examples
4. Guess which digit you said

We're building a **speech recognition system** - a program that converts spoken words into text.

---

### Cell 1 (Markdown)
**Purpose:** This cell provides the title and tells us what this notebook is about.

**Explanation:** This is just a heading - like the title on a book cover. It says "Isolated Word Recognition using HMM-GMMs."

Let's break down what this means:
- **Isolated Word Recognition**: The computer will recognize individual words spoken one at a time (like saying "three" pause "seven" pause "two"), not continuous speech
- **HMM**: Short for "Hidden Markov Model" - this is a mathematical tool that helps the computer understand sequences. Think of it like a flowchart that predicts what comes next
- **GMM**: Short for "Gaussian Mixture Model" - this is a mathematical tool that groups similar sound patterns together, like sorting puzzle pieces by color

Together, HMM-GMM is a specific technique for recognizing speech patterns.

---

### Cell 2 (Code - Installation)
**Purpose:** This cell downloads and installs a special tool we need to build our speech recognition system.

**Explanation:** 

```python
!pip install hmmlearn
```

Think of this like installing an app on your phone. Before you can use Instagram, you need to download it first, right?

Here, we're installing a tool called `hmmlearn`. This is a pre-built library (collection of code) that other programmers created to help us work with Hidden Markov Models.

The `!pip install` command is like going to an app store and saying "download this for me." Once it runs, you'll see output showing the installation progress - just like your phone shows "Installing..." when you download an app.

---

### Cell 3 (Code - Import)
**Purpose:** This cell brings the tool we just installed into our workspace so we can actually use it.

**Explanation:**

```python
from hmmlearn import hmm
```

This is like opening the app after installing it. Installing something doesn't mean it's ready to use - you need to open it first!

In programming terms, `import` means "load this tool into memory so I can use it." Specifically:
- `hmmlearn` is the library we just installed
- `hmm` is a specific part of that library we want to use (the HMM tools)
- `from X import Y` means "from the X toolbox, get me the Y tool"

After this line runs, we can now use HMM functions in our code.

---

### Cell 4 (Code - Training Function)
**Purpose:** This cell defines a "recipe" (function) that teaches the computer to recognize each digit by learning patterns from training examples.

**Explanation:**

This is a big one! Let me break it down step by step.

```python
def train_GMMHMM(dataset):
```

This line says "I'm creating a recipe called `train_GMMHMM`." The recipe takes in one ingredient: `dataset` (which will be our collection of audio recordings).

```python
    GMMHMM_Models = {}
```

This creates an empty container (like an empty toolbox). We'll fill it with 10 trained models - one for each digit (0-9).

```python
    states_num = 5
    GMM_mix_num = 6
```

These are configuration settings - like choosing the difficulty level in a video game. 
- `states_num = 5`: We're saying each word has 5 stages (beginning, middle phases, end)
- `GMM_mix_num = 6`: We're using 6 different sound pattern groups to describe each stage

Think of saying "hello": H-eh-l-l-o has distinct phases. We're doing something similar for digits.

```python
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], 
                              [0, tmp_p, tmp_p, tmp_p , 0], 
                              [0, 0, tmp_p, tmp_p,tmp_p], 
                              [0, 0, 0, 0.5, 0.5], 
                              [0, 0, 0, 0, 1]],dtype=np.float)
```

This is a **transition matrix** - it's like a map showing how likely we are to move from one stage of the word to the next.

Analogy: Imagine climbing stairs. You can go from step 1 to step 2, or step 1 to step 3 (skip one), but you can't jump from step 1 to step 5. This matrix defines those rules for our word stages.

The numbers represent probabilities (chances) of moving between stages. A `0` means "impossible to jump there," and higher numbers mean "more likely to move there."

```python
    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)
```

This defines where we start. It says "when someone starts saying a word, they'll be in stage 1 or stage 2 (50% chance each), but never in the later stages."

```python
    for label in dataset.keys():
```

This starts a loop - "for each digit (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) in our dataset, do the following..."

```python
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, 
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, 
                           covariance_type='diag', n_iter=10)
```

This creates a brand new, untrained model. It's like creating a blank student who's ready to learn. We give it all the configuration settings we defined earlier.

`n_iter=10` means "go through the training examples 10 times to learn the patterns."

```python
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
```

This prepares the training data. Since different people say words at different speeds, each audio recording has a different length. We need to keep track of these lengths.

```python
        trainData = np.vstack(trainData)
```

This stacks all the training examples on top of each other - like stacking pancakes. We're organizing the data into one big pile so the model can learn from all examples at once.

```python
        model.fit(trainData, lengths=length)
```

**This is where the learning happens!** The word `fit` means "train yourself on this data."

Analogy: Imagine showing a child 50 pictures of cats and saying "this is a cat" each time. Eventually, they learn what a cat looks like. Here, we're showing the model many examples of someone saying "zero" and it learns the pattern.

```python
        GMMHMM_Models[label] = model
```

We save the trained model in our toolbox. So `GMMHMM_Models['0']` will be the model that recognizes "zero," `GMMHMM_Models['1']` recognizes "one," etc.

```python
    return GMMHMM_Models
```

Finally, we return the toolbox containing all 10 trained models.

---

### Cell 5 (Markdown)
**Purpose:** This provides a heading for the next section about extracting sound features.

**Explanation:** This is just a section header saying "Extract MFCCs."

**MFCC** stands for "Mel-Frequency Cepstral Coefficients" - don't worry about the complex name! 

Simple explanation: When you speak, your voice creates sound waves. MFCCs are a way to convert those sound waves into numbers that capture the unique characteristics of your voice. It's like taking a fingerprint of the sound.

Why do we need this? Computers can't understand raw audio waves directly. They need numbers they can do math with. MFCCs give us those numbers.

---

### Cell 6 (Raw - Instructions)
**Purpose:** This cell gives instructions for what needs to be done in the next code cell.

**Explanation:** This isn't code that runs - it's a note to the programmer. It says:

"Create a function that extracts MFCCs from an audio file using these specific settings:
- hop length = 0.025 seconds (how often to take a snapshot of the audio)
- number of fft = 2048 (how detailed each snapshot is)
- number of mfcc = 13 (how many number patterns to extract)"

These are like camera settings. Hop length is like frames per second in a video, and the other settings control the quality and detail of the audio analysis.

---

### Cell 7 (Code - Function Template)
**Purpose:** This cell is a template (skeleton) of a function that needs to be completed - it's currently incomplete and won't work.

**Explanation:**

```python
def extract_mfcc(full_audio_path):
    ...
    return mfcc_features
```

This is like a recipe with missing steps. It says:
- "I'm creating a function called `extract_mfcc`"
- "It takes one input: `full_audio_path` (the location of an audio file on your computer)"
- "The `...` means 'someone needs to fill in the code here'"
- "Eventually, it should return `mfcc_features` (the numbers representing the audio)"

This is an exercise left for the programmer to complete based on the instructions in Cell 6.

---

### Cell 8 (Markdown)
**Purpose:** This provides a heading for the section about preparing training data.

**Explanation:** Just another section title: "Build the training data."

This section will be about organizing our audio files so they're ready for training the models.

---

### Cell 9 (Code - Instructions)
**Purpose:** This cell gives step-by-step instructions for what needs to be done to prepare the dataset.

**Explanation:** These are instructions (not executable code) that say:

1. **"Download digit dataset from GitHub"**: First, you need to download a collection of audio recordings of people saying digits 0-9. This is your training data - like a textbook for the computer to learn from.

2. **"List all wave files"**: Once downloaded, scan through all the audio files (they have a `.wav` extension).

3. **"Extract label from file name"**: Each audio file is named something like `zero_jackson_0.wav` or `five_sarah_12.wav`. The first part tells you which digit it is. You need to extract that.

4. **"Split this list to train (70%) and test (30%)"**: Divide your data into two groups:
   - 70% for training (teaching the model)
   - 30% for testing (checking if it learned correctly)
   
   Why? If you only test on examples the model has seen before, you don't know if it truly learned or just memorized. Testing on new examples proves it learned the pattern.

5. **"Save train to train_audio_liste.csv and test to test_audio_liste.csv"**: Save these two lists as CSV files (like Excel spreadsheets) for easy loading later.

---

### Cell 10 (Raw - Instructions)
**Purpose:** This cell explains what the next function should do.

**Explanation:** This says: "Create a function that goes through the dataset and groups the audio files by label."

The function should return a **dictionary** - think of it like a filing cabinet where:
- Each drawer is labeled with a digit (0, 1, 2, ..., 9)
- Inside each drawer are all the MFCC features for that digit

For example:
- Drawer "0" contains MFCCs for all recordings of people saying "zero"
- Drawer "1" contains MFCCs for all recordings of people saying "one"
- And so on...

---

### Cell 11 (Code - Function Template)
**Purpose:** This is another incomplete function template that needs to be filled in.

**Explanation:**

```python
def build_data(fileliste):
    ...
    return dataset
```

This is the skeleton for the function described in Cell 10. It says:
- "I'm creating a function called `build_data`"
- "It takes one input: `fileliste` (a CSV file containing paths to audio files)"
- "The `...` is where you need to write the code"
- "It should return `dataset` (the dictionary/filing cabinet of organized MFCC features)"

When completed, this function would:
1. Read the CSV file
2. For each audio file, extract its MFCCs using the `extract_mfcc` function
3. Group those MFCCs by digit label
4. Return the organized dictionary

---

### Cell 12 (Markdown)
**Purpose:** This provides a heading for the training section.

**Explanation:** Section title: "Train the GMM-HMM model."

This is where we'll actually train our 10 models (one for each digit).

---

### Cell 13 (Code - Training Execution)
**Purpose:** This cell actually runs the training process using all the functions we've defined.

**Explanation:**

```python
trainList = './train_audio_liste.csv'
```

This says "the training data list is located at `./train_audio_liste.csv`" (the `./` means "in the current folder").

```python
trainDataSet = build_data(trainList)
```

This calls the `build_data` function we defined earlier. It:
- Reads the CSV file
- Extracts MFCCs from all training audio files
- Organizes them by digit
- Stores the result in `trainDataSet`

```python
print("Finish prepare the training data")
```

This displays a message saying "I'm done organizing the data!" It's like a progress update.

```python
hmmModels = train_GMMHMM(trainDataSet)
```

This is where the magic happens! It calls the `train_GMMHMM` function we saw in Cell 4. Remember, that function:
- Creates 10 models (one per digit)
- Trains each model on its respective digit recordings
- Returns the toolbox of trained models

We store this toolbox in `hmmModels`.

```python
print("Finish training of the GMM_HMM models for digits 0-9")
```

Another progress message: "Training complete! All 10 models are ready!"

After this cell runs, the computer has learned what each digit sounds like.

---

### Cell 14 (Markdown)
**Purpose:** This provides a heading for the evaluation section.

**Explanation:** Section title: "Evaluation."

Now that we've trained our models, we need to test them. This is like taking a final exam after studying.

---

### Cell 15 (Code - Testing/Evaluation)
**Purpose:** This cell tests our trained models on new audio recordings to see how accurate they are.

**Explanation:**

```python
testList = './test_audio_liste.csv'
```

This points to the test data (the 30% we set aside earlier). These are audio files the models have never seen before.

```python
testDataSet = build_data(testList)
```

We extract MFCCs from all the test audio files and organize them by digit, just like we did for training data.

```python
score_cnt = 0
```

This creates a counter starting at 0. We'll use this to count how many predictions are correct.

```python
for label in testDataSet.keys():
```

This starts a loop: "For each digit in the test dataset (0, 1, 2, ..., 9), do the following..."

```python
    feature = testDataSet[label]
```

Get the MFCC features for this specific digit. For example, if `label` is "3", we get all the test recordings of people saying "three."

```python
    scoreList = {}
```

Create an empty scorecard. We'll fill this with scores from each model.

```python
    for model_label in hmmModels.keys():
        model = hmmModels[model_label]
        score = model.score(feature[0])
        scoreList[model_label] = score
```

This is the prediction process! For each of our 10 trained models:
- Get the model (e.g., the "zero" model, the "one" model, etc.)
- Ask it: "How well does this test audio match your learned pattern?" 
- The answer is a score (higher score = better match)
- Save the score in the scorecard

Think of it like 10 experts each giving their opinion. The "zero" expert says "This sounds 80% like zero to me." The "five" expert says "This sounds 95% like five to me." And so on.

```python
    predict = max(scoreList, key=scoreList.get)
```

Find which model gave the highest score. That's our prediction!

If the "five" model gave the highest score, we predict the person said "five."

```python
    print("Test on true label ", label, ": predict result label is ", predict)
```

Print the results: "The actual digit was X, but I predicted Y."

```python
    if predict == label:
        score_cnt+=1
```

If our prediction was correct (prediction matches the actual label), add 1 to our correct counter.

```python
print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")
```

Finally, calculate and display the accuracy:
- `score_cnt` = number of correct predictions
- `len(testDataSet.keys())` = total number of digits (10)
- The formula gives us the percentage of correct predictions

For example, if we got 8 out of 10 correct, it would print "Final recognition rate is 80.00%"

---

## Summary: The Big Picture

Let's recap the entire flow:

1. **Install tools** (Cell 2-3): Get the software we need
2. **Define training recipe** (Cell 4): Create instructions for how to train models
3. **Define feature extraction** (Cell 6-7): Create instructions for converting audio to numbers
4. **Prepare dataset** (Cell 8-11): Organize audio files for training and testing
5. **Train models** (Cell 12-13): Teach 10 models to recognize each digit
6. **Test models** (Cell 14-15): See how accurate they are on new recordings

The workflow is like teaching a child:
- First, you gather teaching materials (dataset)
- Then, you teach through examples (training)
- Finally, you quiz them on new examples to see if they truly learned (testing)

This entire notebook creates a speech recognition system that can identify spoken digits with measurable accuracy!
