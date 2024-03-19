# Hacktrick2024

In this repo, our team `Spaceship in the multiverse` expalins the approach we used to succefully secure first place Students Track in Dell Hacktrick2024.
---
# Table of Contents
- [Hacktrick2024 problem statement](#problem_statement)
- [Our Approach](#approach)
    - [Fox Solver](#fox)
    - [Riddles](#riddles)
        - [Security Medium](#sec_med)
        - [Security Hard](#sec_hard)
        - [Computer Vision Easy](#cv_easy)
        - [Computer Vision Medium](#cv_med)
        - [Computer Vision Hard](#cv_hard)
        - [Machine Learning Easy](#ml_easy)
        - [Machine Learning Medium](#ml_med)
        - [Problem Solving Easy](#ps_easy)
        - [Problem Solving Medium](#ps_med)
        - [Problem Solving Hard](#ps_hard)

    - [Eagle Solver](#eagle)
- [Contributors](#contributor)
---
<a id='problem_statement'></a>
# Hacktrick2024 problem statement
```
Contestants get to compete in teams against each other, taking one of two roles at a
time but ultimately playing on both sides. One trying to send a secret message, and the other trying hard to intercept it.
```
```
Firstly: The Fox. Mischievous, and sly, the Fox uses all its tactics to fool the Eagle and try to send the message through to the parrot using steganography. As the Fox, you’ll have the opportunity to invest your time wisely into honing your skills and creating distractions to increase your chances of evading the Eagle’s watchful gaze.
```
```
Second: The Eagle. Sharp-eyed and vigilant, the Eagle uses its attentiveness to try to
intercept and decode the messages sent without getting fooled. Beware of the Fox’s devious tricks, for Fake messages may cross your path. Your mission is to distinguish truth from deception, ensuring that only genuine messages are intercepted while avoiding costly mistakes.
```






<a id='approach'></a>
# Our Approach
Our solution prioritized efficiency and deception in both the Fox and Eagle roles. As the Fox, we focused on solving the fastest riddles to earn a budget of fake messages.  We strategically sent a mix of 3 real and 6 fake messages across 3 channels to confuse the Eagle. As the Eagle, we aimed to quickly analyze footprints and identify real messages, minimizing the time spent on misleading fake messages.

---
<a id='fox'></a>
### Fox Solver
Our Fox strategy emphasized speed and deception. We made the calculated decision to skip the time-consuming Computer Vision riddles, instead prioritizing the quickest riddles to solve. This allowed us to rapidly gain a budget of fake messages. We then strategically deployed a mix of 3 real messages and 6 fake messages across all 3 channels.  Our aim was to maximize confusion for the Eagle, hindering their ability to quickly identify the genuine messages.


---
<a id='riddles'></a>
### Riddles
<a id='sec_med'></a>
#### Security Medium
```
Your task is: given an image containing a hidden message, your task is to uncover this secret message. But remember, things are not always as they seem. The key to solving this riddle lies in the hidden layers of the image, where noise is not just noise.
```
> **Our Solution**: Transforming input list into `torch.Tensor`, then decoding it using steganoGAN decoder.
---
<a id='sec_hard'></a>
#### Security Hard
```
In the kingdom of Cypheria, a message of great importance must be encoded to safeguard it from prying eyes. The message, containing the location of an ancient treasure, must be encrypted using the Data Encryption Standard (DES) algorithm. Your task is: given a pair of key and plain text, you should encrypt the plain text using the DES algorithm.
```
> **Our Solution**: We developed efficient impementation of DES algorithm to decipher input text.
---
<a id='cv_easy'></a>
#### Computer Vision Easy
```
A very well-known phrase used by schoolchildren to explain their failure to turn in an assignment is “the dog ate my homework”. But unfortunately, your dog really did shred an important image that you need to turn in, and the dog shredded that image
vertically.

Luckily, you were holding the first shred, so you know the first left most shred is in place. Also, you were extremely lucky to have the shreds of equal width which is 64 pixels.

Your task is: you should write an algorithm to reassemble the paper and solve this puzzle.
```
> **Our Solution**: We developed a similiraity function to calculate similirity between all shreds. Knowing the first shred, we sorted other shreds based on their similiarity score.

------

<a id='cv_med'></a>
#### Computer Vision Medium
```
Having an image with high quality is important for multiple reasons, including better analysis and interpretation, effective image processing, improved machine learning models, and accurate diagnosis and decision making.

Some images may contain patches, and the removal of patches is important when they contain unwanted defects or noise that can affect the analysis or visual quality of an image.

Your task is: given a patched image, you should develop an algorithm that can identify
and remove a small, patched image from a larger image, and then interpolate the
missing pixels.
```
> **Our Solution**: First, finding location of emplaced patch using **SIFT** algorithm, impaint this region.
---

<a id='cv_hard'></a>
#### Computer Vision Hard
```
This CV riddle is about using AI to answer questions about a given image. 
You would be given a question and an image then you are asked to find the answer for the question from the provided image (Note you are bounded by a timeout so human-aided solutions are not possible).
```
> **Our Solution**: Use pretrained Visual Question Answering (VQA) model to answer the question.
---

<a id='ml_easy'></a>
#### Machine Learning Easy
```
Your task is: developing a machine/deep learning model that can accurately forecast the number of attacks on a specific website based on historical time series data that is given in the received materials. This involves preprocessing the data and designing a solution that can handle the inherent unpredictability and seasonality of attacks.

Given a dataset of attacks data with daily intervals for 500 days, the task is to fore-
cast the attacks for the next 50 days.
```
> **Our Solution**: Use Auto-Regression algorithm to predict next 50 timestamps.
---


<a id='ml_med'></a>
#### Machine Learning Medium
```
Your task is: detecting connected components using machine learning algorithms, you have a complex connected shape surrounded by groups of points. 
Machine Learning algorithms such as classification, clustering is helpful to map points based on their features to either belonging to the shape or not. You need to use your skills to map the point to the correct group. Your training data can be found in the received materials,
and consists of two features representing the x and y coordinates, and the label that has values either 0 for connected shape, and -1 for a point that doesn’t belong to the shape.
```
> **Our Solution**: Use prefered to use simple approach to calculate distance between input point and all other points in connected component and if minDistance less then a predefined threshold, then point belongs to graph.
---


<a id='ps_easy'></a>
#### Problem Solving Easy
```
You are assisting Fatima, a linguist analyzing ancient Egyptian texts. Your task is to help her identify the most frequently used words in her collection.  Specifically, given a list of words and a number (X), you need to:
1. Find the X most frequent words.
2. Arrange them in decreasing order of frequency.
3. If multiple words have the same frequency, sort them alphabetically.
```
> **Our Solution**:
The provided Python solution follows a clear step-by-step approach:
> 1. Count Word Frequencies: The code uses a dictionary (mp) to store how many times each word appears in the input list.
> 2. Group by Frequency:  Words are grouped in a dictionary (frq) based on how often they occur. For instance, all words appearing twice would be grouped together.
> 3. Sort and Build Output:
    - The groups are arranged with the most frequent words coming first (descending order).
    - Within each group, the words are sorted alphabetically.
    - The final output list (ans) is built, taking the top words from each group until the desired count (X) is reached.
---

<a id='ps_med'></a>
#### Problem Solving Medium
```
Imagine you're a skilled cryptographer tasked with deciphering a secret language used for encoded messages. This language follows a unique rule:
Encoding Rule: Given an integer x and a string w, the encoded message is constructed as x[w]. This means the string w is repeated x times, creating a more complex sequence.
Your mission is to write a decoder that takes an encoded string and returns the original, decoded message.

```
> **Our Solution**:
The provided Python solution tackles the nested repetitions within the encoded message using a stack-based approach:
> 1. Initialization: Three empty lists are created: counts to store repetition counts, strings to hold partial decoded strings during nested decoding, and out_str to accumulate the final decoded message.
> 2. Iterating through Input: Each character in the encoded string is processed:
>       * Digits: If encountering a digit, it's appended to the current repetition count (count).
>       * '[' (Left Square Bracket): This marks the beginning of a nested repetition. The current count (count) and partial string (out_str) are pushed onto their respective stacks (counts and strings). Both are then reset for the new nested sequence.
>       * ']' (Right Square Bracket): This signifies the end of a nested repetition. We pop the count (count) and partial string (tmp_str) from their stacks. The partial string is then repeated according to the count (count), resulting in a new decoded segment. This segment is appended to the out_str.
>       * Letters: Any lowercase letter encountered gets directly added to the out_str.
> 3. Return Decoded String: After processing all characters, the final out_str contains the decoded message, which is then returned.
---

<a id='ps_hard'></a>
#### Problem Solving Hard
```
A renowned detective with a quirky movement constraint (only moving south or east) needs to navigate a city grid to reach a crime scene. Your task is to calculate the total number of unique paths the detective can take to reach the bottom-right corner of the grid, starting from the top-left corner.
```
> **Our Solution**:
The solution provided uses a dynamic programming approach:
> 1. Creating a Table: A table (or 2D array) is created to store the number of possible paths to reach each cell in the grid. The initial cell (0, 0) is set to 1 (there's only one way to reach the starting point).
> 2. Calculating Paths Iteratively: The table is filled by iterating through each cell. The number of paths to reach a cell is the sum of:
>     - Paths to reach the cell above (if it exists).
>     - Paths to reach the cell to the left (if it exists).
> 3. Final Result:  The bottom-right cell of the table holds the total number of unique paths for the given grid dimensions.
---

<a id='eagle'></a>
### Eagle Solver
Our Eagle implementation offers two distinct approaches for footprint analysis, providing a balance between accuracy and speed. We prioritized swift decision-making and selected Approach 2 for our solution.

#### Approach 1: Deep Learning Model

* Focus: Maximizes accuracy in classifying real and fake messages.
* Technique:
	- Preporcessing: apply logarithmic transformation to footprints, replace all infinity values with 1e5. 
	- Employs a deep learning model that we trained specifically with the provided data.
* Advantage: Provides a higher degree of accuracy in detecting real messages.
* Trade-off: Slower analysis compared to the rapid filtering approach.

#### Approach 2: Distribution Modeling approach (Our Chosen Approach)
After observing some test cases, we noticed that footprints follows three main  distributions, by labeling samples from each distribution using our model we found out: 
* each distribution belongs to one of our three classes (Empty, Fake and Real messages) where: 
	* Empty messages follows standard distribution with mean around zero and standaard deviation around 1.
	* Real meesages follows distribution with `20 < mean < 30`.
	* Fake meesages footprints have higher values than other two distributions.

* Focus: Prioritizes latency in decision-making.
* Technique: Calculates the statistics of each footprint. Footprints with mean values between 20 and 50 are labeled as potentially containing real messages.
* Advantage: Allows for very fast identification of potential real messages.
* Trade-off: Increased risk of misclassification compared to the deep learning model approach.


<a id='contributor'></a>
# Contributors: 
* Osama Hussein
* Mahmoud Elhusseni
* Karim Nady
* Mazen Elnahla
* Mahmoud Hosam 
