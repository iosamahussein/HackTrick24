# Add the necessary imports here
import pandas as pd
import torch
from utils import *
import numpy as np
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from SteganoGAN.utils import decode
from PIL import Image
from transformers import pipeline
vqa_pipeline = pipeline("visual-question-answering")
from sec_hard import sec_hard
from cv_medium import cv_medium


def solve_cv_easy(test_case: tuple) -> list:
    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    shreds = [shredded_image[:, i:i+shred_width] for i in range(0, shredded_image.shape[1], shred_width)]
    last_shred = shreds[0]
    vis = [0]*len(shreds)
    vis[0] = 1 
    ans = [0]
    for _ in range(len(shreds)-1):
        best_match = None
        best_score = -1


        for i, shred in enumerate(shreds):
            if vis[i] == 1:
                continue
            score = np.sum(((last_shred[:, -1] - shred[:, 0])==0))
            if score > best_score:
                best_match = i
                best_score = score 

        vis[best_match] = 1
        ans.append(best_match)
        last_shred = shreds[best_match]
    return ans

def solve_cv_medium(input: tuple) -> list:
    combined_image_array , patch_image_array = input
    combined_image = np.array(combined_image_array,dtype=np.uint8)
    patch_image = np.array(patch_image_array,dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    return cv_medium(combined_image, patch_image)


def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = input
    image = np.array(image,dtype=np.uint8)
    image = Image.fromarray(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """

    return int(vqa_pipeline(image, extracted_question, top_k=1)[0]['answer'])


vqa_pipeline = pipeline("visual-question-answering")


def solve_ml_easy(data: list) -> list:
    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    data = pd.DataFrame(data).values[:, 1].astype(np.float32)
    model = AutoReg(data, lags=10)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(data), end=len(data)+49, dynamic=False).tolist()
    return predictions

# preprocessing for ml_medium
# df = pd.read_csv('../Riddles/ml_medium_dataset/MlMediumTrainingData.csv')
df = pd.read_csv('/content/HackTrick/Riddles/ml_medium_dataset/MlMediumTrainingData.csv')

data = df[df['class']== 0].values[:, :2].tolist()


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    input = np.array(input).reshape(-1, 2)
    print (np.sum((input-data)**2, axis=-1).min())
    return -1 if (np.sum((input-data)**2, axis=-1).min() >= 2) else 0

def solve_sec_medium(input: torch.Tensor) -> str:
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    return decode(torch.Tensor(input))

def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """
    key , pt = input
    return sec_hard(key , pt)

def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    words , num = input
    mp = {}
    frq = {}
    ans = []

    for i in words:
        if mp.get(i, 0)==0:
            mp[i] = 0
        mp[i] += 1
  
    for first, second in mp.items():
        if frq.get(second, 0)==0:
            frq[second] = []
        frq[second].append(first)
    
    # sort the frq list
    frq = sorted(frq.items(), key=lambda x: x[0], reverse=True)
    for freq, freq_list in frq:
        freq_list.sort()
        for item in freq_list:
            if num == 0:
                return ans
            ans.append(item)
            num -= 1
   
    return ans


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    counts = []
    strings = []
    count = 0
    out_str = ""
    for c in input:  # 3[d1[e2[l]]]
        if '1' <= c <= '9':
            count = count * 10 + int(c)
        elif c == '[':
            counts.append(count)
            count = 0
            strings.append(out_str)
            out_str = ""
        elif c == ']':
            tmp = counts.pop()
            tmp_str = strings.pop()
            out_str = tmp_str + out_str * tmp
        else:
            out_str += c

    return out_str

# preprocessing for problem_solving_hard
n = 111
m = 111
table = [[0 for j in range(n)] for i in range(m)]
table[0][0] = 1
for i in range (n):
    for j in range(m):
        if i > 0 :
            table[i][j] +=  table[i-1][j]
        if j > 0 :
            table[i][j] += table[i][j - 1]

def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    x , y = input
    return table[x-1][y-1]


riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_medium_stegano': solve_sec_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}
