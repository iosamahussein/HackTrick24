import requests
import numpy as np

from LSBSteg import encode
from Solvers.riddle_solvers import riddle_solvers

api_base_url = "http://3.70.97.142:5000"
team_id='Kps2iU3'

def init_fox(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    url = f"{api_base_url}/fox/start"
    data = {"teamId": team_id}
    response = requests.post(url, json = data , headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise Exception(f"Fox API request failed: {response.text}")
    response_data = response.json()
    return response_data['msg'], np.array(response_data['carrier_image'])


def generate_message_array(message, image_carrier):
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier
    '''
    return encode(image_carrier , message)


def get_riddle(team_id, riddle_id):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that:
        1. Once you requested a riddle you cannot request it again per game.
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle.
    '''
    url = f"{api_base_url}/fox/get-riddle"
    data = {"teamId": team_id, "riddleId": riddle_id}
    response = requests.post(url, json= data  ,headers={"Content-Type": "application/json"} )
    if response.status_code >= 400:
        raise Exception(f"Fox API request failed: {response.text}")
    response_data = response.json()
    return response_data['test_case']  # Access the 'test case'

def solve_riddle(team_id, solution):
    '''
    In this function you will solve the riddle that you have requested.
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    '''
    url = f"{api_base_url}/fox/solve-riddle"
    data = {"teamId": team_id, "solution": solution}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise Exception(f"Fox API request failed: {response.text}")
    response_data = response.json()
    return response_data['budget_increase'], response_data['total_budget'], response_data['status']

def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    '''
    Use this function to call the api end point to send one chunk of the message.
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call.
    '''
    url = f"{api_base_url}/fox/send-message"
    data = {"teamId": team_id, "messages": messages, "message_entities": message_entities}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise Exception(f"Fox API request failed: {response.text}")
    response_data = response.json()
    return response_data['status']


def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    '''
    url = f"{api_base_url}/fox/end-game"
    data = {"teamId": team_id}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise Exception(f"Fox API request failed: {response.text}")
    print (response.text)




# saved returns for all riddles in case of any error
return_solved = {
    'ml_easy' : [],
    'ml_medium' : 0,
    'cv_easy': [],
    'cv_medium' : [],
    'cv_hard': 0,
    'sec_medium_stegano' : '',
    'sec_hard' : '',
    'problem_solving_easy' : [],
    'problem_solving_medium' : '',
    'problem_solving_hard' : 0
}



if __name__=='__main__' : 

    # define riddle to solve during game
    riddles = ['cv_easy',
            'cv_medium',
            'cv_hard',
            'sec_medium_stegano',
            'sec_hard',
            'ml_easy',
            'ml_medium',
            'problem_solving_easy',
            'problem_solving_medium',
            'problem_solving_hard'
            ]
    
    # start game
    msg , carrier_image = init_fox(team_id)
    

    for riddle in riddles:
        testcase = get_riddle(team_id, riddle)
        try :
            budget_increase , total_budget, status = solve_riddle(team_id, riddle_solvers[riddle](testcase))
        except Exception as e :
            budget_increase , total_budget, status = solve_riddle(team_id,return_solved[riddle])


    chunks = [slice(0, 7) , slice(7, 13) , slice(13, 20)]
    ls = [ ['F' , 'F' , 'R']  , ['R' , 'F' , 'F'] , ['F' , 'R' , 'F'] ]
    r_img =  generate_message_array( msg[chunks[0]] , carrier_image)
    res = send_message( team_id ,[ r_img , r_img , r_img ] , ls[0])
    r_img =  generate_message_array( msg[chunks[1]] , carrier_image)
    res = send_message( team_id ,[ r_img , r_img , r_img ] , ls[1])
    r_img =  generate_message_array( msg[chunks[2]] , carrier_image)
    res = send_message( team_id ,[ r_img , r_img , r_img ] , ls[2])

    end_fox(team_id)
