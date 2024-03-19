import requests
import numpy as np
from LSBSteg import decode

import torch 
from torch import nn

from configs import DEVICE
from model import CNNNetwork


api_base_url = "http://3.70.97.142:5000"
team_id="Kps2iU3"


def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    url = f"{api_base_url}/eagle/start"
    data = {"teamId": team_id}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise Exception(f"Eagle API request failed: {response.text}")
    response_data = response.json()
    return response_data['footprint']

def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''
    pass

def skip_msg(team_id):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    url = f"{api_base_url}/eagle/skip-message"
    data = {"teamId": team_id}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.text == "End of message reached" :
        return None
    if response.status_code >= 400:
        raise Exception(f"Eagle API request failed: {response.text}")
    response_data = response.json()
    return response_data['nextFootprint']
  
def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    url = f"{api_base_url}/eagle/request-message"
    data = {"teamId": team_id , "channelId" : channel_id}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.text == "End of message reached" :
        return None
    if response.status_code >= 400:
        raise Exception(f"Eagle API request failed: {response.text}")
    response_data = response.json()
    return response_data['encodedMsg']

def submit_msg(team_id, decoded_msg):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    url = f"{api_base_url}/eagle/submit-message"
    data = {"teamId": team_id , "decodedMsg" : decoded_msg }
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.text == "End of message reached" :
        return None
    if response.status_code >= 400:
        raise Exception(f"Eagle API request failed: {response.text}")
    response_data = response.json()
    return response_data['nextFootprint']

  
def end_eagle(team_id):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    url = f"{api_base_url}/eagle/end-game"
    data = {"teamId": team_id}
    response = requests.post(url, json= data , headers={"Content-Type": "application/json"})
    if response.status_code >= 400:
        raise Exception(f"Eagle API request failed: {response.text}")
    return response.text


def load_model(ckpt_pth : str) -> torch.nn.Module:
    '''
    Load the model from the checkpoint path
    '''
    model = CNNNetwork()
    model.load_state_dict(torch.load(ckpt_pth))
    model.to(DEVICE)
    return model

                                     
def submit_eagle_attempt(team_id):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    footprints = init_eagle(team_id)

    while footprints is not None :
        footprints = [footprints['1'] , footprints['2'] , footprints['3']]
        footprints = np.array(footprints).astype(np.float32)
        footprints[np.isinf(footprints)] = 1e5

        m1 , m2 , m3 =  np.mean(footprints, axis=1).mean(axis=1)

        if m1 > 20 and m1 < 50:
            encoded_msg = request_msg(team_id, 1)
            encoded_msg = np.array(encoded_msg)
            decoded_msg = decode(encoded_msg)
            footprints = submit_msg(team_id, decoded_msg)

        elif m2 > 20 and m2 < 50 :
            encoded_msg = request_msg(team_id, 2)
            encoded_msg = np.array(encoded_msg)
            decoded_msg = decode(encoded_msg)
            footprints = submit_msg(team_id, decoded_msg)

        elif m3 > 20 and m3 < 50 :
            encoded_msg = request_msg(team_id, 3)
            encoded_msg = np.array(encoded_msg)
            decoded_msg = decode(encoded_msg)
            footprints = submit_msg(team_id, decoded_msg)

        else :
            footprints = skip_msg(team_id)

    print(end_eagle(team_id))


if __name__ == "__main__":
    submit_eagle_attempt(team_id)
