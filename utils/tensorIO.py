import torch
import pickle
import base64

CHUNK_SIZE = 1024

def fromTensor(tensor):
    data = pickle.dumps(tensor.to('cpu'))
    return data
    #data = base64.b64encode(data).decode('utf-8')
    #return data
    #data_chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
    #return data_chunks

def toTensor(data, device='cpu'):
    #data = ''.join(data)
    #data = base64.b64decode(data.encode('utf-8'))
    tensor = pickle.loads(data)
    return tensor.to(device)

def toStateDict(stateDict,device='cpu'):
    for key in stateDict:
        stateDict[key] = toTensor(stateDict[key],device)
    return stateDict

def fromStateDict(stateDict):
    for key in stateDict:
        stateDict[key] = fromTensor(stateDict[key])
    return stateDict
