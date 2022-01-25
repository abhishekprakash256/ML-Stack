import torch

'''
making the alter data 
'''

def data_set(num):
    data = (torch.rand(num) <.5).int()

    return(data)

def parallel_data(array):

    parallel_datasets = []

    for i in range(len(array)):
        new_data = torch.cat((array[0:i],array[i+1:]),0)
        parallel_datasets.append(new_data)
    
    #new_data = torch.cat((array[0:1],array[2:]),0)

    return parallel_datasets

one_data = (data_set(500))

parallel_db = (parallel_data(one_data))

def query(db):
    return db.sum()
full_db_result = query(one_data)

max_distance = 0

for j in parallel_db:
    pd_result = query(j)

    db_distance = torch.abs(pd_result - full_db_result)

    if( db_distance > max_distance):
        max_distance = db_distance
    
print(max_distance)