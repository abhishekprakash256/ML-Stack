#the imports
import torch

num = 10000

def create_db(num):
    db = ((torch.rand(num)) > 0.5).float()

    return db


#creating a coin fliip
first_coin_flip = ((torch.rand(num))> 0.5).float()

second_coin_flip = ((torch.rand(num))> 0.5).float()

data = create_db(num)

#creating the augmented data base

augmented_db = data*first_coin_flip + (1-first_coin_flip) * second_coin_flip

#print(data)

#print(augmented_db)

print("original data",data.mean())
print("augmented data",augmented_db.mean())

print("operation on augmented data", augmented_db.mean() * 2 - 0.5 )

print("mean of new data" ,create_db(num).mean())