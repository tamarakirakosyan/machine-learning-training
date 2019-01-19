import torch
from Models import Simple_FC
from Load import load_data_train

from tqdm import tqdm


EPOCHS = 10
Learning_rate = 0.001
Batch_size = 64
L2_rate = 1e-5

data_size = 4076
num_batches = data_size//Batch_size
num_classes = 5

Train_path = "C:/Users/DELL/Documents/Tamara/2018 GIT ML course/3 Exams/3 Exam/Flowers Ardalan/data/train"

model = Simple_FC()

model.load_state_dict(torch.load("trained/Simple_FC_9.model"))

data_loader = load_data_train(Train_path, Batch_size)


def train():
    model.train()

    crossentropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=L2_rate)

    for epoch in range(EPOCHS):    
        epoch_loss = 0
        epoch_acc = 0
        for X, y in tqdm(data_loader):
            X = X.view(-1, 3*64*64)

            optimizer.zero_grad()

            out = model(X)

            loss = crossentropy(out, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() 

            predictions = torch.argmax(out, 1)
            epoch_acc += torch.sum(predictions==y).item()

        epoch_loss = epoch_loss/num_batches
        epoch_acc = epoch_acc/data_size
        print(f"Epoch {epoch}:")
        print("ACC:", epoch_acc, "LOSS:", epoch_loss)


        torch.save(model.state_dict(), f"trained/Simple_FC_{epoch}.model")



if __name__ == "__main__": #if we are runnign this file directly, run this
    train()

