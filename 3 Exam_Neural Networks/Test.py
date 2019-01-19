import torch
from Models import Simple_FC
from Load import load_data_test
from sklearn.metrics import classification_report

from tqdm import tqdm

model = Simple_FC() 

model.load_state_dict(torch.load("trained/Simple_FC_9.model")) 

data_size = 250

Test_path = "C:/Users/DELL/Documents/Tamara/2018 GIT ML course/3 Exams/3 Exam/Flowers Ardalan/data/test"

data_loader = load_data_test(Test_path, 64)

def test():
    model.eval()
    acc = 0
    y_hat = []
    y_true = []
    for X, y in tqdm(data_loader):
        X = X.view(-1, 3*64*64)
        out = model(X)

        predictions = torch.argmax(out, 1)
        acc += torch.sum(predictions == y).item()
        y_hat.append(predictions)
        y_true.append(y)

    y_hat = torch.cat(y_hat)
    y_true = torch.cat(y_true)
    acc = acc/data_size
    print(acc)
    print(classification_report(y_hat, y_true))


if __name__ == "__main__":
    test()
