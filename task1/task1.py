import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 任务 1-1: 基础模型构建
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        self.hidden_layer = nn.Linear(4, 16) 
        self.relu = nn.ReLU()                
        self.output_layer = nn.Linear(16, 3) 

    # 自行实现 Softmax 层 
    def custom_softmax(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        logits = self.output_layer(x)
        probs = self.custom_softmax(logits)
        return probs
    
# 任务 1-2: 训练与评估
def main():
    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.float32) 
    y = torch.tensor(data.target, dtype=torch.long)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("开始训练 MLP 模型...")
    for epoch in range(100):
        model.train()         
        optimizer.zero_grad() 

        probs = model(X_train)
        
        log_probs = torch.log(probs + 1e-8) 
        loss = -torch.mean(log_probs[range(len(y_train)), y_train])
        
        loss.backward()
        optimizer.step()

    model.eval() 
    with torch.no_grad():
        test_probs = model(X_test)
        predictions = torch.argmax(test_probs, dim=1)
        accuracy = (predictions == y_test).float().mean().item()
        print(f"训练完成！模型在测试集上的准确率(Accuracy)为: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()