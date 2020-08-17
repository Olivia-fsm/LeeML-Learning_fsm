# Error Log

---

| NAME      | ID           |
| --------- | ------------ |
| Fan Simin | 518370910184 |



## 2020-Aug.17     Homework2-Income Prediction

---

### Error1: Log Zero-division

- Error Info:

>**Runtime Warning: divide by zero encountered in log**

- Cause

```python
def _sigmoid(z):
    return 1/(1+np.exp(-z));


def _cross_entropy_loss(Y_label, y_pred):
    crossEntropy = -np.dot(Y_label.T, np.log(y_pred)) - np.dot((1-Y_label).T, np.log(1-y_pred))
    return crossEntropy
```

​	- ***sigmoid*** 函数进行 $$exp(-z)$$ 运算时，因为输入的**z值太大（正值）或太小（负值）**，产生了内存溢出，最终得到的结果是**nan**。所以在 **cross_entropy** 函数中的 **log** 计算引发此警告。

- Solution

```python
def _sigmoid(z):
    return np.clip(1/(1+np.exp(-z)), 1e-8, 1-(1e-8));
```

​	- [np.clip(*function_exptression*, *minimum_value*, *maximum_value*)](https://www.cnblogs.com/romin/p/10125174.html): 防止溢出！

​		等价于  if(overflow): return maximum_value(minimum_value)

### Useful Tool [tqdm](https://pypi.org/project/tqdm/):  训练模型时显示进度条~ 

```python
from tqdm import tqdm

def train_model(X_train, Y_train, X_dev, Y_dev, batch_size = 200, epochs = 100, lr = 0.5):
    trainLoss = []
    trainAcc=[]
    devLoss = []
    devAcc = []
    w_Grads = []
    b_Grads = []
    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    w, b = initialize_parameters(X_train)
    for epoch in range(epochs):
        X_train, Y_train = _shuffle(X_train, Y_train)
        with tqdm(total = int(train_size/batch_size), desc = f'epoch{epoch}:') as pbar:
            for batch in range(int(train_size/batch_size)):
                if batch_size*(batch+1)<train_size :
                    X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
                    Y_batch = Y_train[batch*batch_size:(batch+1)*batch_size]
                else:
                    X_batch = X_train[batch*batch_size:]
                    Y_batch = Y_train[batch*batch_size:]              
    #             X_batch = (batch_size*(batch+1)<train_size)? X_train[batch*batch_size:(batch+1)*batch_size]:X_train[batch*batch_size:]
    #             Y_batch = (batch_size*(batch+1)<train_size)? Y_train[batch*batch_size:(batch+1)*batch_size]:Y_train[batch*batch_size:]
                w_grad_batch, b_grad_batch = _gradient(X_batch, Y_batch, w, b)
                w = w-lr* w_grad_batch /np.sqrt(batch+1)
                b = b-lr*b_grad_batch / np.sqrt(batch+1)
                pbar.update(1)
            
            
        y_hat = _f(X_train, w, b)
        y_dev_pred =_f(X_dev, w, b)
        epoch_loss = _cross_entropy_loss(Y_train, y_hat)
        dev_loss = _cross_entropy_loss(Y_dev, y_dev_pred)
        y_hat_output = np.round(_f(X_train, w, b))
        y_dev_pred_output = np.round(_f(X_dev, w, b))
        dev_acc = _accuracy(y_dev_pred_output, Y_dev)
        train_acc = _accuracy(y_hat_output, Y_train)
        trainLoss.append(epoch_loss/train_size)
        devLoss.append(dev_loss/dev_size)
        trainAcc.append(train_acc)
        devAcc.append(dev_acc)
        w_Grads.append(w_grad_batch)
        b_Grads.append(b_grad_batch)
#         print(f'Epoch{epoch+1}: training loss={epoch_loss/train_size}, dev loss = {dev_loss/dev_size};')
    parameters = {}
    parameters['weights'] = w
    parameters['bias'] = b
    parameters['train_Loss'] = trainLoss
    parameters['train_Accuracy'] = trainAcc
    parameters['dev_Loss'] = devLoss
    parameters['dev_Accuracy'] = devAcc
    return parameters
        
train_model(X_trainSet, Y_trainSet, X_devSet, Y_devSet)
```

![image-20200817110149517](C:\Users\Olivia\AppData\Roaming\Typora\typora-user-images\image-20200817110149517.png)