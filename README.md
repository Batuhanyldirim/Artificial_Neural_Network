# Artificial_Neural_Network

This code gets %89 accuracy on MNIST dataset to label hadwritten digits which coded without using any libraries except pandas or numpy





### Figure of accuracy


![image](https://user-images.githubusercontent.com/41572446/122049297-8e61a900-cde2-11eb-8ae0-2446f7ba22a6.png)

This is the validation accuracy during training process



### Structure of the ANN

The ANN designed as 3 layers of neurons which they have 784, 256, 10 neurons, respectively. The first layer is input layer and it has 784 neurons. Those inputs are directly coming from inputs so there are no extra calculations in this neuron. The second layer has 256 neurons, and it is called as hidden layer. This layer takes inputs from first layer after processed with weights and biases. The last layer has 10 neurons, and it is called as output layer. This layer takes inputs from hidden layer after processed with weights and biases. Eventually there is a softmax function applied in last layer to decide which one has the highest probability as a result

![image](https://user-images.githubusercontent.com/41572446/122049500-c8cb4600-cde2-11eb-90cb-f2155e0abe25.png)



### Equations that used to update weights.

The generic formula for updating weights is 
![image](https://user-images.githubusercontent.com/41572446/122049650-f44e3080-cde2-11eb-8ea1-008fd4e68678.png)
can be considered as δj.ⴄmeans learning rate. This is a coefficient which scales the change on the weight. If it is so small, then the learning is so slow and if it is so big we can step over the the optimum pointand may not converge it. 

δjis the error term and it is the error between desired output and given output from the neuron. This term is calculated differently if it belongs to an output layer or a hidden layer.

*  **If δjis an output layer then** ![image](https://user-images.githubusercontent.com/41572446/122050384-b43b7d80-cde3-11eb-9b78-09fc994f1ddc.png)

> tkmeans the target value for kthneuron in the layer
> okmeans the output value for the kthneuron in the layer
> f’(ok) means the derivative of activation function, so in this case it is f’(ok) = ok(1-ok)

*  **If δjis a hidden layer then δj= f’(ok)** ![image](https://user-images.githubusercontent.com/41572446/122050820-290eb780-cde4-11eb-8715-9827f6349de3.png)

> f’(ok) means the derivative of activation function, so in this case it is f’(ok) = ok(1-ok)
> δk is the next layers neuron (eventually this is equal to an output neuron)
> wkjis the kthneurons weight of the downstream layer

Xjiis the input of that exact neuron

### Percentage of correctness
The accuracy of total dataset is calculated as 0.89 and the each classes accuracy is given as;

![image](https://user-images.githubusercontent.com/41572446/122051439-e39eba00-cde4-11eb-9ad7-7a7e06ded3e8.png)

 
