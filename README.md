# Character-level-language-model-Dinosaurus-Island

 The code is in Python 3 </br>

### Model Overview 

* Initialize parameters
* Run the optimization loop
    * Forward propagation to compute the loss function
    * Backward propagation to compute the gradients with respect to the loss function
    * Clip the gradients to avoid exploding gradients
    * Using the gradients, update your parameters with the gradient descent update rule.
* Return the learned parameters

* At each time-step, the RNN tries to predict what is the next character given the previous characters.
* The dataset X=(x⟨1⟩,x⟨2⟩,...,x⟨Tx⟩) is a list of characters in the training set.
* Y=(y⟨1⟩,y⟨2⟩,...,y⟨Tx⟩) is the same list of characters but shifted one character forward.
* At every time-step t, y⟨t⟩=x⟨t+1⟩. The prediction at time t is the same as the input at time t+1.

![model](readme_images/rnn.png)
</br>

I used Sampling to generate new names, as shown in the figure . </br>

![sample](readme_images/sampling.png)

We assume the model is trained and paas a dummy vector as input and generate a new name based on the parameters learned by the RNN.
</br >

### How to Run

The model is already implemented in **model.py**, just run it on command line </br >

* Output
The code outputs a set of names on every 2000th optimization loop along with the loss. </br>
  * sample output
  
         Iteration: 34000, Loss: 22.447230

          Onyxipaledisons
          Kiabaeropa
          Lussiamang
          Pacaeptabalsaurus
          Xosalong
          Eiacoteg
          Troia
