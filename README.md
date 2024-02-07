Supervised Learning :Supervised learning is a machine learning category that uses labeled datasets to train algorithms to recognize patterns and predict outcomes

Self -supervised Learning : The process of the self-supervised learning method is to identify any hidden part of the input from any unhidden part of the input. For example, in natural language processing, if we have a few words, using self-supervised learning we can complete the rest of the sentence

Reinforcement Learning :  Reinforcement Learning (RL) is the science of decision making. It is about learning the optimal behaviour in an environment to obtain maximum reward
Robots equipped with visual sensors from to learn their surrounding environment. 


Latent space is a lower-dimensional space that captures the essential features of high-dimensional data.
It is a compressed representation of the original data.
Each dimension corresponds to a specific feature or characteristic.
It is typically learned by a machine learning model, such as a deep learning model or an autoencoder.
It is the lower-dimensional representation of the manifold.
It is like taking a 19D data point and squishing all that information into a 9D data point. 


Cross entropy:

No. of bits required to encode and transmit an event 
lower probability ->more info 
higher probability -> lower info

Entropy : 
Entropy, in a simple sense, is a measure of disorder or randomness in a system

Lower Entropy:

Lower entropy in machine learning implies that the data is more ordered or less uncertain. In the context of decision trees, lower entropy at a node suggests that the data at that node is more homogenous, and the decision-making process is more straightforward. The algorithm may have an easier time making predictions or classifications.

Higher Entropy:

Higher entropy indicates more disorder or uncertainty in the data. In decision trees, a higher entropy at a node suggests that the data is more diverse or mixed. Making decisions based on such data may be more challenging, as the algorithm may need to consider more factors to arrive at accurate predictions or classifications.


In a skewed distribution, the majority of the data points cluster towards one end, creating a long tail in one direction. This concentration of data points in a particular range or direction often leads to a lower entropy compared to a more evenly distributed or uniform dataset.


In a uniform distribution, all possible outcomes have roughly the same probability of occurring. This means that there is no dominant mode or peak, and the data is spread evenly across the entire range. In such a scenario, the uncertainty or lack of predictability is higher, contributing to higher entropy



Cross Entropy with Example :


Cross entropy is a concept used in information theory and machine learning to measure the difference between two probability distributions. In simple terms, it quantifies how well one probability distribution predicts the outcomes of another. Let's break it down with a straightforward example.

Example: Predicting Coin Tosses

Imagine you have a model that predicts the outcomes of coin tosses. You provide this model with a fair coin, and you want to compare its predictions to the actual outcomes.

Actual Coin Tosses:

You have a fair coin, so the probability of getting heads (H) is 0.5, and the probability of getting tails (T) is also 0.5.
Model Predictions:

Your model makes predictions based on its understanding of the coin toss. Let's say the model predicts the probability of heads (H) as 0.6 and the probability of tails (T) as 0.4.
Now, you can calculate the cross entropy to measure how well the model's predictions align with the actual outcomes.

The formula for cross entropy between the actual distribution (p) and the predicted distribution (q) is:

In our example:
H(p,q)=−[0.5⋅log(0.6)+0.5⋅log(0.4)]

This calculation gives you a numerical value that represents how well the model's predictions match the actual distribution. Lower cross entropy indicates better alignment between the predicted and actual distributions.

In summary, cross entropy is a measure of the difference between two probability distributions. In the context of machine learning, it is often used as a loss function to train models by minimizing the disparity between predicted and actual distributions.



Learning Rate:

Imagine you are learning to play a video game that involves jumping over obstacles. If you always jump too early or too late you will keep failing and have to restart the game. But if you try to adjust your timing by a small amount each time we can eventually find the sweet spot where you can consistently get over the obstacles.

Similarily in machine learning, low learning rate will result in longer training times and increased costs. On the other hand high learning rate can cause the model to overshoot or fail to converge. Therefore, finding the right learning rate is crucial to achieving best results without wasting time and resources. 
DEFN:
Machine-learnable parameters are estimated by the algorithm during training, while hyper-parameters are set by the data scientist or ML engineer to regulate how the algorithm learns and modifies the model’s performance. One such hyper-parameter is the learning rate, denoted by α, which controls the pace at which an algorithm updates the values of parameter 

{
DataSet used : MNIST (handwritten images) total:60000 images 
Each batch : 128 pictures 
Each epcoh : 60000/128 = 468.75 
}


![image](https://github.com/manojbinnal/Text-to-Image-generation-model/assets/91670995/adfe4101-abd3-4120-aa6c-f72777d23511)
