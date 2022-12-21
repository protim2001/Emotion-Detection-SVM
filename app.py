# Core Pkgs
import streamlit as st 
import altair as alt

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("./models/Sentiment_Analyser_LR.pickle","rb"))
pipe_svm=joblib.load(open("./models/Sentiment_Analyser_SVM.pickle","rb"))
pipe_knn=joblib.load(open("./models/Sentiment_Analyser_KNN.pickle","rb"))
pipe_nb=joblib.load(open("./models/Sentiment_Analyser_NB.pickle","rb"))
pipe_dct=joblib.load(open("./models/Sentiment_Analyser_DCT.pickle","rb"))
# Fxn
def predict_emotions(newpipe,docx):
	results = newpipe.predict([docx])
	return results[0]

def get_prediction_proba(newpipe,docx):
	results = newpipe.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠", "fear":"😨😱", "joy":"😂","love":"🤗", "sadness":"😔",  "surprise":"😮"}


# Main Application
def main():
	st.title("App to detect emotions from Text")
	menu = ["Home","About","Creators"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Emotion In Text")
		st.caption("Here,LR = Logistic Regression , SVM = Support Vector Machines , DCT = Decision Tree , KNN = K Nearest Neighbours , NB = Naive Bayes")
		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text_lr = st.form_submit_button(label='Predict Using LR')
			submit_text_svm=st.form_submit_button(label='Predict Using SVM')
			submit_text_dct=st.form_submit_button(label='Predict Using DCT')
			submit_text_knn=st.form_submit_button(label='Predict Using KNN')
			submit_text_nb=st.form_submit_button(label='Predict Using NB')
			submit_text=(submit_text_lr or submit_text_dct or submit_text_svm or submit_text_knn or submit_text_nb)
			if(submit_text_lr):
				current_pipe=pipe_lr
			elif (submit_text_svm):
				current_pipe=pipe_svm
			elif(submit_text_dct):
				current_pipe=pipe_dct
			elif(submit_text_knn):
				current_pipe=pipe_knn
			elif(submit_text_nb):
				current_pipe=pipe_nb
		if submit_text:
			col1,col2  = st.beta_columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(current_pipe,raw_text)
			probability = get_prediction_proba(current_pipe,raw_text)
			

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)

	elif choice=="About":
		st.subheader("About Human Emotions")
		st.markdown(" Emotions can be defined as a positive or negative experience that is associated with a particular pattern of physiological activity. Emotions produce different physiological, behavioral and cognitive changes. The original role of emotions was to motivate adaptive behaviors that in the past would have contributed to the passing on of genes through survival, reproduction, and kin selection.")
		st.markdown("Textual based emotion recognition and detection is a recent field of study which is familiar to sentiment analysis. Previous studies summarised that analysis on emotion detection can be divided into two types: (i) sentiment analysis and (ii) emotion analysis (or affective computing). Usually, sentiment analysis and emotion analysis are applied interchangeably. Based on the comparison between sentiment and emotion analysis done by was clearly distinguished the concept for both analysis. Sentiment analysis is used to calculate and to detect the emotional polarity (positive, negative or neutral) from user input in textual based. Conversely, emotion analysis used to recognize and interpret the state of emotion expression from user input texts depends on several emotion categories. According to psychological models, human’s emotion can be categorised into six basic emotions (Ekman’s model) such as anger, joy, fear, happiness, sadness and surprise. Based on the collection information of previous studies on emotion recognition, Ekman emotion categories has been utilised and popular in most of the affective computing research studies and in various systems that develop to detect emotional state from text-based data")
		st.markdown("Research has demonstrated the performance of supervised learning using difference types of machine learning algorithm. However, a limited or no attempts have thus far devoted to measure the performance for several machine algorithms on this field. Therefore, the current investigation also aims to evaluate the performance of five different text classifiers such as Logistic Regression, Support Vector Machine (SVM), Naïve Bayes, k-Nearest Neighbour (k-NN) and Decision Tree")
		st.image("https://www.meaningcloud.com/wp-content/uploads/2019/11/vector-2.png")
		st.subheader("Types of Machine Learning Algorithms")
		st.subheader("Supervised Learning")
		st.markdown("Supervised Learning is a type of learning in which we are given a data set and we already know what are correct output should look like, having the idea that there is a relationship between the input and output.Basically, it is learning task of learning a function that maps an input to an output based on example inputoutput pairs. It infers a function from labeled training data consisting of a set of training examples.")
		st.subheader("Unsupervised Learning")
		st.markdown("Unsupervised Learning is a type of learning that allows us to approach problems with little or no idea what our problem should look like. We can derive the structure by clustering the data based on a relationship among the variables in data. With unsupervised learning there is no feedback based on prediction result. Basically, it is a type of self-organized learning that helps in finding previously unknown patterns in data set without pre-existing label.")
		st.subheader("Reinforcement Learning")
		st.markdown("Reinforcement learning is a learning method that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best.")
		st.subheader("Semi-Supervised Learning")
		st.markdown("Semi-supervised learning fall somewhere in between supervised and unsupervised learning, since they use both labeled and unlabeled data for training – typically a small amount of labeled data and a large amount of unlabeled data. The systems that use this method are able to considerably improve learning accuracy. Usually, semi-supervised learning is chosen when the acquired labeled data requires skilled and relevant resources in order to train it / learn from it. Otherwise, acquiring unlabeled data generally doesn’t require additional resources. ")
		st.subheader("Machine Learning Models used in the project")
		st.subheader("Logistic Regression")
		st.markdown("Logistic regression is a supervised classification is unique Machine Learning algorithms in Python that finds its use in estimating discrete values like 0/1, yes/no, and true/false. This is based on a given set of independent variables. We use a logistic function to predict the probability of an event and this gives us an output between 0 and 1. Although it says ‘regression’, this is actually a classification algorithm. Logistic regression fits data into a logit function and is also called logit regression. ")
		st.image("https://pimages.toolbox.com/wp-content/uploads/2022/04/11040522/46-4.png")
		st.subheader("Support Vector Machines")
		st.markdown("SVM is a supervised classification is one of the most important Machines Learning algorithms in Python, that plots a line that divides different categories of your data. In this ML algorithm, we calculate the vector to optimize the line. This is to ensure that the closest point in each group lies farthest from each other. While you will almost always find this to be a linear vector, it can be other than that. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. When data are unlabeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. ")
		st.image("https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm5.png")
		st.subheader("Naive Bayes Algorithm")
		st.markdown("Naive Bayes is a classification method which is based on Bayes’ theorem. This assumes independence between predictors. A Naive Bayes classifier will assume that a feature in a class is unrelated to any other. Consider a fruit. This is an apple if it is round, red, and 2.5 inches in diameter. A Naive Bayes classifier will say these characteristics independently contribute to the probability of the fruit being an apple. This is even if features depend on each other. For very large data sets, it is easy to build a Naive Bayesian model. Not only is this model very simple, it performs better than many highly sophisticated classification methods. Naïve Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression, which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers")
		st.image("https://miro.medium.com/max/1200/1*39U1Ln3tSdFqsfQy6ndxOA.png")
		st.subheader("K Nearest Neighbors")
		st.markdown("This is a Python Machine Learning algorithm for classification and regression- mostly for classification. This is a supervised learning algorithm that considers different centroids and uses a usually Euclidean function to compare distance. Then, it analyzes the results and classifies each point to the group to optimize it to place with all closest points to it. It classifies new cases using a majority vote of k of its neighbors. The case it assigns to a class is the one most common among its K nearest neighbors. For this, it uses a distance function. k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. k-NN is a special case of a variablebandwidth, kernel density 'balloon' estimator with a uniform kernel. ")
		st.image("http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png")
		st.subheader("Decision Tree")
		st.markdown("A decision tree falls under supervised Machine Learning Algorithms in Python and comes of use for both classification and regression- although mostly for classification. This model takes an instance, traverses the tree, and compares important features with a determined conditional statement. Whether it descends to the left child branch or the right depends on the result. Usually, more important features are closer to the root. Decision Tree, a Machine Learning algorithm in Python can work on both categorical and continuous dependent variables. Here, we split a population into two or more homogeneous sets. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. ")
		st.image("https://static.javatpoint.com/tutorial/machine-learning/images/decision-tree-classification-algorithm.png")
	else:
		st.subheader("Project members")
		st.subheader("Protim Aich")
		st.markdown("Autonomy Roll Number - 12619001099")
		st.markdown("Class Roll Number - 1951064")
		st.subheader("Swapnil Chowdhury")
		st.markdown("Autonomy Roll Number - 12619001172")
		st.markdown("Class Roll Number - 1951065")
		st.subheader("Anurag Nayak")
		st.markdown("Autonomy Roll Number - 12619001035")
		st.markdown("Class Roll Number - 1951011")
		st.subheader("Shivam Shresth")
		st.markdown("Autonomy Roll Number - 12619001145")
		st.markdown("Class Roll Number - 1951001")

if __name__ == '__main__':
	main()