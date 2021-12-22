<h1> Machine Learning App</h1>

<h2> Introduction </h2>
<p>Since the advent of machine learning, various app, websites, blogs uses various techniques, styles to use power of AI to train and give their customer the advantage or prediction of what about to come. If a person knows what is coming, he/she can prepare for the worst and avoid the worst case. This has been made possible by various machine learning, deep learning etc. techniques. </p>

<p>With the development of machine learning with ages, various improvement, techniques has been developed to harness the power of machine learning to tachle all kinds of problems or more precisely to be used on various problems. Various techniques include- </p>

<ul>
  <li>Machine Learning</li>
  <li>Deep Learning</li>
  <li>Natural Language Processing</li>
  <li>Generative Adversarial Networks</li>
  <li>Reinforcement Learning</li>
</ul>

These techniques uses different kinds of inputs and output the result, like- 
Machine Learning can be used on Tabular Structured Data, Clustering Data Points etc., 
Deep Learning can be used on Images, Videos etc.,
Natural Language Processing can be used on Text Processing, Language Translation etc.,
Generative Adversarial Network can be used to generate fake image,
Reinforcement Learning is all about self learning.

Since these techniques work on different inputs so various of these techniques can be applied to various kinds of problems to tackle/predict the best solution possible.

<h2> What does it do ? </h2>

<h3> Home Page </h3>

<img src="/static/main_page.png" alt="main_page" width="30%"/>

<p>This web app takes on user inputs and give out prediction based on user options. It takes on user's data as inputs and gives various metrics and prediction in tabular.</p>

<p>This app holds the implementation of machine learning models. The models has been fitted and trained over a particular datasets. Based on different types of applications and different types of algorithms, the outcomes will be different for different datasets. Each dataset went through machine learning cycle. Starting with- </p>
<ul>
    <li>EDA (Exploratory Data Analysis)</li>
    <li>Feature Engineering</li>
    <li>Selection of Model</li>
    <li>Training</li>
    <li>Validation</li>
</ul>

<h3> Projects </h3>

<img src="/static/projects.png" alt="projects" width="30%"/>

<p>There are currently 4 types of projects been implemented across multiple machine learning techniques- </p>
<ul>
  <li>Machine Learning</li>
  <li>Deep Learning</li>
  <li>Natural Language Processing</li>
</ul>

<p>across multiple domains- </p>
<ul>
  <li>Agriculture</li>
  <li>Pets/Animal</li>
  <li>Media</li>
  <li>Conversation</li>
</ul>
etc.

Each projects has a litle detailed information about the problem, its type, steps taken, dataset etc. Each projects has 2 types of forms, one is prediction form and one is the demo form.
The <b>prediction form</b> takes on user data as inputs and do the necessary EDA and ouput the results.
The <b>demo form</b> is for the users who do not have their own data, they just want to see what it does.

<h3> a) Iris Classification </h3>

<img src="/static/iris_upload.png" alt="iris_upload" width="30%"/> <img src="/static/iris_demo.png" alt="iris_demo" width="30%"/> <img src="/static/iris_result.png" alt="iris_result" width="30%"/>

<h3>1. Dataset: </h3>
<p>
  The dataset can be found on <a href="https://www.kaggle.com/uciml/iris" target="_blank"><b style="color: #22D3EE;">Kaggle</b></a>.
</p>

<h3>2. Steps taken to train the model: </h3>
<ul>
  <li>For Upload Data:
      <ul>
          <li>Load the dataset using <b>Pandas</b>.</li>
          <li>Checking for unnecessary columns.</li>
          <li>Check for <b>Imbalanced</b> dataset, since this is a classification problem</li>
          <li>Checking for <b>Correlation</b>.</li>
          <li>Checking for <b>Outliers</b>.</li>
          <li>Checking for <b>Missing, NaN</b> values.</li>
          <li>Checking the <b>Distribution</b> of columns/features, especially <b>Gaussian Distribution</b>.</li>
          <li>Used <b>Label Encoder</b> to encode the target values</li>
          <li>Used <b>Standard Scaler</b> transformer to standardize the data</li>
          <li>Used <b>Train Test Split</b>.</li>
          <li><b>Fit</b> the model with the following algorithms:
              <ul>
                  <li><b>Logistic Regression</b>.</li>
                  <li><b>K Nearest Neighbors</b>.</li>
                  <li><b>SVC</b>.</li>
                  <li><b>Decision Tree</b>.</li>
                  <li><b>Random Forest</b>.</li>
                  <li><b>Ada Boost</b>.</li>
                  <li><b>Gradient Boosting</b>.</li>
                  <li><b>Extra Trees</b>.</li>
                  <li><b>XGBoost</b>.</li>
                  <li><b>Gaussian Naive Bayes</b>.</li>
                  <li><b>Multilayer Perceptron Classifier</b>.</li>
              </ul>
          </li>
      </ul>
  </li>
  <li>For Demo Data:
      <ul>
          <li>The demo csv has been uploaded to <b>Cassandra</b> database</li>
          <li>The data is requested from the database whenever necessary</li>
          <li>Necessary prediction is done based on the selected model on the data</li>
      </ul>
  </li>
</ul>

<h3>3. Metrics Calculation: </h3>
<p>
  Different classification metrics are used to check the efficiency of the validated model given the actual data:
</p>
<ul>
  <li><b>Accuracy Score</b>.</li>
  <li><b>Classification Report</b>.</li>
  <li><b>F1 Score</b>.</li>
  <li><b>Precision Score</b>.</li>
  <li><b>Recall Score</b>.</li>
  <li><b>Confusion Matrix</b>.</li>
</ul>


<h3> b) Cat Dog Classification </h3>

<img src="/static/cog_upload.png" alt="cog_upload" width="30%"/> <img src="/static/cog_demo.png" alt="cog_demo" width="30%"/> <img src="/static/cog_result.png" alt="cog_result" width="30%"/>

<h3>1. Dataset: </h3>
<p>
    The dataset can be found on <a href="https://www.kaggle.com/c/dogs-vs-cats" target="_blank"><b style="color: #22D3EE;">Kaggle</b></a>.
</p>

<h3>2. Steps taken to train the model: </h3>
<ul><li>Create the CNN architecture. Here 2 kinds of architecture are used to train:
    <ul>
        <li>Simple Custom CNN <a href="/static/CatDog.svg" target="_blank" ><b style="color: #22D3EE;">architecture</b></a></li>
        <li>Transfer Learning using Resnet50 <a href="/static/CatDogResnet50.svg" target="_blank" ><b style="color: #22D3EE;">architecture</b></a></li>
    </ul>
</li>
<li>Load the images in the directory.</li>
<li>Read the images via <b>Keras</b> with size of <b>256x256</b> for simple model and <b>224x224</b> for the resnet50 model.</li>
<li>Split the images for the <b>Train Validation Split</b>.</li>
<li>Construct the layers for the simple architecture or using transfer learning for the resnet50 architecture</li>
<li><b>Fit</b> the model with the CNN architecture with <b>10</b> epochs and batch size of <b>32</b>.</li>
</ul>
<p><b style="color:red;">NOTE:</b>Even though the dataset contains altogether around <b>24000</b> images of cats and dogs.
The training process actually has been done on around <b>6000</b> images of cats and around <b>6000</b> images of dogs for the both architecture, due to the limitation of hardware or processing power.
As a result of these, the validation accuracy, with validation split as <b>0.2</b>, is around <b>74%</b> for the simple cnn model and around <b>97%</b> for the resnet50 model.
So, the predicted result might show some discrepancies in accuracy or also might get wrong result.
In case of transfer learning with ResNet50, the layers are not trained again, <b>imagenet</b> weights are been used.
</p>
<p><b style="color:red;">NOTE:</b>The number of epochs is chosen to be this small because of limitation of processing power.
This took around 5 hours to train. Since the epochs are small, the <b>accuracy</b> is around <b>94.49%</b> for the simple model and <b>99.63%</b> for the resnet50 model.
</p>

<h3>3. Metrics Calculation: </h3>
<p>
Different classification metrics are used to check the efficiency of the validated model given the actual data:
</p>
<ul>
<li><b>Accuracy Score</b>.</li>
<li><b>Classification Report</b>.</li>
<li><b>F1 Score</b>.</li>
<li><b>Precision Score</b>.</li>
<li><b>Recall Score</b>.</li>
<li><b>Confusion Matrix</b>.</li>
</ul>


<h3> c) Fake News Classification </h3>

<img src="/static/fakenews_single_upload.png" alt="fakenews_single" width="30%" /> <img src="/static/fakenews_multiple_upload" alt="fakenews_multiple" width="30%" /> <img src="/static/fakenews_demo" alt="fakenews_demo" width="30%" /> <img src="/static/fakenews_result" alt="fakenews_result" width="30%" />

<h3>1. Dataset: </h3>
  <p>
      The dataset can be found on <a href="https://www.kaggle.com/c/fake-news/data" target="_blank"><b style="color: #22D3EE;">Kaggle</b></a>.
  </p>
  
  <h3>2. Steps taken to train the model: </h3>
  <ul><li>Create the NLP architecture. Here's only one kind of architecture is used to train:
          <ul>
              <li>Simple Custom NLP Model <a href="/static/FakeNewsClassifier.svg" target="_blank" ><b style="color: #22D3EE;">architecture</b></a></li>
          </ul>
      </li>
      <li>Load the dataset containing the text data.</li>
      <li>Combine all the text column data into one column</li>
      <li>Using SnowballStemmer to filter different characters which we do not need</li>
      <li>Tokenize the words using keras tokenizer</li>
      <li>All the rows are fixed to a fixed column length size using padding. Here the max length used is <b>100</b></li>
      <li>Split the padded rows using <b>Train-Test-Split</b></li>
      <li>Construct the layers for the simple architecture</li>
      <li>All the word are vectorized of length <b>1000</b> using <b>Word2Vec</b> in the <b>Embedding layer</b> of keras</li>
      <li><b>Fit</b> the model with the NLP architecture with <b>5</b> epochs and batch size of <b>32</b>.</li>
  </ul>
  <p><b style="color:red;">NOTE:</b>Even though the dataset contains multiple columns. Here only some columns are used to train the model
      as the no. of words gets bigger and bigger and so is the vector size and the vocabulary size. So the stemming process will take a lot of time.
  </p>
  <p><b style="color:red;">NOTE:</b>The number of epochs is chosen to be this small because of limitation of processing power.
      Since the epochs are small, the <b>accuracy</b> is around <b>99.92%</b> and <b>validation-accuracy</b> of <b>99.48%</b> for the simple model.
  </p>

  <h3>3. Metrics Calculation: </h3>
  <p>
      Different classification metrics are used to check the efficiency of the validated model given the actual label data:
  </p>
  <ul>
      <li><b>Accuracy Score</b>.</li>
      <li><b>Classification Report</b>.</li>
      <li><b>F1 Score</b>.</li>
      <li><b>Precision Score</b>.</li>
      <li><b>Recall Score</b>.</li>
      <li><b>Confusion Matrix</b>.</li>
  </ul>

<h3> d) Mask Token Prediction </h3>

  Using the models **BERT** and **DistilBERT** from [Huggingface](https://huggingface.co)
  
  <img src="/static/mask_upload.png" alt="mask_upload" width="30%" /> <img src="mask_result" alt="mask_result" width="30%" />
  
<h3> e) Text Generation </h3>

  Using the models **GPT2** and **DistilGPT2** from [Huggingface](https://huggingface.co)
  
  <img src="/static/textgenerate_upload.png" alt="textgenerate_upload" width="30%" /> <img src="/static/textgenerate_result.png" alt="textgenerate_result" width="30%" />
  
<h3> f) Conversation </h3>

  Using the models **Blender Bot Small** from [Huggingface](https://huggingface.co)
  
  <img src="/static/conversation_upload.png" alt="conversation_upload" width="30%" /> <img src="/static/conversation_result.png" alt="conversation_result" width="30%" />
