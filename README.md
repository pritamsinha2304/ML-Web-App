# Machine Learning App

## Introduction.
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

## What does it do ?

### Home Page

<img src="/static/Screenshot 2021-12-22 202851.png" alt="main_page" />

<p>This web app takes on user inputs and give out prediction based on user options. It takes on user's data as inputs and gives various metrics and prediction in tabular.</p>

<p>This app holds the implementation of machine learning models. The models has been fitted and trained over a particular datasets. Based on different types of applications and different types of algorithms, the outcomes will be different for different datasets. Each dataset went through machine learning cycle. Starting with- </p>
<ul>
    <li>EDA (Exploratory Data Analysis)</li>
    <li>Feature Engineering</li>
    <li>Selection of Model</li>
    <li>Training</li>
    <li>Validation</li>
</ul>

### Projects

<img src="" alt="projects" />

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

#### a) Iris Classification

<h2>1. Problem Statement: </h2>
<p>
  We need to classify species of Iris flower based on some features of the flower.
</p>

<h2>2. Problem Type: </h2>
<p>
  <b>Multiclass Classification</b>. There are 3 species given namely:
</p>
<ul>
  <li><b><i>Iris Setosa.</i></b></li>
  <li><b><i>Iris Versicolor.</i></b></li>
  <li><b><i>Iris Virginica</i></b></li>
</ul>

<h2>3. Dataset: </h2>
<p>
  The dataset can be found on <a href="https://www.kaggle.com/uciml/iris" target="_blank"><b style="color: #22D3EE;">Kaggle</b></a>.
  The <b>Training</b> dataset contains <i>150 rows x 5 columns</i>. The columns are <b>Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm</b>.
</p>

<h2>4. Tools/Libraries Used: </h2>
<ul>
  <li>Programming Language Used:
      <ul>
          <li>Python 3.8</li>
          <li>HTML</li>
          <li>CSS</li>
          <li>Javascript</li>
      </ul>
  </li>
  <li>File Types Used:
      <ul>
          <li>.py</li>
          <li>.html</li>
          <li>.css</li>
          <li>.js</li>
          <li>.env</li>
          <li>.gitignore</li>
          <li>.gcloudignore</li>
          <li>.pickle</li>
          <li>.conf</li>
          <li>.txt</li>
          <li>.log</li>
          <li>Dockerfile</li>
      </ul>
  </li>
  <li>Libraries Used:
      <ul>
          <li>scikit-Learn</li>
          <li>matplotlib</li>
          <li>numpy</li>
          <li>pandas</li>
          <li>seaborn</li>
          <li>statsmodels</li>
          <li>dataprep</li>
          <li>pickle</li>
          <li>logging</li>
          <li>yaml</li>
          <li>flask</li>
          <li>wtforms</li>
          <li>scipy</li>
          <li>dotenv</li>
      </ul>
  </li>
  <li>Tools Used:
      <ul>
          <li>Pycharm IDE</li>
          <li>Jupyter Notebook</li>
          <li>Apache Cassandra</li>
          <li>Docker</li>
          <li>Google Cloud Platform</li>
      </ul>
  </li>
</ul>

<h2>5. Steps taken to train the model: </h2>
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

<h2>6. Validation: </h2>
<p>
  The fitted model has been validated with <b>K Fold Cross Validation</b>.
</p>

<h2>7. Prediction: </h2>
<p>
  The fitted and validated model has been used for <b>Prediction</b> of the test data.
</p>

<h2>8. Metrics Calculation: </h2>
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
