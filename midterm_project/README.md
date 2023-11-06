# Movie Success Prediction Project

## Problem Description

In this project, as a part of [MLZoomCamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), I am tackling the question: "Can I predict whether a movie will be successful?" Since success can mean different things, I am focusing on three specific areas: a movie's profitability (Revenue vs Budget), its performance at award ceremonies, and the ratings it receives from viewers and critics.

## Multi-Label Classification

To make these predictions, I use a technique called [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification). This means that instead of predicting just one outcome, I am going to predict multiple outcomes at once – in this case, the three different success metrics. It's like giving each movie a report card that indicates whether it's likely to be profitable, win awards, and receive favorable ratings.

## My Plan of Action

I explore my dataset and evaluate different models on it to identify the most effective one for predicting movie success. I transform the top performer model into a service that anyone can use. Additionally, I package it in a Docker file, ensuring it can be run in various environments.

## Dataset

**Movies Dataset**: My dataset, consisting of around 8000 rows, combines data from the OMDB and TMDB APIs and is saved in [`data/movies.parquet`](https://github.com/olgazju/ml_camp_2023/blob/main/midterm_project/data/movies.parquet).

## Data

The data for this project was collected from two different APIs: [OMDB API](https://www.omdbapi.com/apikey.aspx) and [TMDB API](https://developer.themoviedb.org/reference/intro/getting-started). The data from both APIs was merged into a single dataset based on the `imdb_id` that they both share during scrapping.

The reason for fetching data from two APIs is to compile a comprehensive dataset, where OMDb provides additional data regarding awards and nominations, and the tmdb metadata + credits provide insights into the individuals involved in the making of the movies and common information about movies.

To reproduce my Notebooks you don't need to recollect data, I'm just placing here an exact description of how the data was collected so the process can be repeated by anyone.

### Data Collection

1. **API Keys and Tokens:**

   - Before starting the data collection process, you'll need to obtain the necessary API keys and tokens.
   - For TMDB, get your `API_KEY` and `API_TOKEN`.
   - For OMDB, get your `OMDB_KEY`.
   - Place these keys and tokens in a `.env` file in the project root.

2. **Data Scraping:**

   - The data scraping process is documented in detail in the Jupyter Notebook [**notebooks/scrape_data.ipynb**](https://github.com/olgazju/ml_camp_2023/blob/main/midterm_project/notebooks/scrape_data.ipynb).
   - A function `fetch_all_movies(start_id, last_id)` is used to fetch movie data within a specified range of IDs.
   - Due to the OMDB API limit of 1000 requests per day, a condition was set to filter out unnecessary data and to ensure that the essential data is collected without exceeding the daily quota.

    ```python

    if movie is not None and movie["imdb_id"] and \
        movie['revenue'] !=0 and movie['status'] == 'Released' and \
        movie ['budget'] !=0:
    ```

    - The movies were scraped in batches of 1000 per day to abide by the OMDB API limit, and saved in separate Parquet files.
    - All the scraped data was then combined and saved into `./data/movies.parquet`.

## Instructions on How to Run the Project

### Setting up the Environment

1. First, ensure you have [pyenv](https://github.com/pyenv/pyenv) installed on your system.

2. Install Python 3.11.4 using pyenv:

    ```bash
    pyenv install 3.11.4
    ```

3. Navigate to the project directory:

    ```bash
    cd midterm_project
    ```

4. Create a new virtual environment for this project:

    ```bash
    pyenv virtualenv 3.11.4 midterm-project
    ```

5. Set the local Python version to use the virtual environment you just created:

    ```bash
    pyenv local midterm-project
    ```

6. Now, the project directory is set up with a local virtual environment using Python 3.11.4.

### Installing Dependencies

1. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Setting up Visual Studio Code (VSCode)

1. Open the project directory in VSCode.
2. Ensure you have the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) installed in VSCode.
3. You can now create and open Jupyter Notebooks within VSCode.
4. When you run Jupyter Notebook in VSCode choose midterm-project Python environment from drop-down list of avaliavle environments.

### Setting up Jupyter Notebook Independently

1. If you prefer to use Jupyter Notebook outside of VSCode, ensure you have Jupyter installed:

    ```bash
    pip install jupyter
    ```

2. Launch Jupyter Notebook from the project directory:

    ```bash
    jupyter notebook
    ```

3. Jupyter Notebook will open in your web browser, and you can create new notebooks or open existing notebooks from the browser interface.

## Data Preparation and Cleaning

I have documented the data preparation and cleaning process in the [**notebooks/1-Data-Cleaning.ipynb**](https://github.com/olgazju/ml_camp_2023/blob/main/midterm_project/notebooks/1-Data-Cleaning.ipynb) notebook. During this process, I addressed various inconsistencies, missing values, and irrelevant columns to ensure a clean and reliable dataset for subsequent analysis. I dropped columns that were redundant or not useful for my analysis and created new columns to better represent the data where necessary.

In the end, the cleaned dataset was saved to **data/cleaned/movies_dataset.parquet** for use in EDA.

## Exploratory Data Analysis (EDA) and Feature Importance Analysis

The exploratory data analysis is detailed in [**notebooks/2-EDA.ipynb**](https://github.com/olgazju/ml_camp_2023/blob/main/midterm_project/notebooks/2-EDA.ipynb), using the dataset located at **data/cleaned/movies_dataset.parquet**. In the EDA, I dive into the distribution and characteristics of both the labels and features. I explore how different features correlate with the labels and perform a feature importance analysis. This helps me understand which features have the most influence on a movie's success and guides my model selection and tuning efforts.

In the end, the cleaned dataset was saved to **data/cleaned/selected_features.parquet** for use in Modeling.

## Model Selection Process and Parameter Tuning

In [**notebooks/3-Model.ipynb**](https://github.com/olgazju/ml_camp_2023/blob/main/midterm_project/notebooks/3-Model.ipynb), I explore a variety of models suitable for multilabel classification tasks. I evaluated their performance using several metrics, applied cross-validation to ensure their generalizability, and fine-tuned the parameters of the most promising models. Through this process, I identified the best-performing model that I will move forward with.

## Training the Final Model

I copied the code from data cleaning and features extraction process into `midterm_project/train.py` file.

It loads the dataset from 'midterm_project/data/movies.parquet', cleans it and extract features. Then it train catboost_classifier = CatBoostClassifier(loss_function='MultiLogloss',eval_metric='HammingLoss', iterations=400, depth=6, learning_rate=0.1, random_state=42) and saves the model, dictvectorizer and standartscaler to midterm_project/models_binary folder as 3 files:

- catboost_classifier_model.pkl
- dict_vectorizer.pkl
- standard_scaler.pkl

How to run:

```python
python train.py
```

## Loading the Model and Serving It via a Web Service

Finally, I've set up a web service (Fast API) to serve the trained model. In real-world applications, a web service should handle prediction, data cleaning, and feature extraction directly on the input data. However, for the purposes of this study project, I've simplified the process. I fetched a few sample movies from movie APIs, cleaned them, extracted features, and then converted them into JSON format for HTTP requests.

The code for the service is located in the `midterm_project/predict.py` file.

### How to Run the Service

To run the web service, use the following command in the terminal:

```bash
uvicorn predict:app --reload
```

Then open `midterm_project/predict_request.ipynb`. Here you can find code for request to the server (sort of client) and JSON samples for predictions, copy different JSONs to client variable and click run.

<img width="1028" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/fd52d2c0-81fc-465d-95bf-62ce443e464b">


## Docker

To containerize and run the model locally using Docker, you'll need to follow these steps:

### 1. **Install Docker Desktop for Mac**

- Go to the [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop) download page.
- Click on "Download for Mac" and the download will start automatically.
- Once downloaded, double click on the Docker.dmg file to open it.
- Drag the Docker icon to your Applications folder to complete the installation.
- Launch Docker Desktop from your Applications folder.

### 2. **Build the Docker Image**

- Open a terminal.
- Navigate to the project root directory midterm_project.
- Run the following command to build `movie-success` Docker image:

```bash
docker build -t movie-success .
```

### 3. **Run the Docker Image**

- After successfully building the image, run the following command to start a container from the image

```bash
docker run -p 8000:8000 movie-success
```

### 4. **Accessing the Service**

- Now that the model is running in a Docker container, open `midterm_project/predict_request.ipynb`. Here you can find code for request to the server (sort of client) and JSON samples for predictions, copy different JSONs to client variable and click run.

<img width="1028" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/544956c4-0791-4d51-a3ba-44d228e1fb0b">


### 5. **Stop the Docker Container**

- To stop the Docker container, find the container ID with the following command:

```bash
docker ps
```

Then stop the container with:

```bash
docker stop <container-id>
```

## Cloud Deployment

### Installing and Running Minikube

Minikube is a tool that lets you run Kubernetes locally. Minikube runs a single-node Kubernetes cluster on your personal computer (including Windows, macOS, and Linux PCs) so that you can try out Kubernetes, or for daily development work.

1. **Install Homebrew** (if it's not already installed):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install Minikube via Homebrew**:

    ```bash
    brew install minikube
    ```

3. **Start Minikube**:

    ```bash
    minikube start
    ```

4. **Check Minikube Installation**:

    ```bash
    minikube status
    ```

   <img width="748" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/9eec7d93-15bf-4066-80f7-2b9a6552241a">

4. **Install kubectl**:

   ```bash
       brew install kubectl
    ```
   
5. **To point your terminal to use the docker daemon inside minikube run this**

   ```bash
       eval $(minikube docker-env)
    ```

   Now any ‘docker’ command you run in this current terminal will run against the docker inside minikube cluster.

6. **To build docker image inside minikube**

   ```bash
       minikube cache add python:3.10.12-slim
       docker build -t movie-success . --progress=plain
    ```

   If I run
   
   ```bash
       docker ps
    ```

   I should see my movie-success:latest image:
   <img width="726" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/0baab5a6-2558-42a2-8f61-dd254e3d58d5">

7. **Let's deploy the model**

   You have deployment.yaml file in the project folder. Deploy it and check if it went well.

   ```bash
       kubectl apply -f deployment.yaml
       kubectl get deployments
    ```

   <img width="533" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/ad462aef-0834-428c-b2cf-ff1fe09ae76a">

8. **Expose the application**

   Create a Kubernetes service:

   ```bash
       kubectl expose deployment fastapi-movie-success-deployment --type=NodePort --port=8000
    ```

   Find the Minikube Service URL. The command below will open a browser with the right link for your service. Copy this link, open `midterm_project/predict_request.ipynb`. Here you can find code for 'Request for minikube deployment:' and insert it to url = variable.

   ```bash
       minikube service fastapi-movie-success-deployment
    ```

   <img width="713" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/821bec3c-2939-4ee7-8e37-44548e9c8855">

   <img width="314" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/67d05ec3-2b98-44d4-9c88-c422cfe33fcd">

   <img width="1032" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/b0781013-0490-462c-9ed9-47032bc81314">

9. **To stop minukube**

   ```bash
       minikube stop
    ```

#### What is this deployment.yaml file?

This YAML file is for a Kubernetes deployment configuration named `fastapi-movie-success-deployment`. It's designed to deploy a FastAPI application containerized with the Docker image `movie-success:latest`. The deployment ensures that one replica of the application is running (`replicas: 1`). It uses the label `app: fastapi` to manage and identify the pods. The `imagePullPolicy: IfNotPresent` setting is used to avoid pulling the image from a remote registry if it already exists locally. The application listens on port 8000 inside the container, aligning with the FastAPI server's default port.

