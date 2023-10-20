# Movie Success Prediction Project

## Description of the Problem

- Brief introduction about the problem you are solving.
- Explanation of how a model could be used to solve this problem.

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

## Data

The data for this project was collected from two different APIs: [OMDB API](https://www.omdbapi.com/apikey.aspx) and [TMDB API](https://developer.themoviedb.org/reference/intro/getting-started). The data from both APIs was merged into a single dataset based on the `imdb_id` that they both share. Additionally, credits data is loaded from [TMDb](https://developer.themoviedb.org/reference/movie-credits) to gather more information about the cast and crew of the movies.

The reason for fetching data from two APIs and loading credits is to compile a comprehensive dataset, where OMDb provides additional data regarding awards and nominations, and the credits provide insights into the individuals involved in the making of the movies.

### Data Collection

1. **API Keys and Tokens:**

   - Before starting the data collection process, you'll need to obtain the necessary API keys and tokens.
   - For TMDB, get your `API_KEY` and `API_TOKEN`.
   - For OMDB, get your `OMDB_KEY`.
   - Place these keys and tokens in a `.env` file in the project root.

2. **Data Scraping:**

   - The data scraping process is documented in detail in the Jupyter Notebook `notebooks/scrape_data.ipynb`.
   - A function `fetch_all_movies(start_id, last_id)` is used to fetch movie data within a specified range of IDs.
   - Due to the OMDB API limit of 1000 requests per day, a condition was set to filter out unnecessary data and to ensure that the essential data is collected without exceeding the daily quota.

    ```python

    if movie is not None and movie["imdb_id"] and \
        movie['revenue'] !=0 and movie['status'] == 'Released' and \
        movie ['budget'] !=0:
    ```

    - The movies were scraped in batches of 1000 per day to abide by the OMDB API limit, and saved in separate Parquet files.
    - All the scraped data was then combined and saved into ./data/movies.parquet.

3. **Additional Data:**
    A separate dataset containing credits (crew and cast) for each movie was also scraped and saved to ./data/credits.parquet.

### Datasets

**Movies Dataset**: Contains combined data from OMDB and TMDB APIs, saved in data/movies.parquet.
**Credits Dataset**: Contains crew and cast data for each movie, saved in data/credits.parquet.
Ensure to follow the detailed instructions in notebooks/scrape_data.ipynb to understand and replicate the data scraping process if you want to.

## Data Preparation and Cleaning

- Description of the steps taken to prepare and clean the dataset.

## Exploratory Data Analysis (EDA) and Feature Importance Analysis

- Analysis of important features and any insights derived from EDA.

## Model Selection Process and Parameter Tuning

- Description of the models trained, the parameter tuning process, and the selection of the best model.

## Training the Final Model

- Script (`train.py`)
- Explanation of the code and steps taken to train the final model.
- Instructions on how the model is saved to a file (e.g., using Pickle or BentoML).

## Loading the Model and Serving It via a Web Service

- Script (`predict.py`)
- Instructions on how to load the trained model.
- Explanation of how the model is served via a web service (e.g., using Flask or BentoML).

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
- Navigate to the project root directory.
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

- Now that the model is running in a Docker container, you can access it by going to http://localhost:8000 in your browser

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

### Installing Minikube

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

### Running Minikube

1. **Access Minikube Dashboard**:

    - Once Minikube is started, you can access the Kubernetes Dashboard, a web-based Kubernetes user interface, by running:

    ```bash
    minikube dashboard
    ```

2. **Deploy Model**:

3. **Access the Model**:

4. **Delete the Service and Deployment**:

5. **Stop Minikube**:

    ```bash
    minikube stop
    ```
