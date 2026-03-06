# 📊 Facebook Live Posts Engagement Analysis & Clustering

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** and **K-Means clustering** on Facebook Live post data to understand engagement patterns.

The analysis investigates how **likes, comments, shares, and reactions** vary across different types of posts and posting times. Using clustering techniques, posts are grouped into similar engagement patterns.

Social media analytics involves collecting and analyzing data from social networks to discover insights and trends in user engagement.

---

# 🎯 Objectives

The main objectives of this project are:

* Analyze engagement metrics on Facebook posts
* Understand the relationship between **likes, comments, shares, and reactions**
* Study how **posting time affects engagement**
* Perform **Exploratory Data Analysis (EDA)**
* Use **K-Means clustering** to group posts with similar engagement patterns

---

# 📂 Dataset

The dataset contains Facebook Live post information including:

* Post type (video, photo, status, link)
* Post publication time
* Number of reactions
* Number of comments
* Number of shares
* Reaction types (likes, loves, haha, wow, sad, angry)

This dataset is commonly used to analyze Facebook Live sellers and audience engagement behavior. 

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# 📊 Exploratory Data Analysis (EDA)

The notebook performs several analysis steps:

### 1️⃣ Data Cleaning

* Handling missing values
* Removing unnecessary columns
* Checking dataset structure

### 2️⃣ Statistical Overview

* Dataset shape
* Summary statistics
* Feature distributions

### 3️⃣ Data Visualization

Visualizations used in this project include:

* Distribution plots
* Bar charts
* Scatter plots
* Correlation heatmaps
* Engagement trends by time

These visualizations help reveal patterns in user interactions with Facebook posts.

---

# 📈 Engagement Analysis

The analysis focuses on the following engagement metrics:

* 👍 Likes
* 💬 Comments
* 🔁 Shares
* ❤️ Other reactions (love, wow, haha, sad, angry)

Key observations include:

* Video posts often generate higher engagement.
* Posting time influences audience interaction.
* Certain reaction types dominate engagement patterns.

---

# 🤖 Machine Learning Model

## K-Means Clustering

K-Means is an **unsupervised learning algorithm** used to group similar data points into clusters based on feature similarity. ([RStudio Pubs][3])

In this project, clustering is used to identify groups of posts with similar engagement behavior.

### Steps:

1. Feature selection
2. Data normalization
3. Determining optimal number of clusters
4. Applying K-Means algorithm
5. Visualizing cluster groups

---

# 📊 Clustering Insights

The clustering model helps identify patterns such as:

* Highly engaging posts
* Moderately engaging posts
* Low engagement posts

This helps understand **which types of posts perform best on Facebook Live**.

---

# 📷 Example Visualizations

The notebook generates insights through graphs such as:

* Engagement distribution plots
* Reaction comparison charts
* Posting time vs engagement analysis
* Cluster visualization graphs


---

# 📁 Project Structure

```
Facebook-Live-project
│
├── Case_Statement_Facebook_live.ipynb
├── Live.csv
├── README.md
```

---

# 🚀 How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/bhavyacloud29/Beginner-Python-projects.git
```

### 2️⃣ Navigate to project folder

```
cd Beginner-Python-projects/Facebook-Live-project
```

### 3️⃣ Install required libraries

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4️⃣ Run the notebook

```
jupyter notebook Case_Statement_Facebook_live.ipynb
```

---

# 🌍 Applications

This analysis can be useful for:

* Social media marketers
* Digital marketing analysts
* Data science beginners learning EDA
* Businesses optimizing Facebook content strategy

---

# 👨‍💻 Author

**Bhavya**

GitHub:
https://github.com/bhavyacloud29

---

# ⭐ Support

If you like this project, please consider **starring ⭐ the repository**.

[1]: https://en.wikipedia.org/wiki/Social_media_analytics?utm_source=chatgpt.com "Social media analytics"
[2]: https://medium.com/%40hariprasad.thalishetti/facebook-live-sellers-in-thailand-856b4d751059?utm_source=chatgpt.com "Facebook Live Sellers in Thailand | by Hariprasad Thalisetti"
[3]: https://rstudio-pubs-static.s3.amazonaws.com/732645_a99baedc1e294ac88f7a146a53e3e949.html?utm_source=chatgpt.com "KMeans Clustering with Facebook Thailand Data"

