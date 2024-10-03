# 🚀 Vanguard A/B Testing: Analysis of New UI Impact on Process Completion

## 💻 Team Members
- Adrián Lardiés
- Irene Sifre

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources))
3. [Methodology](#methodology)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Performance and Insights](#performance-and-insights)
6. [Conclusions](#conclusions)
---
## ✨ Project Overview

**Context**  
As data analysts at Vanguard, our task was to evaluate the effectiveness of a new user interface designed to improve customer engagement and process completion rates through a controlled A/B test.

**The Digital Challenge**  
Vanguard recognized the evolving digital landscape and the need for a more intuitive and modern User Interface (UI) with timely in-context prompts. 
The critical question was: *Would these enhancements encourage more clients to successfully complete their online processes?*

**Experiment Conducted**  
This analysis focuses on the impact of the new user interface compared to the traditional interface, assessing its influence on customer engagement and completion rates.

The study primarily focuses on:
- Conversion rates by process step
- Error rates and user drop-offs
- Differences in behavior between users who successfully completed the process and those who didn’t
- Average time spent moving between steps for both the test and control groups

**Study Period:** March 15th, 2017 - June 20th, 2017

## 📑 Data Sources
- **Client Profiles**: Demographic data for each customer.
- **Digital Footprints**: Web interaction data including timestamps, actions taken at each step, and process completion.
- **Experiment Roster**: Data indicating whether a user was in the test or control group.

## 📚 Methodology

### **1. Data Preparation and Cleaning**
- **Merging Datasets**: Combined client demographic data with digital footprints and the experiment roster for each user to create a comprehensive dataset.
- **Missing Values**: Analyzed missing or null values and applied appropriate treatments (e.g., removal or imputation) to ensure data integrity.

### **2. User Journey Segmentation**
- **Group Classification**: Divided users into control and test groups. Further categorizing them based on their performance: those who completed the process correctly and those who did not.  
- **Performance Analysis**: Measured the time taken between each step  for both control and test groups, calculating individual averages per step as well as aggregated averages for the entire process.

### **3. Data Visualization and Analysis**
- **Visualization**: Utilized libraries such as Matplotlib and Seaborn to create visual representations of user behaviors and performance metrics, facilitating easier comparison between groups.
- **Statistical Analysis**: Conducted tests to compare averages and percentages of users who completed the process correctly step-by-step across both the control and test groups.

### **4. Testing and Implementation of Insights**
- **Hypothesis Testing**: Performed statistical tests to evaluate the significance of the observed differences in performance metrics between the two groups.
- **Implementation of Findings**: Developed actionable insights based on the analysis, such as identifying areas for interface improvement.
---
## 🔎 Exploratory Data Analysis (EDA)

The EDA phase focused on understanding key metrics and user behavior through visualizations and statistical tests.

### 1. Visualizations
- **Box Plots**: Visualized the distribution of time spent on each process step across user groups (test vs. control), identifying outliers and data spread.
- **Histograms**: Assessed the frequency distribution of key metrics, such as time spent per step and completion rates, to evaluate skewness.

### 2. Statistical Testing
- Conducted statistical tests (e.g., t-tests or Mann-Whitney U tests) to compare means and medians between the test and control groups, identifying statistically significant differences.
- Evaluated effect sizes to understand the practical significance of results, providing insight into the impact of the new interface.

## 📊 Performance and Insights
In this section, we present the graphs the project required for the results of the A/B test analysis.
Focusing on key metrics such as error rates, completion.

### **1. Average time for each step **  
![image](https://github.com/user-attachments/assets/736359e2-9c73-4fed-b062-4fb4c494a072)
![image](https://github.com/user-attachments/assets/2a71d468-9ee5-4521-a021-70679833a634)
- Lower times for each step in clients doing the correct process (confirmation of our filter), except in Confirm.
- In all steps the new version is more efficient than the old one, except in Confirm.
- Overall Improvement the UX experience in Confirm.

### **2. Error rates**  
![image](https://github.com/user-attachments/assets/030392d8-d5f3-44c3-8c28-a4ded2674e80) 
 - Error rate in Start is higher in Test, we could interpret that it is due to the fact that users were used to the classic use of the web.
 - Step 1 is notably deficient in the new version as well.
 - Step 2 and 3 performance is improved with respect to error rate in the new version.
 - Focus on improving the UX experience in Start, Step 1 and 2.

### **3. Completion rates** 
![image](https://github.com/user-attachments/assets/25fab45e-6ed0-41c7-80f7-d67ab9503193)
- Lower times for each step in clients doing the correct process (confirmation of our filter), except in Confirm.
- In all steps the new version is more efficient than the old one, except in Confirm.
- Improve the UX experience in Confirm.
## 📈 Insights and Conclusions

- **Process Completion Rates**: The new user interface (test group) showed a slight improvement in completion rates compared to the traditional interface (control group).
- However, the difference was not overwhelming, suggesting that while the UI changes had a positive effect, further enhancements might be needed to drive significant improvements.
- **Statistical Significance**: While some differences between the test and control groups were noticeable, statistical tests revealed that not all differences were significant.
- **User Segmentation**: By dividing users into those who completed the process correctly and those who didn’t, we identified that users who did not follow the steps correctly a key point in the analysis.

- **Recommendations and Limitations**:
  - While the new UI shows certain efficiency improvements, additional changes are needed in several steps.
  - A significant amount of data (20.000 observations) was rejected as it was not classified in control or test.
  - Further testing is recommended, including additional metrics or features to address the behavior of users who deviate from the expected process.
