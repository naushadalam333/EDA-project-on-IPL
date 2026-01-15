#!/usr/bin/env python
# coding: utf-8

# # Perform Exploratory Data Analysis on 'Indian Premiere League'
# 
# Objective:
# 
# ● Perform ‘Exploratory Data Analysis’ on dataset ‘Indian Premier League’
# 
# ● As a sports analysts, find out the most successful teams, players and factors contributing win or loss of a team.
# 
# ● Recommend teams or players for product endorsements by a company.

# In[163]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# In[164]:


# Load features and labels datasets
matches_df = pd.read_csv("D:\DATASETS\matches.csv")
deliveries_df = pd.read_csv("D:\DATASETS\deliveries.csv\deliveries.csv")


# In[165]:


matches_df.head()


# In[166]:


deliveries_df.head()


# In[167]:


matches_df.info()


# In[168]:


deliveries_df.info()


# In[169]:


matches_df["umpire3"].isnull().sum()


# In[170]:


matches_df["umpire3"].tail(10)


# In[171]:


matches_df.describe()


# In[172]:


# Matches we have got in the dataset 
matches_df['id'].max()


# In[173]:


# Seasons we have got in the dataset
matches_df['season'].unique()


# In[174]:


#Team won by Maximum Runs
matches_df.iloc[matches_df['win_by_runs'].idxmax()]


# In[175]:


matches_df.iloc[matches_df['win_by_runs'].idxmax()]['winner']


# In[176]:


#Team won by Maximum Wickets
matches_df.iloc[matches_df['win_by_wickets'].idxmax()]['winner']


# In[177]:


#Team won by minimum runs
matches_df.iloc[matches_df[matches_df['win_by_runs'].ge(1)].win_by_runs.idxmin()]['winner']


# In[178]:


#Team won by Minimum Wickets
matches_df.iloc[matches_df[matches_df['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]


# In[179]:


matches_df.iloc[matches_df[matches_df['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]['winner']


# Season Which had most number of matches

# In[180]:


import matplotlib.pyplot as plt

# Assuming 'matches_df' is your DataFrame with the 'season' column
plt.figure(figsize=(12, 6))
matches_df['season'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Number of Matches per Season')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#In 2013, we have the most number of matches


# In[181]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'matches_df' is your DataFrame with the 'winner' column
plt.figure(figsize=(12, 6))
data = matches_df['winner'].value_counts()
sns.barplot(y=data.index, x=data, orient='h', palette='viridis')  # Change 'viridis' to your desired palette
plt.title('Number of Wins by Team')
plt.xlabel('Number of Wins')
plt.ylabel('Team')
plt.show()
#Mumbai Indians are the winners in most of the matches


# Top Player of the match winners

# In[182]:


top_players = matches_df.player_of_match.value_counts()[:10]
#sns.barplot(x="day", y="total_bill", data=df)
fig, ax = plt.subplots(figsize=(15,8))
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v', palette="inferno");
plt.show()
#CH Gayle is the most Successful player in all match winners


# Number of matches in each venue:

# In[183]:


plt.figure(figsize=(12,6))
sns.countplot(x='venue', data=matches_df)
plt.xticks(rotation='vertical')
plt.show()
#There are quite a few venues present in the data with "M Chinnaswamy Stadium" being the one with most number of matches followed by "Eden Gardens


# Number of matches played by each team:

# In[184]:


temp_df = pd.melt(matches_df, id_vars=['id','season'], value_vars=['team1', 'team2'])

plt.figure(figsize=(12,6))
sns.countplot(x='value', data=temp_df)
plt.xticks(rotation='vertical')
plt.show()
#MI and RCB played most of the matches


# In[185]:


#Number of wins per team: MI again leads
plt.figure(figsize=(12,6))
sns.countplot(x='winner', data=matches_df)
plt.xticks(rotation=90)
plt.show()


# Champions each season:
# 
# Now let us see the champions in each season.

# In[186]:


temp_df = matches_df.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)
temp_df


# Toss decision:
# 
# Let us see the toss decisions taken so far.

# In[187]:


temp_series = matches_df.toss_decision.value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['silver', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage indicator")
plt.show()


# the toss decisions are made to field first. Now let us see how this decision varied over time.

# In[188]:


plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='toss_decision', data=matches_df)
plt.xticks(rotation='vertical')
plt.show()
#2016 season, most of the toss decisions are to field first.


# In[189]:


# Since there is a very strong trend towards batting second let us see the win percentage of teams batting second.
num_of_wins = (matches_df.win_by_wickets>0).sum()
num_of_loss = (matches_df.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['silver', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage batting second")
plt.show()


# Top Umpires:

# In[190]:


# Define autolabel function
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Your data
temp_df = pd.melt(matches_df, id_vars=['id'], value_vars=['umpire1', 'umpire2'])
temp_series = temp_df['value'].value_counts()[:10]

# Set a colorful palette
colors = sns.color_palette('viridis', len(temp_series))

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
rects = ax.bar(temp_series.index, temp_series, color=colors)
ax.set_ylabel("Count")
ax.set_title("Top Umpires")

# Call the autolabel function
autolabel(rects, ax)

plt.xticks(rotation='vertical')
plt.show()


# Dharmasena seems to be the most sought after umpire for IPL matches followed by Ravi. 

# deliveries Data Set

# In[191]:


deliveries_df.head()


# Batsman analysis:
# Let us start our analysis with batsman. Let us first see the ones with most number of IPL runs under their belt.

# In[192]:


temp_df = deliveries_df.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='skyblue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top run scorers in IPL")
ax.set_xlabel('Batsmane Name')
autolabel(rects, ax)
plt.show()


# Virat Kohli is leading the chart followed closely by Raina. Gayle is the top scorer among foreign players.

# In[193]:


# Now let us see the players with more number of boundaries in IPL.
temp_df = deliveries_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='lightskyblue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of boundaries.!",fontsize = 10)
autolabel(rects, ax)
plt.show()


# Gambhir is way ahead of others - almost 60 boundaries more than Kohli.! Nice to Sachin in the top 10 list :)

# In[194]:


# Now let us check the number of 6's
temp_df = deliveries_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='m')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of sixes.!")
ax.set_xlabel('Batsmane Name')
autolabel(rects, ax)
plt.show()


# There you see the big man. Gayle, the unassailable leader in the number of sixes.
# 
# Raina is third in both number of 4's and 6's

# In[195]:


# Now let us see the batsman who has played the most number of dot balls.
temp_df = deliveries_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='c')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of dot balls.!")
ax.set_xlabel('Batsmane Name')
autolabel(rects, ax)
plt.show()


# In[196]:


# Let us check the percentage distribution now.
def balls_faced(x):
    return len(x)

def dot_balls(x):
    return (x==0).sum()

temp_df = deliveries_df.groupby('batsman')['batsman_runs'].agg([balls_faced, dot_balls]).reset_index()
temp_df = temp_df.loc[temp_df.balls_faced>200,:]
temp_df['percentage_of_dot_balls'] = (temp_df['dot_balls'] / temp_df['balls_faced'])*100.
temp_df = temp_df.sort_values(by='percentage_of_dot_balls', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

fig, ax1 = plt.subplots(figsize=(15,8))
ax2 = ax1.twinx()
labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
rects = ax1.bar(ind, np.array(temp_df['dot_balls']), width=width, color='brown')
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Count of dot balls", color='brown')
ax1.set_title("Batsman with highest percentage of dot balls (balls faced > 200)")
ax2.plot(ind+0.45, np.array(temp_df['percentage_of_dot_balls']), color='b', marker='o')
ax2.set_ylabel("Percentage of dot balls", color='b')
ax2.set_ylim([0,100])
ax2.grid(b=False)
plt.show()


# Batsman with more than 300 balls faced in taken and the ones with higher percentage of dot balls are seen.
# It is interesting to see Ganguly with more than 1000 balls and nearly half of them are dot balls.
# It is surprising to see names like Jayasuriya and Gibbs in there.!

# Bowler Analysis
# Now let us see the bowlers who has bowled most number of balls in IPL.

# In[197]:


temp_df = deliveries_df.groupby('bowler')['ball'].agg('count').reset_index().sort_values(by='ball', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['bowler'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['ball']), width=width, color='cyan')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Bowlers - Number of balls bowled in IPL")
ax.set_xlabel('Bowler Names')
autolabel(rects, ax)
plt.show()


# Harbhajan Singh is the the bowler with most number of balls bowled in IPL matches.
# Now let us see the bowler with more number of dot balls.

# In[198]:


temp_df = deliveries_df.groupby('bowler')['total_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='total_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['bowler'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['total_runs']), width=width, color='lightskyblue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Bowlers - Number of dot balls bowled in IPL")
ax.set_xlabel('Bowler Names')
autolabel(rects, ax)
plt.show()


# Pravin Kumar is the one with more number of dot balls followed by Steyn and Malinga

# In[199]:


# Now let us see the bowlers who has bowled more number of extras in IPL.
temp_df = deliveries_df.groupby('bowler')['extra_runs'].agg(lambda x: (x>0).sum()).reset_index().sort_values(by='extra_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['bowler'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['extra_runs']), width=width, color='g')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Bowlers with more extras in IPL")
ax.set_xlabel('Bowler Names')
autolabel(rects, ax)
plt.show()


# Malinga tops the chart with 221 extra runs followed by Pravin Kumar.

# In[200]:


# Now let us see most common dismissal types in IPL.
plt.figure(figsize=(12,6))
sns.countplot(x='dismissal_kind', data=deliveries_df)
plt.xticks(rotation='vertical')
plt.show()


# Caught is the most common dismissal type in IPL followed by Bowled. 
# There are very few instances of hit wicket as well. 
# 'Obstructing the field' is one of the dismissal type as well in IPL.!

# In[201]:


# Create a function to categorize overs into phases
def categorize_overs(over):
    if over <= 6:
        return 'Powerplay'
    elif 6 < over <= 15:
        return 'Middle Overs'
    else:
        return 'Death Overs'

# Apply the function to create a new column 'over_phase'
deliveries_df['over_phase'] = deliveries_df['over'].apply(categorize_overs)

# Visualize player performance in different phases
plt.figure(figsize=(12, 6))
sns.countplot(x='over_phase', hue='batsman', data=deliveries_df, palette='viridis')
plt.title('Player Performance in Different Phases of the Game')
plt.xlabel('Overs Phase')
plt.ylabel('Number of Runs')
plt.show()


# In[202]:


# Display first few rows to understand the data
print("Features:")
print(matches_df.head())
print("\nLabels:")
print(deliveries_df.head())


# In[203]:


# Check basic info about the features dataset
print("Features Info:")
print(X.info())

# Check basic info about the labels dataset
print("\nLabels Info:")
print(y.info())


# In[204]:


# Concatenate features and labels for joint analysis
full_data = pd.concat([X, y], axis=1)

# Correlation between features and labels
correlation_matrix = full_data.corr()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix between Features and Labels')
plt.show()


# In[205]:


# Assuming you loaded 'Features' DataFrame from a CSV file
Features = pd.read_csv("D:\DATASETS\matches.csv")

# Now you can use the code to handle missing values
Features['city'].fillna(Features['city'].mode()[0], inplace=True)
Features['winner'].fillna(Features['winner'].mode()[0], inplace=True)
Features['player_of_match'].fillna(Features['player_of_match'].mode()[0], inplace=True)
Features['umpire1'].fillna(Features['umpire1'].mode()[0], inplace=True)
Features['umpire2'].fillna(Features['umpire2'].mode()[0], inplace=True)

# Drop 'umpire3' column as it has no non-null values
Features.drop('umpire3', axis=1, inplace=True)

# Verify changes
print(Features.info())


# In[206]:


# Load the 'Labels' DataFrame from the CSV file
Labels = pd.read_csv("D:/DATASETS/deliveries.csv/deliveries.csv")

# Handling missing values in 'player_dismissed', 'dismissal_kind', 'fielder'
Labels['player_dismissed'].fillna('Not Dismissed', inplace=True)
Labels['dismissal_kind'].fillna('Not Dismissed', inplace=True)
Labels['fielder'].fillna('Not Dismissed', inplace=True)

# Verify changes
print(Labels.info())


# In[207]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Assuming 'winner' is your target variable
y = Features['winner']

# Dropping non-numeric and non-relevant columns for this analysis
X = Features.drop(['id', 'date', 'winner', 'player_of_match', 'umpire1', 'umpire2'], axis=1)

# Handling categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X)

# Handling missing values using mean imputation
X_encoded = X_encoded.fillna(X_encoded.mean())

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Fitting the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Displaying the top 10 important features
feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
top_features = feature_importances.nlargest(10)
print("\nTop 10 Important Features:")
print(top_features)

# Plotting feature importance
plt.figure(figsize=(12, 6))
top_features.plot(kind='barh')
plt.title('Top 10 Features Importance')
plt.xlabel('Importance')
plt.show()


# In[208]:


# Assuming 'model' is your trained RandomForestClassifier model
# You may replace it with 'best_model' if you performed hyperparameter tuning

# Make predictions on the entire dataset
all_predictions = model.predict(X_encoded)

# Create a DataFrame with actual and predicted labels
predictions_df = pd.DataFrame({'Actual': y, 'Predicted': all_predictions})

# Filter rows where the predicted label is different from the actual label
misclassifications = predictions_df[predictions_df['Actual'] != predictions_df['Predicted']]

# Display the misclassifications and relevant features
print("Misclassifications:")
print(misclassifications)

# If you want to explore specific instances, you can filter the DataFrame further
# Example: Display details of the first misclassification
first_misclassification = misclassifications.iloc[0]
print("\nDetails of the First Misclassification:")
print(first_misclassification)

# Alternatively, you can investigate features of misclassifications with a specific class
# Example: Display details of misclassifications where the actual label is 'TeamA' and predicted label is 'TeamB'
teamA_vs_teamB_misclassifications = misclassifications[
    (misclassifications['Actual'] == 'TeamA') & (misclassifications['Predicted'] == 'TeamB')
]
print("\nDetails of Misclassifications (TeamA vs. TeamB):")
print(teamA_vs_teamB_misclassifications)


# In[ ]:





# In[ ]:




