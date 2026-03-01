import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_plots(df):
    sns.set(style="whitegrid")

    # Final grade distribution
    sns.histplot(df["finalgrade"], bins=20, kde=True)
    plt.title("Final Grade Distribution")
    plt.show()

    # Number of sessions
    sns.histplot(df["Num_Sessions"], bins=30, kde=True)
    plt.title("Number of Sessions per Student")
    plt.show()

    # Total clicks
    sns.histplot(df["Total_Clicks"], bins=30, kde=True)
    plt.title("Total Clicks per Student")
    plt.show()

    # Total time online
    sns.histplot(df["Total_Time_Online"], bins=30, kde=True)
    plt.title("Total Time Online (minutes)")
    plt.show()

    # Average time per session
    sns.histplot(df["Avg_Time_Per_Session"], bins=30, kde=True)
    plt.title("Average Time per Session (minutes)")
    plt.show()

    # Max inactivity period
    sns.histplot(df["Max_Inactivity_Period"], bins=50, kde=True)
    plt.title("Max Inactivity Period (minutes)")
    plt.show()

    
    # Correlation heatmap
    corr = df.drop(columns=["userid", "courseid"]).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Feature Correlation Matrix", fontsize=12)
    plt.show()
    