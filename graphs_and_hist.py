import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel chart into a Pandas DataFrame
df = pd.read_csv("Training_Dataset.csv")

df = pd.DataFrame((df['having_IP_Address'], df['URL_Length'], df['Shortening_Service'],
                   df['having_At_Symbol'], df['double_slash_redirecting'], df['Prefix_Suffix'],
                   df['having_Sub_Domain'], df['SSLfinal_State'], df['Domain_registration_length'],
                   df['Favicon'], df['HTTPS_token'], df['Request_URL'], df['URL_of_Anchor'], df['Links_in_tags'],
                   df['Redirect'], df['on_mouseover'], df['RightClick'], df['popUpWindow'], df['Iframe'],
                   df['age_of_domain'], df['DNSRecord'], df['Result'])).T

# Loop through each column (excluding "Result") and plot a graph
for col in df.columns[:-1]:
    if col not in ['having_Sub_Domain', 'Shortening_Service']:
        plt.scatter(df[col], df["Result"])
        plt.xlabel(col)
        plt.ylabel("Result")
        plt.title(f"{col} vs. Result")
        plt.show()

# Plot a histogram of the "Result" column
plt.hist(df["Result"])
plt.xlabel("Result")
plt.ylabel("Frequency")
plt.title("Distribution of Result")
plt.show()

# Plot a histogram of the "URL_Length" column
plt.hist(df["URL_Length"])
plt.xlabel("URL_Length")
plt.ylabel("Frequency")
plt.title("Distribution of URL Length")
plt.show()
