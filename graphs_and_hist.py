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
# Loop through each column (excluding "Result") and plot a graph
for col in df.columns[:-1]:
    plt.plot(df[col], df["Result"], linestyle="-")
    plt.xlabel(col)
    plt.ylabel("Result")
    plt.show()
