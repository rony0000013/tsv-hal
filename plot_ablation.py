import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('ablation.csv')

# Forward fill the 'Models' column
df['Models'] = df['Models'].ffill()

# 1. Average Heatmap (Existing)
transfer_df = df.groupby('Language Trained on')[['English', 'Hindi', 'Bengali']].mean()
transfer_df = transfer_df.reindex(['English', 'Hindi', 'Bengali'])
transfer_df = transfer_df[['English', 'Hindi', 'Bengali']]

plt.figure(figsize=(8, 6))
sns.heatmap(transfer_df, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'Mean AUROC'})
plt.title('Cross-Lingual Transferability of TSV (Average Across Models)', fontsize=14)
plt.ylabel('Training Language', fontsize=12)
plt.xlabel('Testing Language', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig('ablation_chart.pdf')
plt.close()

# Calculate Differences (Delta = Test Language AUROC - Original AUROC)
# df['English Diff'] = df['English'] 
# df['Hindi Diff'] = df['Hindi']
# df['Bengali Diff'] = df['Bengali']

# 2. Difference Heatmap
# Average the difference across all models
diff_heatmap_df = df.groupby('Language Trained on')[['English Diff', 'Hindi Diff', 'Bengali Diff']].mean()
diff_heatmap_df = diff_heatmap_df.reindex(['English', 'Hindi', 'Bengali'])
diff_heatmap_df.columns = ['English', 'Hindi', 'Bengali'] # Rename columns for the plot

plt.figure(figsize=(8, 6))
# Use a diverging colormap centered at 0
sns.heatmap(diff_heatmap_df, annot=True, cmap='RdBu', center=0, fmt='.3f', cbar_kws={'label': 'Mean AUROC Delta'})
plt.title('Performance Delta: Tested Language vs Original AUROC (Average)', fontsize=14)
plt.ylabel('Training Language', fontsize=12)
plt.xlabel('Testing Language', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig('ablation_diff_heatmap.pdf')
plt.close()

# 3. Bar Chart (Delta per Model and Train Language)
# Create a new column for the x-axis labels: Model (Trained Language)
df['Model_TrainLang'] = df['Models'] + ' (' + df['Language Trained on'] + ')'

# Set up the bar chart
x = np.arange(len(df))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 6))
rects1 = ax.bar(x - width, df['English Diff'], width, label='English Diff', color='#d62728')
rects2 = ax.bar(x, df['Hindi Diff'], width, label='Hindi Diff', color='#2ca02c')
rects3 = ax.bar(x + width, df['Bengali Diff'], width, label='Bengali Diff', color='#f39c12')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Delta', fontsize=12)
ax.set_xlabel('Model (Trained Language)', fontsize=12)
ax.set_title('Performance Delta: Tested Language vs Original AUROC', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df['Model_TrainLang'], rotation=45, ha='right', fontsize=14)
ax.legend(fontsize=14)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.axhline(0, color='black', linewidth=0.8) # Add a horizontal line at y=0

plt.tight_layout()
plt.savefig('ablation_bar_chart.pdf')
plt.close()

print("Saved ablation_chart.pdf, ablation_diff_heatmap.pdf, and ablation_bar_chart.pdf")
