import pandas as pd
import os
import re

# Define the directories
directories = ['metrics/trials_album_split', 'metrics/trials_songs_split']

# Iterate through each directory
for directory in directories:

    # Prepare the data for the summary CSV
    summary_data = {'Average F1': [], 'Maximum F1': []}
    summary_data_pooled = {'Average F1': [], 'Maximum F1': []}

    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate through each file
    for file in csv_files:

        # Extract the number from the filename
        numbers = re.findall(r'\d+', file)

        # If no number is found, skip this file
        if not numbers:
            continue

        num = int(numbers[0])

        # Load the CSV file
        df = pd.read_csv(os.path.join(directory, file))

        # Calculate the average and maximum F1 score
        avg_f1 = df['f1-score'].mean()
        max_f1 = df['f1-score'].max()

        # Check if it's a pooled file or a normal file
        if "pooled" in file:
            summary_data_pooled['Average F1'].append((num, avg_f1))
            summary_data_pooled['Maximum F1'].append((num, max_f1))
        else:
            summary_data['Average F1'].append((num, avg_f1))
            summary_data['Maximum F1'].append((num, max_f1))

    # Sort the data by the number
    for data in (summary_data, summary_data_pooled):
        data['Average F1'].sort(key=lambda x: x[0])
        data['Maximum F1'].sort(key=lambda x: x[0])

    # Prepare the data for the DataFrame
    data = {
        'Average/Maximum F1': ['Average F1', 'Maximum F1', 'Average F1 (Pooled)', 'Maximum F1 (Pooled)'],
        **{str(num): [avg, max, avg_pooled, max_pooled] for ((num, avg), (_, max), (num_pooled, avg_pooled), (_, max_pooled)) in zip(summary_data['Average F1'], summary_data['Maximum F1'], summary_data_pooled['Average F1'], summary_data_pooled['Maximum F1'])}
    }

    # Create the DataFrame
    summary_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    summary_df.to_csv(os.path.join(directory, 'summary.csv'), index=False)
