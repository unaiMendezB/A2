import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_and_save(file_path):
    # Read the data from the file
    data = pd.read_csv(file_path, sep='\t', header=None)

    # Assign columns to variables
    x = data[0]
    y = data[1]
    colors = data[2]

    # Map 0 to 'green' and 1 to 'red'
    color_map = {0: 'green', 1: 'red'}
    colors = colors.map(color_map)

    # Create the plot
    plt.scatter(x, y, c=colors)

    # Add title
    title = os.path.basename(file_path).split('.')[0]
    plt.title(title)

    # Save the plot as an image
    plt.savefig(file_path.replace('.txt', '.png'))

    # Display the plot
    plt.show()

# Call the function for each file
plot_and_save('A2-ring/A2-ring-separable.txt')
plot_and_save('A2-ring/A2-ring-merged.txt')
plot_and_save('A2-ring/A2-ring-test.txt')
