import pickle

# Path to the .pkl file
filename = "data/1_Social_Processing/resampling/jobs_masked_synthetic_data.pkl"

# Open the file in binary read mode
with open(filename, 'rb') as file:
    # Load the data from the file
    jobs = pickle.load(file)

failed_job = jobs[370]
print(failed_job)