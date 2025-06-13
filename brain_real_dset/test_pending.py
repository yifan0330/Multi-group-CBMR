import pickle
filepath = "data/5_Cue_Reactivity/Poisson_model/monte_carlo_group_comparison/jobs_masked_synthetic_data.pkl"
# Open the file in read-binary mode
with open(filepath, 'rb') as file:
    data = pickle.load(file)

# 'data' now contains the deserialized Python object
indices = [key for key, value in data.items() if data[key]['status'] == 'running']
print(indices)
exit()

for i in indices:
    data[i]['status'] = 'pending'

# indices = [key for key, value in data.items() if data[key]['status'] == 'running']
# print(indices)

# Open a file in write-binary ('wb') mode to save the dictionary
with open(filepath, 'wb') as file:
    pickle.dump(data, file)