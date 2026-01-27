import nbformat

# Read the notebook
with open('grpo_for_verifiable_tasks.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Remove widget metadata
if 'widgets' in nb.metadata:
    del nb.metadata['widgets']

# Save the cleaned notebook
with open('grpo_for_verifiable_tasks.ipynb', 'w') as f:
    nbformat.write(nb, f)