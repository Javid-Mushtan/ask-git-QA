import os

def load_files(repo_path):
    documents = []

    for roots, dirs, files in os.walk(repo_path):
