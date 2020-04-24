# 481capstone
Repository for Team Null's project (CSE 481N)

## Docker Command

Recommended docker commands:

Anytime a requirement changes, run the following. Note it might take a while on the first time you run it.

```ps1
docker build . --tag 481:latest
```

When you want to develop, mount the current directory in the container using the following **powershell** command (achieved by typing `powershell` into the terminal):

```ps1
cd the/project/directory/
docker run -it -p "8888:8888" --volume "${PWD}:/home/jovyan/project" --rm 481:latest
```

Then a message similar to the following will appear:

```
Executing the command: jupyter notebook
[I 21:06:13.774 NotebookApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 21:06:14.511 NotebookApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 21:06:14.511 NotebookApp] JupyterLab application directory is /opt/conda/share/jupyter/lab  
[I 21:06:14.514 NotebookApp] Serving notebooks from local directory: /home/jovyan
[I 21:06:14.514 NotebookApp] The Jupyter Notebook is running at:
[I 21:06:14.514 NotebookApp] http://f65a2fe5d031:8888/?token=6ad75d5cccaf175b15d1411ce7186c39fbe71ac29fb994c5
[I 21:06:14.514 NotebookApp]  or http://127.0.0.1:8888/?token=6ad75d5cccaf175b15d1411ce7186c39fbe71ac29fb994c5
[I 21:06:14.514 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Connect to the address that begins with `http://127.0.0.1:8888/?` and click on `project`. You are then able to start development.