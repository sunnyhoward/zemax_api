# Zemax Api

In this we'll develop codes for using optimisation techniques along with Zemax's api.

**If using the API for the first time**, go to zemax Optics Studio then:  **`programming tab/zos-api.net Application Builders/python/interactive extension`**. This will create the zos connection file.  

To connect python to the API from there onwards, we go to the **`programming tab/zos-api.NET Applications/Interactive Extension`**. 

Within jupyter notebooks we can connect to the api by running: 

```
%run -i "...\Zemax\ZOS-API Projects\PythonZOSConnection\PythonZOSConnection.py"
```
Where you replace `...` with the path to your Zemax folder.