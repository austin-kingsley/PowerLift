## Running the .exe

Ensure you are on Windows 10

Extract ``project.zip`` and navigate to the ``project`` directory

Copy contents in ``videos`` to your ``Videos`` directory

Navigate to ``app -> dist -> myApp``

Run ``MyApp.exe``



## Running source files

To run the raw source files ensure you have ``python3``, ``venv``, and ``pip`` installed

Navigate to the ``project`` directory on your command line

Create a virtual environment using
```bash
python3 -m venv env
```

Activate the virtual environment using
```bash
.\env\Scripts\activate
```

Install dependencies (this may take a few minutes)
```bash
pip install -r requirements.txt
```

Run source python file
```bash
python3 app.py
```

Feel free play around by modifying source files