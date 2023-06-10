# PowerLyft

The sport of powerlifting involves a maximal weight single lift on three distinct exercises: the squat, bench press, and deadlift, while adhering to strict rules set out by the International Powerlifting Federation. However, in competition, judging may be subjective and the process by which an athlete is declared victorious can be unclear. Likewise in training, athletes may be unsure if their lifts are being executed to legal standards, as they may not necessarily have an experienced coach to observe them.

This project aims to provide gym-goers who practice the sport of powerlifting an application to analyse video recordings of their lifts. The analysis will include: tracing out the bar path, calculating velocity data of the bar, and feedback on whether the lift is legal. The application will also be able to classify exercises into either squat, bench press or deadlift, and identify the barbell without user input.

## Running the .exe

Ensure you are on Windows 10

Copy contents of ``videos`` to your ``Videos`` directory

Extract ``app.zip`` and navigate to the ``app`` directory

Navigate to ``app -> myApp``

Run ``MyApp.exe``



## Running source files

To run the raw source files, ensure you have ``python3``, ``venv``, and ``pip`` installed

Navigate to the ``source`` directory on your command line

Create a virtual environment using
```bash
python3 -m venv env
```

Activate the virtual environment using
```bash
.\env\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run python file
```bash
python3 app.py
```
