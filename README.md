## PowerLyft
The sport of powerlifting involves a maximal weight single lift on three distinct exercises: the squat, bench press, and deadlift, while adhering to strict rules set out by the International Powerlifting Federation (IPF). However, in competition, judging may be subjective and the process by which an athlete is declared victorious can be unclear. Likewise in training, athletes may be unsure if their lifts are being executed to legal standards, as they may not necessarily have an experienced coach to observe them.

This project involves the creation of an application for powerlifters to analyse video recordings of their lifts. The application features the ability to classify exercises, automatically identify the weight plate at the end of the barbell, classify the validity of the lift, and more. Our exercise classifier achieved an accuracy of 91% in correctly determining the exercise being performed, and our weight plate detector an accuracy of 100% with 95% precision. Our application was also able to classify exercises into ‘legal’ or ‘illegal’ according to IPF standards with an accuracy of 96%, henceforth sounding promising as a tool for powerlifters to use in training, and federations to use when judging lifts during competition

## Running source files
Clone this repository to your machine

Ensure you have ``python3``, ``venv``, and ``pip`` installed

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

And enjoy!
