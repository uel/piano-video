- segmentations
  - [ ] Add proper logging/exceptions
  - [ ] Make into class?
  - [x] Add parameters
  - [ ] Add datatypes 
  - [ ] edges can be found using reusable function
  - [ ] Caching (not worth saving masks, just segments, maybe piano location and white key thresh?)

-  PianoVideo
  - [ ] Add image scaling to config
  - [ ] add other parameters

- processing
  - [ ] make it possible to run on a folder of videos for a specific step/steps of the pipeline, force rerun

- fingerings
  - [ ] generate estimated fingerings and fixed midi using a heuristic

- tests
  - [ ] clean up tests, remove unused functions and make sure run

- [ ] double check videos that are flagged as having no sections ( short flowkey videos )
- [ ] add progress bars / info
- [ ] add setup script for downloading models and data?
- [ ] add dockerfile
- [ ] create a readme and gifs for the repo
- [ ] install file
- [ ] __init__.py file

# 2023-12-07
- Background should be capped to 640x640
- All coordinates should be in float format

- [x] yolo detection
  - Had to optimize the threshold for the detection, training threshold was located in data
  - Stretching is a problem because videos have different aspect ratios
  - Now has 100% detection accuracy on the test and train set

# 2023-11-07
- Speed issues in sections
  - [x] Don't compute hands/midi for non-section areas
  - [x] Two-part algorithm
  - [ ] Feature matching? Speed OK already

- Fine-tune top and bottom after left and right are found
  - [x] Use two masks with colors sampled from around white_y and black_y

# 2023-11-02
- Last key is not segmented right
  - A# is the last key but only one key is shown to the right
- Top & bottom wrong on wide Flowkey videos
  - Scaled was wrong because matcher was re-used
- Remove background from demo
  - Background subtraction + change in draw hand style
- Average brightness to get an accurate estimate of white_keys
  - Averaging across the whole width works best! Issue if padding is dark or isn't part of the piano
- Fix black bars
  - Probably fixed for at least Flowkey by counting NaNs

# 2023-11-01
- Warnings/info: nan background, no sections
- Hand detection accuracy is much better - background could rely on both hands being detected
  - Background better but still not perfect - tracking nan value needed
- Zoom for the best template match is not wide enough, testing best template match
  - Zoom at the beginning of the video, affects all triad videos from Flowkey
    - Solved by not setting best_scale_factor until stable
- Segmentation is not working for most other videos
  - Color thresholds differ by image - dynamic thresholds or preprocessing of the image
    - HSL color space fixed

# 2023-10-21
- Caching and central place to enable single video usage
  - Video
  - Audio split
  - Background
  - Keyboard location, segmentations
  - Extracted MIDI
  - Hand tracking

# 2023-10-09
- Solution to background removal
  - Iterating over every frame is slow
  - Accuracy of hand detection is low
  - `processing.py` should start by pressing F5
  - Results from KeyBoard frames should be cached if no keyboard exists
  - Validate contains keyboard phase on all videos

# 2023-09-07 (Meeting Notes)
Model na nalezení hraničních bodů kláves a detekci kláves
Předchozí krok, Získání pozadí be rukou, nutné pro segmentaci
Použití mediánu a modelu na sledování rukou ( obrázek ) k odstranění popředí
Urychlení ( trvá 1 minutu / video ) přesnost detekce rukou, velikost paměti, různorodost snímků, ořezaní snímků
Předměty MFF

# 2023-09-03
- Keyboard detection (template matching) fails on Piano Man - GP09dN
- Accuracy of hand tracking is not good enough for extracting the background, setting if hands == 2 fixes the issue but will fail where no hand or one hand is present
- Hand tracking accuracy is higher with smaller dt, but even sample also needed. The whole video needs to be processed.
- Precomputing hand landmarks will improve speed and accuracy of testing
- Change in location disrupts estimated keyboard background, e.g., Piano Man - Ju6BAMJugC4, QZiVSDA8zhk, could be fixed by monitoring changes in location/size using template matching before
- Monitor and balance NaN values on the horizontal axis; can we skip frames?
- Maybe audio/notes could be integrated for better accuracy
- Detection doesn't work for Jane videos
- Before:
  - Median works better than mean
  - Have to use hand tracking to 'remove hands' before background estimation
  - `find_interval` for efficiently finding bounds with a bool function (Contains keyboard)
  - NN detection has almost the same accuracy as template matching